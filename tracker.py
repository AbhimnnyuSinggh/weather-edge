"""
tracker.py — Database Operations, Bias EWMA, Resolution Detection

All database operations: init, schema migration, store/query helpers,
resolution processing, bias updates, circuit breakers, reports, exports.
"""

import asyncio
import csv
import io
import json
import logging
import os
import uuid
import zipfile
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import pytz

logger = logging.getLogger("tracker")

# ---------------------------------------------------------------------------
# Global DB pool
# ---------------------------------------------------------------------------
_pool: Optional[asyncpg.Pool] = None


async def init_db() -> asyncpg.Pool:
    """Connect to PostgreSQL and run schema migration."""
    global _pool
    database_url = os.environ["DATABASE_URL"]
    # asyncpg needs 'postgresql://' scheme
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
    await _run_migrations()
    logger.info("Database initialised (pool size 2-10)")
    return _pool


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _pool


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------
async def _run_migrations():
    """Execute schema.sql to create tables if they don't exist."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(sql)
    logger.info("Schema migration complete (11 tables)")


# ---------------------------------------------------------------------------
# Station initialisation from config
# ---------------------------------------------------------------------------
async def init_stations(stations_cfg: dict):
    """Insert or update station rows from config.yaml."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        for icao, cfg in stations_cfg.items():
            await conn.execute(
                """
                INSERT INTO stations (icao, city, country, latitude, longitude,
                    timezone, unit, wunderground_url, is_coastal,
                    bias_ecmwf, bias_gfs, bias_icon, bias_nws)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT (icao) DO UPDATE SET
                    city=EXCLUDED.city, country=EXCLUDED.country,
                    latitude=EXCLUDED.latitude, longitude=EXCLUDED.longitude,
                    timezone=EXCLUDED.timezone, unit=EXCLUDED.unit,
                    wunderground_url=EXCLUDED.wunderground_url,
                    is_coastal=EXCLUDED.is_coastal
                """,
                icao,
                cfg["city"],
                cfg.get("country"),
                cfg["lat"],
                cfg["lon"],
                cfg["timezone"],
                cfg.get("unit", "C"),
                cfg.get("wunderground_url"),
                cfg.get("is_coastal", False),
                cfg.get("starting_bias", {}).get("ecmwf", 0.0),
                cfg.get("starting_bias", {}).get("gfs", 0.0),
                cfg.get("starting_bias", {}).get("icon", 0.0),
                cfg.get("starting_bias", {}).get("nws", 0.0),
            )
    logger.info("Stations initialised: %s", list(stations_cfg.keys()))


# ---------------------------------------------------------------------------
# METAR storage
# ---------------------------------------------------------------------------
async def store_metar(station: str, observed_at: datetime, temp_c: float,
                      temp_f: float, dewpoint_c: float, wind_dir: int,
                      wind_speed_kt: int, wind_gust_kt: Optional[int],
                      visibility_m: Optional[float],
                      t_group_temp_c: Optional[float], raw_metar: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO metar_readings
                (station, observed_at, temp_c, temp_f, dewpoint_c,
                 wind_dir, wind_speed_kt, wind_gust_kt, visibility_m,
                 t_group_temp_c, raw_metar)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (station, observed_at) DO NOTHING
            """,
            station, observed_at, temp_c, temp_f, dewpoint_c,
            wind_dir, wind_speed_kt, wind_gust_kt, visibility_m,
            t_group_temp_c, raw_metar,
        )


async def get_recent_metar(station: str, hours: int = 12) -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM metar_readings
            WHERE station=$1 AND observed_at > NOW() - INTERVAL '%s hours'
            ORDER BY observed_at DESC
            """ % hours,
            station,
        )


async def get_today_metar(station: str, station_tz: str) -> List[asyncpg.Record]:
    """Return today's METAR readings for a station in its local timezone."""
    tz = pytz.timezone(station_tz)
    now_local = datetime.now(tz)
    start_of_day_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_of_day_local.astimezone(pytz.utc).replace(tzinfo=None)
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM metar_readings
            WHERE station=$1 AND observed_at >= $2
            ORDER BY observed_at ASC
            """,
            station, start_utc,
        )


# ---------------------------------------------------------------------------
# Forecast storage
# ---------------------------------------------------------------------------
async def store_forecast(station: str, target_date: date, model_name: str,
                         raw_high_c: float, raw_high_f: float,
                         bias_corrected_c: float, bias_corrected_f: float):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO model_forecasts
                (station, target_date, model_name, raw_high_c, raw_high_f,
                 bias_corrected_c, bias_corrected_f)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT DO NOTHING
            """,
            station, target_date, model_name,
            raw_high_c, raw_high_f, bias_corrected_c, bias_corrected_f,
        )


async def get_latest_forecasts(station: str, target_date: date) -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT DISTINCT ON (model_name)
                   station, target_date, model_name,
                   raw_high_c, raw_high_f,
                   bias_corrected_c, bias_corrected_f, fetched_at
            FROM model_forecasts
            WHERE station=$1 AND target_date=$2
            ORDER BY model_name, fetched_at DESC
            """,
            station, target_date,
        )


# ---------------------------------------------------------------------------
# Market snapshot storage
# ---------------------------------------------------------------------------
async def store_market_snapshot(market_data: dict):
    """Store all bin prices.  Only insert if any price changed ≥1¢ since last."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        for key, group in market_data.items():
            # group is a MarketGroup dataclass
            for mbin in group.bins:
                last = await conn.fetchrow(
                    """
                    SELECT yes_price FROM market_snapshots
                    WHERE market_id=$1 ORDER BY captured_at DESC LIMIT 1
                    """,
                    mbin.market_id,
                )
                if last and abs((last["yes_price"] or 0) - mbin.yes_price) < 0.01:
                    continue  # no meaningful change
                await conn.execute(
                    """
                    INSERT INTO market_snapshots
                        (market_id, station, target_date, bin_label,
                         bin_low, bin_high, yes_price, volume)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    """,
                    mbin.market_id,
                    group.station,
                    group.target_date,
                    mbin.bin.label,
                    mbin.bin.low,
                    mbin.bin.high,
                    mbin.yes_price,
                    mbin.volume_24h,
                )


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------
async def store_trade(alert: dict):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO trades
                (station, target_date, bin_label, side, trade_type,
                 entry_price, shares, cost, confidence_score,
                 confidence_components, ev_at_entry, alert_id)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            """,
            alert["station"],
            alert["target_date"],
            alert["bin_label"],
            alert["side"],
            alert["trade_type"],
            alert["entry_price"],
            alert["shares"],
            alert["cost"],
            alert.get("confidence_score"),
            json.dumps(alert.get("confidence_components", {})),
            alert.get("ev"),
            alert.get("alert_id"),
        )


async def get_open_trades() -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT * FROM trades WHERE resolved=FALSE ORDER BY opened_at DESC"
        )


async def get_trades_for_station_date(station: str, target_date: date) -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT * FROM trades WHERE station=$1 AND target_date=$2",
            station, target_date,
        )


async def get_capital_summary() -> dict:
    """
    Calculate capital status from trades table.
    Returns: {total_profit, total_loss, deployed, open_positions: [...]}
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Profits from resolved winning trades
        profit_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(profit_loss), 0) AS total "
            "FROM trades WHERE resolved=TRUE AND outcome='win'"
        )
        total_profit = float(profit_row["total"]) if profit_row else 0.0

        # Losses from resolved losing trades (profit_loss is negative)
        loss_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(ABS(profit_loss)), 0) AS total "
            "FROM trades WHERE resolved=TRUE AND outcome='loss'"
        )
        total_loss = float(loss_row["total"]) if loss_row else 0.0

        # Currently deployed (open trades)
        deployed_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(cost), 0) AS total "
            "FROM trades WHERE resolved=FALSE"
        )
        deployed = float(deployed_row["total"]) if deployed_row else 0.0

        # Open positions list
        open_rows = await conn.fetch(
            "SELECT * FROM trades WHERE resolved=FALSE ORDER BY opened_at DESC"
        )
        open_positions = []
        for r in open_rows:
            open_positions.append({
                "market_id": r.get("market_id", r.get("id", "")),
                "station": r["station"],
                "target_date": r["target_date"],
                "bin_label": r["bin_label"],
                "side": r["side"],
                "shares": float(r["shares"]),
                "entry_price": float(r["entry_price"]),
                "cost": float(r["cost"]),
            })

    return {
        "total_profit": total_profit,
        "total_loss": total_loss,
        "deployed": deployed,
        "open_positions": open_positions,
    }


async def get_total_trades_count() -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM trades WHERE resolved=TRUE")
        return row["cnt"] if row else 0


async def get_today_trades() -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT * FROM trades WHERE opened_at::date = CURRENT_DATE ORDER BY opened_at DESC"
        )


async def log_manual_trade(trade: dict) -> int:
    """Insert a manually-entered trade and return its ID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO trades
                (station, target_date, bin_label, side, trade_type,
                 entry_price, shares, cost)
            VALUES ($1, CURRENT_DATE, $2, $3, 'manual', $4, $5, $6)
            RETURNING id
            """,
            trade["station"],
            trade["bin_label"],
            trade["side"],
            trade["entry_price"],
            trade["shares"],
            trade["cost"],
        )
        trade_id = row["id"]
        logger.info("Manual trade logged: id=%d %s %s %s",
                     trade_id, trade["station"], trade["side"], trade["bin_label"])
        return trade_id


async def resolve_trade(trade_id: int, outcome: str,
                         actual_high: Optional[float] = None) -> Optional[dict]:
    """Manually resolve a trade by ID. Returns result dict or None."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        trade = await conn.fetchrow("SELECT * FROM trades WHERE id=$1", trade_id)
        if not trade:
            return None

        cost = float(trade["cost"])
        shares = float(trade["shares"])

        if outcome == "win":
            payout = shares  # YES pays $1/share on win
            profit_loss = payout - cost
        elif outcome == "loss":
            payout = 0.0
            profit_loss = -cost
        else:  # push/void
            payout = cost
            profit_loss = 0.0

        await conn.execute(
            """
            UPDATE trades SET
                resolved=TRUE, resolved_at=NOW(),
                outcome=$1, payout=$2, profit_loss=$3
            WHERE id=$4
            """,
            outcome, payout, profit_loss, trade_id,
        )

        logger.info("Trade #%d resolved: %s pnl=%.2f", trade_id, outcome, profit_loss)

        # Update daily summary
        await _update_daily_summary_on_resolution(outcome, profit_loss, payout, cost)

        return {"trade_id": trade_id, "outcome": outcome, "profit_loss": profit_loss}


async def process_resolution(resolved_position: dict):
    """
    Mark a trade as resolved and update P&L.

    resolved_position keys:
        market_id, token_id, station, target_date, side,
        payout (float), cost (float)
    """
    pool = await get_pool()
    payout = resolved_position.get("payout", 0.0)
    cost = resolved_position.get("cost", 0.0)

    if payout > cost * 0.95:
        outcome = "win"
    elif payout < 0.01:
        outcome = "loss"
    else:
        outcome = "void"

    profit_loss = payout - cost

    async with pool.acquire() as conn:
        # Find the matching open trade
        trade = await conn.fetchrow(
            """
            SELECT id FROM trades
            WHERE station=$1 AND target_date=$2 AND resolved=FALSE
            ORDER BY opened_at DESC LIMIT 1
            """,
            resolved_position.get("station"),
            resolved_position.get("target_date"),
        )
        if trade:
            await conn.execute(
                """
                UPDATE trades SET
                    resolved=TRUE, resolved_at=NOW(),
                    outcome=$1, payout=$2, profit_loss=$3
                WHERE id=$4
                """,
                outcome, payout, profit_loss, trade["id"],
            )
            logger.info(
                "Resolution: station=%s date=%s outcome=%s pnl=%.2f",
                resolved_position.get("station"),
                resolved_position.get("target_date"),
                outcome, profit_loss,
            )

    # Update model accuracy if we know the actual high
    actual_high_c = resolved_position.get("actual_high_c")
    if actual_high_c is not None:
        from models import update_accuracy_on_resolution
        await update_accuracy_on_resolution(
            resolved_position["station"],
            resolved_position["target_date"],
            actual_high_c,
        )

    # Update daily summary
    await _update_daily_summary_on_resolution(outcome, profit_loss, payout, cost)


async def _update_daily_summary_on_resolution(outcome: str, profit_loss: float,
                                               payout: float, cost: float):
    pool = await get_pool()
    today = date.today()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM daily_summary WHERE date=$1", today
        )
        if not existing:
            await conn.execute(
                "INSERT INTO daily_summary (date) VALUES ($1) ON CONFLICT DO NOTHING",
                today,
            )
        if outcome == "win":
            await conn.execute(
                """
                UPDATE daily_summary SET
                    trades_won = trades_won + 1,
                    gross_profit = gross_profit + $1,
                    net_pnl = net_pnl + $1
                WHERE date=$2
                """,
                profit_loss, today,
            )
        elif outcome == "loss":
            await conn.execute(
                """
                UPDATE daily_summary SET
                    trades_lost = trades_lost + 1,
                    gross_loss = gross_loss + $1,
                    net_pnl = net_pnl + $1
                WHERE date=$2
                """,
                profit_loss, today,
            )
        else:
            await conn.execute(
                "UPDATE daily_summary SET trades_void=trades_void+1 WHERE date=$1",
                today,
            )


# ---------------------------------------------------------------------------
# Alerts log
# ---------------------------------------------------------------------------
async def log_alert(alert: dict) -> str:
    """Insert an alert into alerts_log and return the generated alert_id."""
    alert_id = str(uuid.uuid4())[:12]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO alerts_log
                (alert_id, station, trade_type, bin_label, side,
                 suggested_price, suggested_shares, suggested_cost,
                 confidence_score, ev)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            """,
            alert_id,
            alert.get("station"),
            alert.get("trade_type"),
            alert.get("bin_label"),
            alert.get("side"),
            alert.get("entry_price"),
            alert.get("shares"),
            alert.get("cost"),
            alert.get("confidence_score"),
            alert.get("ev"),
        )
    return alert_id


async def update_alert_acted(alert_id: str, acted: bool):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE alerts_log SET was_acted_on=$1 WHERE alert_id=$2",
            acted, alert_id,
        )


# ---------------------------------------------------------------------------
# Capital reservations
# ---------------------------------------------------------------------------
async def create_reservation(amount: float, reason: str) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO capital_reservations (amount, reason, expires_at)
            VALUES ($1, $2, NOW() + INTERVAL '10 minutes')
            RETURNING id
            """,
            amount, reason,
        )
        return row["id"]


async def expire_reservations():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM capital_reservations WHERE expires_at < NOW() AND consumed=FALSE"
        )


async def consume_reservation(reason: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE capital_reservations SET consumed=TRUE WHERE reason=$1",
            reason,
        )


async def get_active_reservations_total() -> float:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM capital_reservations
            WHERE consumed=FALSE AND expires_at > NOW()
            """
        )
        return float(row["total"]) if row else 0.0


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------
async def check_circuit_breakers(wallet_state: dict, config: dict) -> Optional[str]:
    """
    Returns the active circuit breaker action string, or None if all clear.
    """
    total_value = wallet_state.get("total_value", 0)
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get initial capital (from earliest daily_summary or trades)
        row = await conn.fetchrow(
            "SELECT starting_capital FROM daily_summary ORDER BY date ASC LIMIT 1"
        )
        initial_capital = row["starting_capital"] if row and row["starting_capital"] else total_value

    if initial_capital <= 0:
        return None

    ratio = total_value / initial_capital
    breakers = config.get("circuit_breakers", {})

    if ratio < 0.50:
        return breakers.get("capital_50pct", {}).get("action", "halt_3_days")
    if ratio < 0.60:
        return breakers.get("capital_60pct", {}).get("action", "reduce_size_50pct_no_only")
    if ratio < 0.75:
        return breakers.get("capital_75pct", {}).get("action", "reduce_size_30pct")

    # Check 3 consecutive losing days
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT net_pnl FROM daily_summary ORDER BY date DESC LIMIT 3"
        )
    if len(rows) >= 3 and all(r["net_pnl"] < 0 for r in rows):
        return breakers.get("three_losing_days", {}).get("action", "reduce_size_30pct_2_days")

    return None


async def daily_loss_exceeded(wallet_state: dict, config: dict) -> bool:
    """Check if today's losses exceed the daily limit."""
    pool = await get_pool()
    today = date.today()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT starting_capital FROM daily_summary WHERE date=$1", today
        )
    if not row or not row["starting_capital"]:
        return False
    starting = row["starting_capital"]
    limit_pct = config.get("trading", {}).get("daily_loss_limit_pct", 8)
    current = wallet_state.get("total_value", starting)
    loss = starting - current
    return loss >= starting * (limit_pct / 100.0)


# ---------------------------------------------------------------------------
# Model accuracy helpers
# ---------------------------------------------------------------------------
async def get_model_accuracy(station: str, model_name: str) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    now = datetime.utcnow()
    year_month = now.strftime("%Y-%m")
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT * FROM model_accuracy
            WHERE station=$1 AND model_name=$2 AND year_month=$3
            """,
            station, model_name, year_month,
        )


async def upsert_model_accuracy(station: str, model_name: str,
                                 ewma_error: float, ewma_bias: float,
                                 sample_count: int, weight: float):
    pool = await get_pool()
    year_month = datetime.utcnow().strftime("%Y-%m")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO model_accuracy (station, model_name, year_month,
                ewma_error, ewma_bias, sample_count, weight)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (station, model_name, year_month)
            DO UPDATE SET ewma_error=$4, ewma_bias=$5,
                sample_count=$6, weight=$7, last_updated=NOW()
            """,
            station, model_name, year_month,
            ewma_error, ewma_bias, sample_count, weight,
        )


# ---------------------------------------------------------------------------
# Time-of-high stats
# ---------------------------------------------------------------------------
async def get_time_of_high(station: str, month: int) -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT hour_local, frequency, sample_count
            FROM time_of_high_stats
            WHERE station=$1 AND month=$2
            ORDER BY hour_local
            """,
            station, month,
        )


async def upsert_time_of_high(station: str, month: int, hour_local: int,
                               frequency: float, sample_count: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO time_of_high_stats (station, month, hour_local, frequency, sample_count)
            VALUES ($1,$2,$3,$4,$5)
            ON CONFLICT (station, month, hour_local)
            DO UPDATE SET frequency=$4, sample_count=$5
            """,
            station, month, hour_local, frequency, sample_count,
        )


# ---------------------------------------------------------------------------
# Warming rates
# ---------------------------------------------------------------------------
async def get_warming_rate(station: str, month: int) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT * FROM warming_rates WHERE station=$1 AND month=$2",
            station, month,
        )


async def upsert_warming_rate(station: str, month: int,
                               max_rate: float, p95_rate: float,
                               p75_rate: float, sample_days: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO warming_rates (station, month,
                max_warming_rate_c_per_hr, p95_warming_rate_c_per_hr,
                p75_warming_rate_c_per_hr, sample_days)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (station, month)
            DO UPDATE SET max_warming_rate_c_per_hr=$3,
                p95_warming_rate_c_per_hr=$4,
                p75_warming_rate_c_per_hr=$5, sample_days=$6
            """,
            station, month, max_rate, p95_rate, p75_rate, sample_days,
        )


# ---------------------------------------------------------------------------
# Deployed capital per market
# ---------------------------------------------------------------------------
async def deployed_on_station_date(station: str, target_date: date) -> float:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(cost), 0) AS total
            FROM trades WHERE station=$1 AND target_date=$2 AND resolved=FALSE
            """,
            station, target_date,
        )
        return float(row["total"]) if row else 0.0


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
async def generate_daily_summary_text(wallet_state: dict) -> str:
    pool = await get_pool()
    today = date.today()
    async with pool.acquire() as conn:
        # Ensure row exists
        await conn.execute(
            """
            INSERT INTO daily_summary (date, starting_capital, ending_capital)
            VALUES ($1, $2, $3)
            ON CONFLICT (date) DO UPDATE SET ending_capital=$3
            """,
            today,
            wallet_state.get("total_value", 0),
            wallet_state.get("total_value", 0),
        )
        row = await conn.fetchrow("SELECT * FROM daily_summary WHERE date=$1", today)

    if not row:
        return "📋 No data for daily summary yet."

    trades_total = (row["trades_won"] or 0) + (row["trades_lost"] or 0) + (row["trades_void"] or 0)
    win_rate = (
        round((row["trades_won"] or 0) / trades_total * 100)
        if trades_total > 0 else 0
    )
    net = row["net_pnl"] or 0
    sign = "+" if net >= 0 else ""

    return (
        f"📋 DAILY SUMMARY: {today}\n\n"
        f"💰 ${row['starting_capital'] or 0:.2f} → ${row['ending_capital'] or 0:.2f} "
        f"({sign}${net:.2f})\n"
        f"📊 Trades: {trades_total} | Wins: {row['trades_won'] or 0} "
        f"| Losses: {row['trades_lost'] or 0} ({win_rate}%)\n"
        f"Alerts sent: {row['alerts_sent'] or 0} | Skipped: {row['alerts_skipped'] or 0}"
    )


async def generate_weekly_report() -> str:
    pool = await get_pool()
    week_ago = date.today() - timedelta(days=7)
    async with pool.acquire() as conn:
        trades = await conn.fetch(
            "SELECT * FROM trades WHERE opened_at::date >= $1", week_ago
        )
        summaries = await conn.fetch(
            "SELECT * FROM daily_summary WHERE date >= $1 ORDER BY date", week_ago
        )

    total = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "win")
    losses = sum(1 for t in trades if t["outcome"] == "loss")
    win_rate = round(wins / total * 100) if total else 0

    total_pnl = sum((t["profit_loss"] or 0) for t in trades if t["resolved"])
    sign = "+" if total_pnl >= 0 else ""

    by_type: Dict[str, Dict[str, int]] = {}
    for t in trades:
        tt = t["trade_type"]
        if tt not in by_type:
            by_type[tt] = {"total": 0, "wins": 0, "losses": 0}
        by_type[tt]["total"] += 1
        if t["outcome"] == "win":
            by_type[tt]["wins"] += 1
        elif t["outcome"] == "loss":
            by_type[tt]["losses"] += 1

    type_lines = ""
    for tt, stats in by_type.items():
        wr = round(stats["wins"] / stats["total"] * 100) if stats["total"] else 0
        type_lines += f"  {tt}: {stats['total']} trades, {stats['wins']}W/{stats['losses']}L ({wr}%)\n"

    by_station: Dict[str, float] = {}
    for t in trades:
        st = t["station"]
        by_station[st] = by_station.get(st, 0) + (t["profit_loss"] or 0)

    station_lines = ""
    for st, pnl in by_station.items():
        s = "+" if pnl >= 0 else ""
        station_lines += f"  {st}: {s}${pnl:.2f}\n"

    best = max(trades, key=lambda t: t["profit_loss"] or 0) if trades else None
    worst = min(trades, key=lambda t: t["profit_loss"] or 0) if trades else None
    best_str = (
        f"{best['station']} {best['trade_type']} +${best['profit_loss']:.2f}"
        if best and best["profit_loss"] else "N/A"
    )
    worst_str = (
        f"{worst['station']} {worst['trade_type']} ${worst['profit_loss']:.2f}"
        if worst and worst["profit_loss"] else "N/A"
    )

    return (
        f"📋 WEEKLY REPORT: {week_ago} — {date.today()}\n\n"
        f"💰 P&L: {sign}${total_pnl:.2f}\n"
        f"📊 {total} trades | {wins} wins | {losses} losses ({win_rate}%)\n\n"
        f"By type:\n{type_lines}\n"
        f"By station:\n{station_lines}\n"
        f"🏆 Best: {best_str}\n"
        f"💔 Worst: {worst_str}"
    )


# ---------------------------------------------------------------------------
# CSV data export
# ---------------------------------------------------------------------------
async def export_data() -> bytes:
    """Export all key tables as CSV files in a ZIP archive."""
    pool = await get_pool()
    tables = [
        "metar_readings", "model_forecasts", "market_snapshots",
        "trades", "model_accuracy", "daily_summary",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        async with pool.acquire() as conn:
            for table in tables:
                rows = await conn.fetch(f"SELECT * FROM {table}")
                if not rows:
                    continue
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for r in rows:
                    writer.writerow(dict(r))
                zf.writestr(f"{table}.csv", csv_buf.getvalue())
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Bot start timestamp (for conservative 3-day mode)
# ---------------------------------------------------------------------------
async def get_bot_start_date() -> Optional[date]:
    """Return the date the bot first started (earliest daily_summary or trade)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT MIN(date) AS d FROM daily_summary"
        )
        if row and row["d"]:
            return row["d"]
        row = await conn.fetchrow(
            "SELECT MIN(opened_at::date) AS d FROM trades"
        )
        return row["d"] if row else None


async def ensure_today_summary(wallet_state: dict):
    """Make sure today has a daily_summary row with starting capital."""
    pool = await get_pool()
    today = date.today()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO daily_summary (date, starting_capital, ending_capital)
            VALUES ($1, $2, $2)
            ON CONFLICT (date) DO NOTHING
            """,
            today, wallet_state.get("total_value", 0),
        )


# ---------------------------------------------------------------------------
# Station info helper
# ---------------------------------------------------------------------------
async def get_station_info(icao: str) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM stations WHERE icao=$1", icao)


async def get_active_stations() -> List[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM stations WHERE active=TRUE")
