"""
main.py — Main Loop & Orchestration

Entry point. Runs the async event loop. Orchestrates the 12-step scan cycle.
Manages scan frequency (normal 5min vs fast 2min during model releases).
Initialises all modules, connects to database, starts Telegram bot.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, date
from typing import Dict

from aiohttp import web

import yaml

# Try to use uvloop for better async performance (Linux/macOS)
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import alerts
import allocator
import commands
import markets
import metar as metar_mod
import models as models_mod
import probability
import scheduler
import signals as signals_mod
import tracker
import wallet as wallet_mod

from telegram.ext import Application

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Telegram bot setup
# ---------------------------------------------------------------------------
async def setup_telegram(config: dict) -> Application:
    """Build and initialise the Telegram bot application."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    commands.register_handlers(app)
    await app.initialize()
    await app.start()
    # Wait for old instance to release the polling lock during redeploys
    await asyncio.sleep(5)
    # Start polling in background (non-blocking)
    await app.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=["message"],
    )
    logger.info("Telegram bot started")
    return app


# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------
async def startup(config: dict):
    """
    1. Connect to PostgreSQL
    2. Run schema migration
    3. Initialise stations from config
    4. Initialise Telegram bot
    5. Sync wallet
    6. Populate historical stats
    7. Send startup message
    """
    logger.info("=== Weather-Edge Bot Starting ===")

    # 1. Database
    await tracker.init_db()
    logger.info("Database connected")

    # 2. Stations
    stations_cfg = config.get("stations", {})
    await tracker.init_stations(stations_cfg)

    # 2b. Initialize wallet with config capital
    starting_capital = config.get("bot", {}).get("starting_capital", 100.0)
    wallet_mod.set_config_capital(starting_capital)

    # 3. Telegram
    tg_app = await setup_telegram(config)

    # 4. Wallet sync
    try:
        ws = await wallet_mod.sync()
        logger.info("Wallet synced: balance=%.2f total=%.2f", ws.balance, ws.total_value)
    except Exception as e:
        logger.error("Initial wallet sync failed: %s", e)
        ws = wallet_mod.WalletState(balance=0.0)

    # 5. Ensure today's summary row exists
    await tracker.ensure_today_summary(
        {"total_value": ws.total_value}
    )

    # 6. Populate historical stats for each station
    for icao, cfg in stations_cfg.items():
        try:
            await probability.populate_historical_stats(icao, cfg["timezone"])
        except Exception as e:
            logger.error("Historical stats error for %s: %s", icao, e)

    # 7. Startup message
    await alerts.send_startup_message(ws, list(stations_cfg.keys()))
    logger.info("Startup complete — entering main loop")

    return tg_app


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------
async def main_loop(config: dict):
    """The forever-running scan cycle."""
    tg_app = await startup(config)
    stations_cfg = config.get("stations", {})
    station_ids = list(stations_cfg.keys())

    # Deduplication cache: {signal_key: timestamp_sent}
    # Prevents sending the same alert within 30 minutes
    _recent_alerts: Dict[str, datetime] = {}
    DEDUP_MINUTES = 30

    while True:
        scan_start = datetime.utcnow()  # noqa: DTZ003
        logger.info("--- Scan cycle start ---")

        try:
            # Determine scan interval
            interval = scheduler.check_if_model_release_window(config)

            # Expire old capital reservations
            await allocator.expire_reservations()

            # Step 1: Wallet sync
            try:
                wallet_state = await wallet_mod.sync()
            except Exception as e:
                logger.error("Wallet sync failed: %s — skipping cycle", e)
                await asyncio.sleep(interval)
                continue

            # Ensure today's summary
            await tracker.ensure_today_summary(
                {"total_value": wallet_state.total_value}
            )

            # Step 2: Process resolutions
            for rp in wallet_state.resolved:
                try:
                    await tracker.process_resolution({
                        "station": rp.station,
                        "target_date": rp.target_date,
                        "payout": rp.payout,
                        "cost": rp.cost,
                    })
                    await alerts.send_resolution_notification({
                        "station": rp.station,
                        "target_date": rp.target_date,
                        "side": rp.side,
                        "bin_label": rp.bin_label,
                        "cost": rp.cost,
                        "payout": rp.payout,
                        "profit_loss": rp.profit_loss,
                        "outcome": "win" if rp.profit_loss > 0 else "loss",
                    })
                except Exception as e:
                    logger.error("Resolution processing error: %s", e)

            # Step 3: Circuit breakers
            breaker = await tracker.check_circuit_breakers(
                {"total_value": wallet_state.total_value}, config
            )
            if breaker and "halt" in breaker:
                logger.info("Circuit breaker HALT active: %s", breaker)
                await alerts.send_circuit_breaker_alert(breaker, config)
                await asyncio.sleep(interval)
                continue

            # Step 4: Daily loss limit
            if await tracker.daily_loss_exceeded(
                {"total_value": wallet_state.total_value}, config
            ):
                await alerts.send_daily_loss_limit_message()
                await asyncio.sleep(interval)
                continue

            # Step 5: Fetch METAR for all stations
            metar_data = {}
            try:
                raw_metar = await metar_mod.fetch_all_stations(station_ids)
                # Enrich with velocity and rounding edge
                for icao, m in raw_metar.items():
                    cfg = stations_cfg.get(icao, {})
                    metar_data[icao] = await metar_mod.enrich_metar(
                        m, cfg.get("timezone", "UTC"), cfg.get("unit", "C")
                    )
            except Exception as e:
                logger.error("METAR fetch error: %s — using cached", e)

            # Step 6: Fetch model forecasts
            model_data = {}
            try:
                if scheduler.should_fetch_models(config):
                    model_data = await models_mod.fetch_all_stations(stations_cfg)
                    scheduler.mark_models_fetched()
                else:
                    model_data = await models_mod.get_latest_from_db(stations_cfg)
            except Exception as e:
                logger.error("Model fetch error: %s", e)

            # Step 7: Fetch Polymarket markets
            market_data = {}
            try:
                market_data = await markets.fetch_active_weather_markets(station_ids)
            except Exception as e:
                logger.error("Market fetch error: %s — retrying in 30s", e)
                await asyncio.sleep(30)
                try:
                    market_data = await markets.fetch_active_weather_markets(station_ids)
                except Exception as e2:
                    logger.error("Market retry also failed: %s", e2)

            if not market_data:
                logger.info("No active markets found — sleeping")
                await asyncio.sleep(interval)
                continue

            # Step 8: Probability engine
            prob_data = {}
            try:
                prob_data = await probability.calculate_all(
                    metar_data, model_data, market_data, stations_cfg
                )
            except Exception as e:
                logger.error("Probability calc error: %s", e)

            # Step 9: Generate signals
            raw_signals = []
            try:
                raw_signals = await signals_mod.generate_all(
                    metar_data, model_data, market_data, prob_data,
                    wallet_state, config, stations_cfg,
                )
                logger.info("Signals generated: %d", len(raw_signals))
            except Exception as e:
                logger.error("Signal generation error: %s", e)

            # Step 10: Allocate and rank
            ranked_alerts = []
            try:
                ranked_alerts = await allocator.rank_and_size(
                    raw_signals, wallet_state, config
                )
                logger.info("Alerts ranked: %d", len(ranked_alerts))
            except Exception as e:
                logger.error("Allocation error: %s", e)

            # Step 11: Send alerts (if not paused)
            if not commands.is_paused():
                if ranked_alerts:
                    # Clean expired dedup entries
                    now = datetime.utcnow()
                    expired = [k for k, v in _recent_alerts.items()
                               if (now - v).total_seconds() > DEDUP_MINUTES * 60]
                    for k in expired:
                        del _recent_alerts[k]

                    for alert in ranked_alerts:
                        # Dedup key: station + date + trade_type + bin_label
                        dedup_key = f"{alert.station}_{alert.target_date}_{alert.trade_type}_{alert.bin_label}"
                        if dedup_key in _recent_alerts:
                            logger.info("Skipping duplicate alert: %s", dedup_key)
                            continue

                        try:
                            await alerts.send_trade_alert(alert, wallet_state)
                            _recent_alerts[dedup_key] = now
                            await allocator.reserve_capital(
                                alert.sized_cost, alert.alert_id
                            )
                        except Exception as e:
                            logger.error("Alert send error: %s", e)
                elif scheduler.is_end_of_trading_day(stations_cfg):
                    best_ev = max(
                        (s.ev for s in raw_signals), default=0
                    )
                    silent_threshold = config.get("trading", {}).get(
                        "silent_day_min_ev", 0.50
                    )
                    if best_ev < silent_threshold:
                        await alerts.send_silent_day_message(
                            best_ev=best_ev,
                            markets_scanned=len(market_data),
                            stations_scanned=len(station_ids),
                            balance=wallet_state.balance,
                        )

            # Step 12: Store market snapshot
            try:
                await tracker.store_market_snapshot(market_data)
            except Exception as e:
                logger.error("Snapshot store error: %s", e)

            # Check scheduled events (reports, reminders, stats updates)
            try:
                await scheduler.check_scheduled_events(
                    config, wallet_state, stations_cfg
                )
            except Exception as e:
                logger.error("Scheduler error: %s", e)

        except Exception as e:
            logger.error("CRITICAL scan cycle error: %s", e, exc_info=True)
            try:
                await alerts._send(f"🚨 Bot error (cycle continues): {str(e)[:200]}")
            except Exception:
                pass

        # Sleep until next cycle
        elapsed = (datetime.utcnow() - scan_start).total_seconds()  # noqa: DTZ003
        sleep_time = max(0, interval - elapsed)
        logger.info(
            "--- Scan cycle done (%.1fs) — sleeping %.0fs ---",
            elapsed, sleep_time,
        )
        await asyncio.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Health check HTTP server (keeps Render free web service alive)
# ---------------------------------------------------------------------------
async def health_handler(request):
    """Health check endpoint for Render / UptimeRobot."""
    return web.Response(text="Weather-Edge Bot is running", status=200)


async def status_handler(request):
    """Quick status endpoint."""
    uptime = (datetime.utcnow() - _boot_time).total_seconds() / 3600.0
    return web.Response(
        text=f"Weather-Edge Bot | Uptime: {uptime:.1f}h",
        status=200,
    )


_boot_time = datetime.utcnow()


async def run_health_server():
    """Run a lightweight HTTP server on $PORT for Render."""
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)

    port = int(os.environ.get("PORT", 10000))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Health check server started on port %d", port)

    # Keep running forever
    while True:
        await asyncio.sleep(3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def run_all():
    """Run health check server and main scan loop concurrently."""
    config = load_config()
    logger.info("Config loaded: %d stations", len(config.get("stations", {})))
    await asyncio.gather(
        run_health_server(),
        main_loop(config),
    )


def main():
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
