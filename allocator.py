"""
allocator.py — Priority Queue, Dynamic Sizing, Capital Reservation

Takes raw signals, ranks by EV, calculates exact position sizes based on
real wallet balance, checks all limits, returns final alerts to send.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import tracker
import wallet as wallet_mod
from signals import Signal

logger = logging.getLogger("allocator")


# ---------------------------------------------------------------------------
# AlertReady data model
# ---------------------------------------------------------------------------
@dataclass
class AlertReady:
    # Inherits all Signal fields
    trade_type: str
    station: str
    city: str
    target_date: date
    side: str
    bin_label: str
    bins: List[dict] = field(default_factory=list)
    entry_price: float = 0.0
    confidence_score: int = 0
    confidence_components: dict = field(default_factory=dict)
    ev: float = 0.0
    win_probability: float = 0.0
    profit_if_win: float = 0.0
    loss_if_lose: float = 0.0
    market_id: str = ""
    polymarket_url: str = ""
    book_depth: int = 0
    metar_summary: str = ""
    model_summary: str = ""
    # Sized fields
    shares: float = 0.0
    cost: float = 0.0
    sized_shares: float = 0.0
    sized_cost: float = 0.0
    avg_fill_price: float = 0.0
    sized_profit_if_win: float = 0.0
    sized_ev: float = 0.0
    available_after: float = 0.0
    is_combo: bool = False
    combo_alerts: List = field(default_factory=list)
    alert_id: str = ""


# ---------------------------------------------------------------------------
# Main ranking and sizing
# ---------------------------------------------------------------------------
async def rank_and_size(signals: List[Signal], wallet_state,
                         config: dict) -> List[AlertReady]:
    """
    Rank signals by EV, size positions, check limits, return top alerts.
    """
    if not signals:
        return []

    trading_cfg = config.get("trading", {})
    max_alerts = trading_cfg.get("max_alerts_per_cycle", 7)

    # Step 1: Calculate available capital
    balance = wallet_state.balance if hasattr(wallet_state, "balance") else wallet_state.get("balance", 0)
    total_value = wallet_state.total_value if hasattr(wallet_state, "total_value") else wallet_state.get("total_value", 0)

    reservations = await tracker.get_active_reservations_total()
    reserve_pct = trading_cfg.get("reserve_pct", 15) / 100.0
    reserve_amount = total_value * reserve_pct
    available = max(0, balance - reservations - reserve_amount)

    # Step 2: Determine sizing mode
    total_resolved = await tracker.get_total_trades_count()
    initial_flat_trades = trading_cfg.get("initial_flat_trades", 30)

    # Check conservative 3-day mode
    bot_start = await tracker.get_bot_start_date()
    conservative_days = config.get("bot", {}).get("conservative_days", 3)
    in_conservative_mode = False
    if bot_start:
        days_running = (date.today() - bot_start).days
        in_conservative_mode = days_running < conservative_days

    # Check circuit breaker sizing adjustments
    breaker = await tracker.check_circuit_breakers(
        {"total_value": total_value}, config
    )
    size_multiplier = 1.0
    if breaker:
        if "reduce_size_30pct" in breaker:
            size_multiplier = 0.70
        elif "reduce_size_50pct" in breaker:
            size_multiplier = 0.50

    # Step 3: Size each signal
    sized_alerts: List[AlertReady] = []

    for signal in signals:
        # Calculate position size
        if in_conservative_mode or total_resolved < initial_flat_trades:
            # Flat sizing: 5% of portfolio
            size_pct = trading_cfg.get("initial_flat_size_pct", 5) / 100.0
        else:
            # Dynamic sizing based on confidence
            dyn = trading_cfg.get("dynamic_sizing", {})
            if signal.confidence_score >= 80:
                size_pct = dyn.get("high_confidence", {}).get("max_pct", 10) / 100.0
            elif signal.confidence_score >= 60:
                size_pct = dyn.get("medium_confidence", {}).get("max_pct", 7) / 100.0
            else:
                size_pct = dyn.get("low_confidence", {}).get("max_pct", 4) / 100.0

        position_size = total_value * size_pct * size_multiplier

        # Hard limits
        max_single_pct = trading_cfg.get("max_single_trade_pct", 12) / 100.0
        position_size = min(position_size, total_value * max_single_pct)
        position_size = min(position_size, available)

        if position_size < 0.50:  # minimum meaningful position
            continue

        # Per-market limit check
        deployed = await tracker.deployed_on_station_date(
            signal.station, signal.target_date
        )
        max_market_pct = trading_cfg.get("max_single_market_pct", 25) / 100.0
        max_market = total_value * max_market_pct
        if deployed + position_size > max_market:
            position_size = max(0, max_market - deployed)
            if position_size < 0.50:
                continue

        # Calculate shares and cost
        entry_price = signal.entry_price
        if entry_price <= 0:
            continue

        shares = position_size / entry_price
        cost = shares * entry_price

        # Recalculate EV with real sizing
        if signal.side == "YES":
            profit_if_win = shares * (1.0 - entry_price)
        else:
            profit_if_win = shares * (1.0 - entry_price)  # NO win profit
        loss_if_lose = cost

        sized_ev = signal.win_probability * profit_if_win - (1 - signal.win_probability) * loss_if_lose

        alert = AlertReady(
            trade_type=signal.trade_type,
            station=signal.station,
            city=signal.city,
            target_date=signal.target_date,
            side=signal.side,
            bin_label=signal.bin_label,
            bins=signal.bins,
            entry_price=entry_price,
            confidence_score=signal.confidence_score,
            confidence_components=signal.confidence_components,
            ev=signal.ev,
            win_probability=signal.win_probability,
            profit_if_win=round(profit_if_win, 2),
            loss_if_lose=round(loss_if_lose, 2),
            market_id=signal.market_id,
            polymarket_url=signal.polymarket_url,
            book_depth=signal.book_depth,
            metar_summary=signal.metar_summary,
            model_summary=signal.model_summary,
            shares=round(shares, 1),
            cost=round(cost, 2),
            sized_shares=round(shares, 1),
            sized_cost=round(cost, 2),
            avg_fill_price=entry_price,
            sized_profit_if_win=round(profit_if_win, 2),
            sized_ev=round(sized_ev, 2),
            available_after=round(available - cost, 2),
        )
        sized_alerts.append(alert)

    # Step 4: Rank by EV descending
    sized_alerts.sort(key=lambda a: a.sized_ev, reverse=True)

    # Step 5: Take top N
    top_alerts = sized_alerts[:max_alerts]

    # Step 6: Group combos (YES lock-in + NO tails on same market)
    top_alerts = _group_combos(top_alerts)

    # Step 7: Final capital check — walk through in order
    final_alerts: List[AlertReady] = []
    running_available = available

    for alert in top_alerts:
        total_cost = alert.sized_cost
        if alert.is_combo:
            total_cost += sum(ca.sized_cost for ca in alert.combo_alerts)

        if total_cost <= running_available:
            alert.available_after = round(running_available - total_cost, 2)
            final_alerts.append(alert)
            running_available -= total_cost

    return final_alerts


# ---------------------------------------------------------------------------
# Combo grouping
# ---------------------------------------------------------------------------
def _group_combos(alerts: List[AlertReady]) -> List[AlertReady]:
    """Group YES lock-in with corresponding NO tails on same station/date."""
    lockins = [a for a in alerts if a.trade_type == "lockin_yes"]
    no_tails = [a for a in alerts if a.trade_type == "no_tail"]
    other = [a for a in alerts if a.trade_type not in ("lockin_yes", "no_tail")]

    grouped: List[AlertReady] = []

    for lockin in lockins:
        matching_tails = [
            n for n in no_tails
            if n.station == lockin.station and n.target_date == lockin.target_date
        ]
        if matching_tails:
            lockin.is_combo = True
            lockin.combo_alerts = matching_tails
            # Remove matched tails from no_tails list
            for mt in matching_tails:
                if mt in no_tails:
                    no_tails.remove(mt)
        grouped.append(lockin)

    # Add remaining ungrouped NO tails and other signals
    grouped.extend(no_tails)
    grouped.extend(other)

    return grouped


# ---------------------------------------------------------------------------
# Capital reservation
# ---------------------------------------------------------------------------
async def reserve_capital(cost: float, alert_id: str):
    """Create a 10-minute capital reservation."""
    await tracker.create_reservation(cost, f"alert_{alert_id}")
    logger.info("Reserved $%.2f for alert %s", cost, alert_id)


async def expire_reservations():
    """Clear expired and consumed reservations."""
    await tracker.expire_reservations()
