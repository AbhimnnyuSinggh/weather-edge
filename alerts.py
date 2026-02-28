"""
alerts.py — Telegram Formatting, Alert Ranking, Silent Days

Formats and sends all Telegram messages: trade alerts, combos, resolutions,
silent day messages, daily summaries, weekly reports, circuit breakers.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import pytz
from telegram import Bot
from telegram.constants import ParseMode

import models
import tracker
from allocator import AlertReady

logger = logging.getLogger("alerts")

# Telegram bot instance (lazy init)
_bot: Optional[Bot] = None


def get_bot() -> Bot:
    global _bot
    if _bot is None:
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        _bot = Bot(token=token)
    return _bot


def _chat_id() -> str:
    return os.environ["TELEGRAM_CHAT_ID"]


def _ist_time() -> str:
    """Current time in IST formatted for display."""
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%I:%M %p IST")


async def _send(text: str):
    """Send a message to the configured Telegram chat."""
    try:
        bot = get_bot()
        # Split long messages (Telegram limit ~4096 chars)
        if len(text) > 4000:
            parts = [text[i:i + 4000] for i in range(0, len(text), 4000)]
            for part in parts:
                await bot.send_message(
                    chat_id=_chat_id(), text=part,
                    disable_web_page_preview=True,
                )
        else:
            await bot.send_message(
                chat_id=_chat_id(), text=text,
                disable_web_page_preview=True,
            )
    except Exception as e:
        logger.error("Telegram send error: %s", e)


# ---------------------------------------------------------------------------
# Trade alerts
# ---------------------------------------------------------------------------
async def send_trade_alert(alert: AlertReady, wallet_state):
    """Format and send a trade alert based on type."""
    balance = wallet_state.balance if hasattr(wallet_state, "balance") else wallet_state.get("balance", 0)
    total_value = wallet_state.total_value if hasattr(wallet_state, "total_value") else wallet_state.get("total_value", 0)

    if alert.is_combo:
        await send_combo_alert(alert, wallet_state)
        return

    if alert.trade_type == "lockin_yes":
        msg = _format_lockin(alert, balance)
    elif alert.trade_type == "no_tail":
        msg = _format_no_tail(alert, balance)
    elif alert.trade_type == "forecast_yes":
        msg = _format_forecast(alert, balance)
    elif alert.trade_type == "ladder":
        msg = _format_ladder(alert, balance)
    elif alert.trade_type == "edge_yes":
        msg = _format_edge_yes(alert, balance)
    elif alert.trade_type == "edge_no":
        msg = _format_edge_no(alert, balance)
    elif alert.trade_type == "edge_ladder":
        msg = _format_edge_ladder(alert, balance)
    else:
        msg = f"📊 Signal: {alert.trade_type} {alert.station} {alert.bin_label}"

    # Log alert
    alert_id = await tracker.log_alert({
        "station": alert.station,
        "trade_type": alert.trade_type,
        "bin_label": alert.bin_label,
        "side": alert.side,
        "entry_price": alert.entry_price,
        "shares": alert.sized_shares,
        "cost": alert.sized_cost,
        "confidence_score": alert.confidence_score,
        "ev": alert.sized_ev,
    })
    alert.alert_id = alert_id

    await _send(msg)
    logger.info("Alert sent: %s %s %s (score=%d, ev=%.2f)",
                alert.trade_type, alert.station, alert.bin_label,
                alert.confidence_score, alert.sized_ev)


def _format_kelly(a: AlertReady, balance: float) -> str:
    edge = max(0.01, a.win_probability - a.entry_price)
    suggested_size = edge * balance * 0.25
    suggested_pct = (suggested_size / balance * 100) if balance > 0 else 0
    max_risk = balance * 0.08
    return f"Suggested size: ${suggested_size:.2f} ({suggested_pct:.1f}% of ${balance:.0f}) | Max risk today: ${max_risk:.2f}"


def _format_lockin(a: AlertReady, balance: float) -> str:
    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"🔒 LOCK-IN: {a.city} {a.station}\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"📡 METAR\n"
        f"{a.metar_summary}\n"
        f"High locked probability: {a.win_probability*100:.0f}%\n"
        f"{a.model_summary}\n\n"
        f"✅ BUY YES: \"{a.bin_label}\" at {a.entry_price*100:.0f}¢\n"
        f"   Shares: {a.sized_shares:.0f} | Cost: ${a.sized_cost:.2f} | "
        f"Profit if win: ${a.sized_profit_if_win:.2f}\n"
        f"   Win prob: {a.win_probability*100:.0f}% | Realistic EV: +${a.sized_ev:.2f} (max 15x return)\n"
        f"   Market: {a.polymarket_url}\n"
        f"   {kelly_str}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trade: ${a.available_after:.2f}"
    )


def _format_no_tail(a: AlertReady, balance: float) -> str:
    yes_price = 1.0 - a.entry_price
    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"🛡️ NO PLAY: {a.city} {a.station}\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"📡 METAR\n"
        f"{a.metar_summary}\n"
        f"Bin impossibility: {a.win_probability*100:.0f}% (math-confirmed)\n\n"
        f"🛡️ BUY NO: \"{a.bin_label}\" (YES at {yes_price*100:.0f}¢, "
        f"NO costs {a.entry_price*100:.0f}¢)\n"
        f"   Shares: {a.sized_shares:.0f} | Cost: ${a.sized_cost:.2f} | "
        f"Profit if win: ${a.sized_profit_if_win:.2f}\n"
        f"   Win prob: {a.win_probability*100:.0f}% | Realistic EV: +${a.sized_ev:.2f} (max 15x return)\n"
        f"   Market: {a.polymarket_url}\n"
        f"   {kelly_str}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trade: ${a.available_after:.2f}"
    )


def _format_forecast(a: AlertReady, balance: float) -> str:
    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"📊 FORECAST: {a.city} {a.station} ({a.target_date})\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"🌡️ Models:\n"
        f"{a.model_summary}\n"
        f"{a.metar_summary}\n"
        f"API Status: {_get_sources_string()}\n\n"
        f"✅ BUY YES: \"{a.bin_label}\" at {a.entry_price*100:.0f}¢\n"
        f"   Shares: {a.sized_shares:.0f} | Cost: ${a.sized_cost:.2f} | "
        f"Profit if win: ${a.sized_profit_if_win:.2f}\n"
        f"   Win prob: {a.win_probability*100:.0f}% | Realistic EV: +${a.sized_ev:.2f} (max 15x return)\n"
        f"   Market: {a.polymarket_url}\n"
        f"   {kelly_str}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trade: ${a.available_after:.2f}"
    )


def _format_ladder(a: AlertReady, balance: float) -> str:
    bin_lines = ""
    for b in a.bins:
        bin_lines += f"  • {b['label']}: {b['yes_price']*100:.0f}¢\n"

    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"🪜 LADDER: {a.city} {a.station} ({a.target_date})\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"🌡️ Models disagree:\n"
        f"{a.model_summary}\n"
        f"API Status: {_get_sources_string()}\n\n"
        f"━━ BUY ALL {len(a.bins)} BINS ━━\n"
        f"{bin_lines}\n"
        f"Total: ${a.sized_cost:.2f} across {len(a.bins)} bins\n"
        f"Range probability: {a.win_probability*100:.0f}% | Realistic EV: +${a.sized_ev:.2f} (max 15x return)\n"
        f"{kelly_str}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trades: ${a.available_after:.2f}"
    )


def _format_edge_yes(a: AlertReady, balance: float) -> str:
    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"📈 EDGE YES: {a.city} {a.station} ({a.target_date})\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"{a.model_summary}\n"
        f"API Status: {_get_sources_string()}\n\n"
        f"✅ BUY YES: \"{a.bin_label}\" at {a.entry_price*100:.0f}¢\n"
        f"   Shares: {a.sized_shares:.0f} | Cost: ${a.sized_cost:.2f} | "
        f"Profit if win: ${a.sized_profit_if_win:.2f}\n"
        f"   Win prob: {a.win_probability*100:.0f}% | Realistic EV: +${a.sized_ev:.2f} (max 15x return)\n"
        f"   Market: {a.polymarket_url}\n"
        f"   {kelly_str}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trade: ${a.available_after:.2f}"
    )


def _format_edge_no(a: AlertReady, balance: float) -> str:
    yes_price = 1.0 - a.entry_price
    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"📉 EDGE NO: {a.city} {a.station} ({a.target_date})\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"{a.model_summary}\n"
        f"API Status: {_get_sources_string()}\n\n"
        f"🛡️ BUY NO: \"{a.bin_label}\" (YES at {yes_price*100:.0f}¢, NO costs {a.entry_price*100:.0f}¢)\n"
        f"   Shares: {a.sized_shares:.0f} | Cost: ${a.sized_cost:.2f} | "
        f"Profit if win: ${a.sized_profit_if_win:.2f}\n"
        f"   Win prob: {a.win_probability*100:.0f}% | Realistic EV: +${a.sized_ev:.2f} (max 15x return)\n"
        f"   Market: {a.polymarket_url}\n"
        f"   {kelly_str}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trade: ${a.available_after:.2f}"
    )


def _format_edge_ladder(a: AlertReady, balance: float) -> str:
    bin_lines = ""
    for b in a.bins:
        url = b.get("polymarket_url", "")
        if url:
            bin_lines += f"  • {b['label']}: [{b['yes_price']*100:.0f}¢]({url})\n"
        else:
            bin_lines += f"  • {b['label']}: {b['yes_price']*100:.0f}¢\n"

    update_str = f"🔄 Prices updated {a.update_minutes_ago} mins ago\n" if a.is_update else ""
    kelly_str = _format_kelly(a, balance)
    return (
        f"🪜 EDGE LADDER: {a.city} {a.station} ({a.target_date})\n"
        f"{update_str}"
        f"Score: {a.confidence_score}/100 | {_ist_time()}\n\n"
        f"{a.model_summary}\n"
        f"API Status: {_get_sources_string()}\n\n"
        f"━━ BUY ALL {len(a.bins)} BINS ━━\n"
        f"{bin_lines}\n"
        f"Total: ${a.sized_cost:.2f} across {len(a.bins)} bins\n"
        f"Range probability: {a.win_probability*100:.0f}% | EV: +${a.sized_ev:.2f}\n\n"
        f"💰 Real Wallet: ${balance:.2f} | After trades: ${a.available_after:.2f}"
    )


def _get_sources_string() -> str:
    rates = models.get_rate_limit_status()
    if not rates:
        return "OK"
    limited = [k for k, v in rates.items() if v == "limited"]
    if limited:
        return f"OK (Limited: {', '.join(limited)})"
    return "OK (All active)"


# ---------------------------------------------------------------------------
# Combo alert
# ---------------------------------------------------------------------------
async def send_combo_alert(alert: AlertReady, wallet_state):
    """Send combined YES + NO alert."""
    balance = wallet_state.balance if hasattr(wallet_state, "balance") else wallet_state.get("balance", 0)

    no_lines = ""
    total_cost = alert.sized_cost
    for na in alert.combo_alerts:
        yes_price = 1.0 - na.entry_price
        no_lines += (
            f"🛡️ BUY NO: \"{na.bin_label}\" (YES at {yes_price*100:.0f}¢, "
            f"NO costs {na.entry_price*100:.0f}¢)\n"
            f"   Shares: {na.sized_shares:.0f} | Cost: ${na.sized_cost:.2f} | "
            f"Profit: ${na.sized_profit_if_win:.2f}\n\n"
        )
        total_cost += na.sized_cost

    all_win = alert.sized_profit_if_win + sum(na.sized_profit_if_win for na in alert.combo_alerts)

    msg = (
        f"🔒+🛡️ COMBO: {alert.city} {alert.station}\n"
        f"Score: {alert.confidence_score}/100 | {_ist_time()}\n\n"
        f"📡 METAR\n"
        f"{alert.metar_summary}\n"
        f"High locked: {alert.win_probability*100:.0f}%\n\n"
        f"━━ YOUR PLAYS ━━\n\n"
        f"✅ BUY YES: \"{alert.bin_label}\" at {alert.entry_price*100:.0f}¢\n"
        f"   Shares: {alert.sized_shares:.0f} | Cost: ${alert.sized_cost:.2f} | "
        f"Profit: ${alert.sized_profit_if_win:.2f}\n\n"
        f"{no_lines}"
        f"━━ SCENARIOS ━━\n"
        f"All win: +${all_win:.2f}\n"
        f"All lose: -${total_cost:.2f}\n\n"
        f"💰 Wallet: ${balance:.2f} | Total deploy: ${total_cost:.2f}\n"
        f"   After trades: ${balance - total_cost:.2f} idle"
    )

    # Log all alerts in combo
    alert_id = await tracker.log_alert({
        "station": alert.station,
        "trade_type": "combo",
        "bin_label": alert.bin_label,
        "side": "COMBO",
        "entry_price": alert.entry_price,
        "shares": alert.sized_shares,
        "cost": total_cost,
        "confidence_score": alert.confidence_score,
        "ev": alert.sized_ev,
    })
    alert.alert_id = alert_id

    await _send(msg)


# ---------------------------------------------------------------------------
# Resolution notification
# ---------------------------------------------------------------------------
async def send_resolution_notification(trade: dict, resolution_data: dict = None):
    """Send resolution result with P&L."""
    outcome = trade.get("outcome", "unknown")
    pnl = trade.get("profit_loss", 0)
    cost = trade.get("cost", 0)
    sign = "+" if pnl >= 0 else ""
    emoji = "✅" if outcome == "win" else "❌" if outcome == "loss" else "↩️"

    msg = (
        f"{emoji} RESOLVED: {trade.get('station', '?')} {trade.get('target_date', '')}\n\n"
        f"Your position: {trade.get('side', '?')} \"{trade.get('bin_label', '?')}\"\n"
        f"Cost: ${cost:.2f} | Payout: ${trade.get('payout', 0):.2f}\n"
        f"Result: {sign}${abs(pnl):.2f} ({outcome.upper()})\n"
    )
    await _send(msg)


# ---------------------------------------------------------------------------
# Silent day
# ---------------------------------------------------------------------------
async def send_silent_day_message(best_ev: float = 0.0, markets_scanned: int = 0,
                                    stations_scanned: int = 0, balance: float = 0.0):
    msg = (
        f"📊 Low edge day — no trades recommended\n"
        f"Best available EV: ${best_ev:.2f} (below $0.50 threshold)\n"
        f"Markets scanned: {markets_scanned} across {stations_scanned} stations\n"
        f"Action: Preserve capital. Markets will be here tomorrow.\n\n"
        f"💰 Capital: ${balance:.2f}"
    )
    await _send(msg)


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------
async def send_daily_loss_limit_message():
    msg = (
        "🛑 DAILY LOSS LIMIT REACHED\n"
        "Action: No more trades today. Review what went wrong.\n"
        "Alerts resume tomorrow."
    )
    await _send(msg)


# ---------------------------------------------------------------------------
# Model release reminder
# ---------------------------------------------------------------------------
async def send_model_release_reminder(model_name: str, stations: List[str]):
    ist_time = _ist_time()
    stations_str = ", ".join(stations)
    msg = (
        f"⏰ {model_name.upper()} just updated | {ist_time}\n"
        f"Scanning: {stations_str}\n"
        f"Alerts coming in 2-3 minutes if opportunities found."
    )
    await _send(msg)


# ---------------------------------------------------------------------------
# New city alert
# ---------------------------------------------------------------------------
async def send_new_city_alert(city: str, event_title: str = ""):
    msg = (
        f"🆕 NEW CITY: {city} launched on Polymarket\n"
        f"{event_title}\n"
        f"⚠️ First-week mispricing expected — reduced confidence (limited data)\n"
        f"Bot will include in next scan cycle."
    )
    await _send(msg)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
async def send_circuit_breaker_alert(breaker_action: str, config: dict):
    breakers = config.get("circuit_breakers", {})
    for name, cfg in breakers.items():
        if cfg.get("action") == breaker_action:
            await _send(cfg.get("alert", f"⚠️ Circuit breaker: {breaker_action}"))
            return
    await _send(f"⚠️ Circuit breaker activated: {breaker_action}")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
async def send_startup_message(wallet_state, stations_list: List[str]):
    balance = wallet_state.balance if hasattr(wallet_state, "balance") else wallet_state.get("balance", 0)
    total = wallet_state.total_value if hasattr(wallet_state, "total_value") else wallet_state.get("total_value", 0)
    stations_str = ", ".join(stations_list)
    msg = (
        f"✅ Weather-Edge bot online\n"
        f"Capital: ${total:.2f} | Stations: {len(stations_list)} active\n"
        f"Tracking: {stations_str}\n"
        f"Next scan in 5 minutes."
    )
    await _send(msg)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
async def send_weekly_report(report: str):
    await _send(report)


async def send_daily_summary(summary: str):
    await _send(summary)
