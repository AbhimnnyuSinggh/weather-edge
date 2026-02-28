"""
commands.py — Telegram Command Handlers

Handles all Telegram commands: /start, /status, /stations, /week, /today,
/data, /pause, /resume, /help
"""

import io
import logging
from datetime import datetime

import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import alerts
import tracker
import wallet as wallet_mod

logger = logging.getLogger("commands")

# Pause state
_paused = False


def is_paused() -> bool:
    return _paused


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Welcome message when user first starts the bot."""
    msg = (
        "🌤️ Weather-Edge Bot\n\n"
        "I monitor Polymarket weather temperature markets and "
        "send you trading signals based on METAR data, model forecasts, "
        "and probability analysis.\n\n"
        "Commands:\n"
        "/status — Current wallet & positions\n"
        "/stations — Active stations with METAR\n"
        "/today — Today's trades\n"
        "/help — All commands\n\n"
        "Bot is running. Signals will appear here automatically."
    )
    await update.message.reply_text(msg)


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show current wallet status, positions, and P&L."""
    try:
        ws = await wallet_mod.sync()
        balance = ws.balance
        total = ws.total_value
        source_label = "🟢 Live" if ws.source == "api" else "🟡 DB fallback"

        positions_text = ""
        if ws.positions:
            for p in ws.positions:
                sign = "+" if p.unrealized_pnl >= 0 else ""
                positions_text += (
                    f"  • {p.station} {p.side} \"{p.bin_label}\" — "
                    f"{p.shares:.0f} shares @ {p.avg_entry_price*100:.0f}¢ "
                    f"(now {p.current_price*100:.0f}¢, {sign}${p.unrealized_pnl:.2f})\n"
                )
        else:
            positions_text = "  No open positions\n"

        deployed = sum(p.cost for p in ws.positions)
        idle = balance

        msg = (
            f"📊 STATUS | {_ist_now()} | {source_label}\n\n"
            f"💰 Cash: ${balance:.2f}\n"
            f"📈 Total value: ${total:.2f}\n"
            f"📊 Deployed: ${deployed:.2f} | Idle: ${idle:.2f}\n\n"
            f"Open positions ({len(ws.positions)}):\n{positions_text}"
        )

        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("/status error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_stations(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show station info with METAR, biases, and weights."""
    try:
        stations = await tracker.get_active_stations()
        msg = "📡 ACTIVE STATIONS\n\n"

        for s in stations:
            # Get latest METAR
            readings = await tracker.get_recent_metar(s["icao"], hours=1)
            temp_str = "no data"
            if readings:
                r = readings[0]
                temp_str = f"{r['temp_c']:.0f}°C / {r['temp_f']:.0f}°F"

            msg += (
                f"• {s['icao']} ({s['city']}, {s['country']})\n"
                f"  Current: {temp_str}\n"
                f"  Unit: {s['unit']} | Coastal: {'Yes' if s['is_coastal'] else 'No'}\n"
                f"  Bias: ECMWF {s['bias_ecmwf']:+.1f} | GFS {s['bias_gfs']:+.1f} | "
                f"ICON {s['bias_icon']:+.1f}\n\n"
            )

        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("/stations error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_week(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show weekly report."""
    try:
        report = await tracker.generate_weekly_report()
        await update.message.reply_text(report)
    except Exception as e:
        logger.error("/week error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show today's trades and activity."""
    try:
        trades = await tracker.get_today_trades()
        if not trades:
            await update.message.reply_text("📋 No trades today yet.")
            return

        msg = f"📋 TODAY'S TRADES | {_ist_now()}\n\n"
        total_pnl = 0.0

        for t in trades:
            status = "OPEN"
            if t["resolved"]:
                status = t["outcome"].upper() if t["outcome"] else "RESOLVED"
                total_pnl += t["profit_loss"] or 0

            msg += (
                f"• {t['station']} {t['side']} \"{t['bin_label']}\" "
                f"({t['trade_type']})\n"
                f"  Entry: {t['entry_price']*100:.0f}¢ × {t['shares']:.0f} = "
                f"${t['cost']:.2f} | {status}"
            )
            if t["resolved"] and t["profit_loss"] is not None:
                sign = "+" if t["profit_loss"] >= 0 else ""
                msg += f" ({sign}${t['profit_loss']:.2f})"
            msg += "\n\n"

        total_sign = "+" if total_pnl >= 0 else ""
        msg += f"Today P&L: {total_sign}${total_pnl:.2f}"

        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("/today error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_data(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Export all data as CSV zip file."""
    try:
        await update.message.reply_text("📦 Exporting data... please wait.")
        data_bytes = await tracker.export_data()

        if not data_bytes:
            await update.message.reply_text("No data to export yet.")
            return

        doc = io.BytesIO(data_bytes)
        doc.name = "weather_edge_export.zip"
        await update.message.reply_document(
            document=doc,
            caption="📊 Weather-Edge data export (all tables as CSV)",
        )
    except Exception as e:
        logger.error("/data error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Pause alert sending (bot still scans and logs)."""
    global _paused
    _paused = True
    await update.message.reply_text(
        "⏸️ Alerts paused. Bot still scanning and recording data.\n"
        "Use /resume to restart alerts."
    )


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Resume alert sending."""
    global _paused
    _paused = False
    await update.message.reply_text("▶️ Alerts resumed. Next scan in progress.")


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show list of all commands."""
    msg = (
        "🤖 Weather-Edge Bot Commands\n\n"
        "/status — Current wallet, positions, P&L\n"
        "/stations — Active stations with METAR & biases\n"
        "/week — Weekly performance report\n"
        "/today — Today's trades and activity\n"
        "/data — Export all data as CSV zip\n"
        "/pause — Pause alerts (bot keeps scanning)\n"
        "/resume — Resume alerts\n"
        "/help — This message"
    )
    await update.message.reply_text(msg)


async def cmd_setcapital(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set current capital amount. Usage: /setcapital 100"""
    try:
        if not ctx.args:
            await update.message.reply_text(
                "Usage: /setcapital <amount>\nExample: /setcapital 100"
            )
            return
        amount = float(ctx.args[0])
        if amount <= 0:
            await update.message.reply_text("Amount must be positive.")
            return
        wallet_mod.set_manual_capital(amount)
        await update.message.reply_text(
            f"✅ Capital set to ${amount:.2f}\n"
            f"Bot will use this for all position sizing."
        )
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Invalid amount. Usage: /setcapital 100"
        )


# ---------------------------------------------------------------------------
# /took — manual trade entry
# ---------------------------------------------------------------------------
async def cmd_took(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Manually log a trade.
    Usage: /took STATION SIDE "BIN" SHARES PRICE
    Example: /took RKSI YES "8-9°C" 50 0.15
    """
    usage = (
        "Usage: /took STATION SIDE BIN SHARES PRICE\n"
        "Example: /took RKSI YES 8-9°C 50 0.15"
    )
    try:
        if not ctx.args or len(ctx.args) < 5:
            await update.message.reply_text(usage)
            return

        station = ctx.args[0].upper()
        side = ctx.args[1].upper()
        bin_label = ctx.args[2]
        shares = float(ctx.args[3])
        price = float(ctx.args[4])
        cost = shares * price

        trade_id = await tracker.log_manual_trade({
            "station": station,
            "side": side,
            "bin_label": bin_label,
            "shares": shares,
            "entry_price": price,
            "cost": cost,
        })

        await update.message.reply_text(
            f"✅ Trade logged (ID: {trade_id})\n"
            f"  {station} {side} \"{bin_label}\"\n"
            f"  {shares:.0f} shares @ {price*100:.0f}¢ = ${cost:.2f}"
        )
    except Exception as e:
        logger.error("/took error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}\n\n{usage}")


# ---------------------------------------------------------------------------
# /resolve — manual trade resolution
# ---------------------------------------------------------------------------
async def cmd_resolve(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Manually resolve a trade.
    Usage: /resolve TRADE_ID win|loss [ACTUAL_HIGH]
    Example: /resolve 5 win 42.3
    """
    usage = (
        "Usage: /resolve TRADE_ID win|loss [ACTUAL_HIGH_F]\n"
        "Example: /resolve 5 win 42.3"
    )
    try:
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text(usage)
            return

        trade_id = int(ctx.args[0])
        outcome = ctx.args[1].lower()
        actual_high = float(ctx.args[2]) if len(ctx.args) > 2 else None

        if outcome not in ("win", "loss", "push"):
            await update.message.reply_text("Outcome must be: win, loss, or push")
            return

        result = await tracker.resolve_trade(trade_id, outcome, actual_high)
        if result:
            pnl = result.get("profit_loss", 0)
            sign = "+" if pnl >= 0 else ""
            await update.message.reply_text(
                f"✅ Trade #{trade_id} resolved: {outcome.upper()}\n"
                f"  P&L: {sign}${pnl:.2f}"
            )
        else:
            await update.message.reply_text(f"❌ Trade #{trade_id} not found.")
    except Exception as e:
        logger.error("/resolve error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}\n\n{usage}")


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------
def register_handlers(app: Application):
    """Register all command handlers with the Telegram application."""
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("stations", cmd_stations))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("data", cmd_data))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("setcapital", cmd_setcapital))
    app.add_handler(CommandHandler("took", cmd_took))
    app.add_handler(CommandHandler("resolve", cmd_resolve))
    logger.info("Telegram command handlers registered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ist_now() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%I:%M %p IST")
