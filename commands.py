"""
commands.py — Telegram Command Handlers

Handles all Telegram commands: /status, /stations, /week, /today,
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
async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show current wallet status, positions, and P&L."""
    try:
        ws = await wallet_mod.sync()
        balance = ws.balance
        total = ws.total_value

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

        # Today's P&L
        today_trades = await tracker.get_today_trades()
        today_pnl = sum(
            (t["profit_loss"] or 0) for t in today_trades if t["resolved"]
        )
        today_sign = "+" if today_pnl >= 0 else ""

        deployed = sum(p.cost for p in ws.positions)
        idle = balance

        msg = (
            f"📊 STATUS | {_ist_now()}\n\n"
            f"💰 Balance: ${balance:.2f}\n"
            f"📈 Total value: ${total:.2f}\n"
            f"📊 Deployed: ${deployed:.2f} | Idle: ${idle:.2f}\n"
            f"Today P&L: {today_sign}${today_pnl:.2f}\n\n"
            f"Open positions:\n{positions_text}"
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


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------
def register_handlers(app: Application):
    """Register all command handlers with the Telegram application."""
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("stations", cmd_stations))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("data", cmd_data))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("help", cmd_help))
    logger.info("Telegram command handlers registered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ist_now() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%I:%M %p IST")
