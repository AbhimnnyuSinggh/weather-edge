"""
commands.py — Telegram Command Handlers

Handles all Telegram commands: /start, /status, /stations, /week, /today,
/data, /pause, /resume, /help
"""

import io
import logging
from datetime import date as _date_type

# ── P6 FIX: Global portfolio state — prevents over-deployment across cities ──
_session_deployed = 0.0
_session_deployed_reset_date = None
from datetime import datetime

import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import alerts
import tracker
import wallet as wallet_mod
import markets
import metar as metar_mod
import models as models_mod
import yaml
import os

logger = logging.getLogger("commands")

def _get_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml") if "USER_AGENT" not in globals() else "config.yaml"
    if os.path.exists("config.yaml"):
        cfg_path = "config.yaml"
    try:
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

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
        "🌤 **WEATHER-EDGE DASHBOARD**\n"
        "Select a city to scan for Polymarket edge:\n\n"
        "🇺🇸 **US CITIES** (F)\n"
        "/nyc - New York\n"
        "/chi - Chicago\n"
        "/mia - Miami\n"
        "/atl - Atlanta\n"
        "/den - Denver\n"
        "/hou - Houston\n"
        "/phx - Phoenix\n"
        "/dal - Dallas\n"
        "/sea - Seattle\n"
        "/bos - Boston\n\n"
        "🌍 **INTL CITIES** (C)\n"
        "/sel - Seoul\n"
        "/lon - London\n"
        "/tok - Tokyo\n"
        "/par - Paris\n"
        "/syd - Sydney"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


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

async def cmd_ledger(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show the real-time Polymarket Accountability Ledger for Auto-Sniper performance."""
    try:
        import ledger
        report = await ledger.generate_ledger_report()
        await update.message.reply_text(report, parse_mode="Markdown")
    except Exception as e:
        logger.error("/ledger error: %s", e)
        await update.message.reply_text(f"❌ Error compiling ledger: {e}")

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
        "/ledger — Autonomous execution tracker (Trailing ROI)\n"
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
# /temp — Live Temperature Dashboard
# ---------------------------------------------------------------------------
async def cmd_city_analysis(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Handle /<city> commands (e.g., /mia).
    Fetches live data and returns a full trading dashboard.
    """
    # 1. Parse which city the user requested
    text = update.message.text or ""
    cmd = text.split()[0].replace("/", "").lower()
    
    city_config = markets.CITIES.get(cmd)
    if not city_config:
        await update.message.reply_text(f"❌ Unknown city command: /{cmd}")
        return
        
    icao = city_config["icao"]
    city_name = city_config["city"]
    unit = city_config["unit"]
    tz_name = city_config["tz"]

    await update.message.reply_text(f"🔄 Scanning Polymarket and models for {city_name}...")

    try:
        # Determine target date (tomorrow, or today depending on resolution rules)
        # PolyWeather typically targets Tomorrow based on exchange rules, but for simplicity we'll assume next resolution day.
        # Let's get "today" in local timezone
        local_tz = pytz.timezone(tz_name)
        now_local = datetime.now(local_tz)
        target_date = now_local.date()
        
        # 2. Fetch Active Market
        market_group = await markets.fetch_city_market(cmd, target_date)
        if not market_group or not market_group.bins:
            # If nothing found for today, try tomorrow
            import datetime as dt
            target_date = target_date + dt.timedelta(days=1)
            market_group = await markets.fetch_city_market(cmd, target_date)
            
        market_link = "No Active Market"
        if market_group and market_group.bins:
            market_link = f"[{city_name} Polymarket]({market_group.bins[0].polymarket_url})"

        # 3. Fetch METAR
        raw_metar = await metar_mod.fetch_all_stations([icao])
        station_metar = None
        now_temp_str = f"—°{unit}"
        high_so_far_str = ""
        high_so_far_val = None
        metar_trend = {}
        
        if icao in raw_metar:
            station_metar = await metar_mod.enrich_metar(raw_metar[icao], tz_name, unit)
            if station_metar:
                now_t = station_metar.temp_c if unit == "C" else station_metar.temp_f
                now_temp_str = f"{now_t:.0f}°{unit}"
                if station_metar.velocity:
                    high_so_far_val = station_metar.velocity.day_high if unit == "C" else station_metar.velocity.day_high_f
                    high_time_str = station_metar.velocity.high_time.strftime('%I:%M %p') if station_metar.velocity.high_time else 'recently'
                    high_so_far_str = f" (hit at {high_time_str})"
            
            metar_trend = await metar_mod.analyze_metar_trend(icao, tz_name, unit)

        # 4. Fetch All Models ON DEMAND
        city_config["target_date"] = target_date
        models_raw = await models_mod.fetch_all_stations({icao: city_config}, use_cache_fallback=False)
        models_data = models_raw.get(icao, {})
        
        ensemble_members = await models_mod.fetch_ensemble(city_config["lat"], city_config["lon"], unit)

        # ─────────────────────────────────────────────────────────────
        # THE REALITY FLOOR (MATHEMATICAL CLAMP TO OBSERVED METAR)
        # If today's known high is already 81°F, any model predicting 
        # 78°F is objectively wrong and must be clamped to reality.
        # ─────────────────────────────────────────────────────────────
        if target_date == now_local.date() and high_so_far_val is not None:
            for mn, fc in models_data.items():
                if unit == "F":
                    if fc.bias_corrected_f < high_so_far_val:
                        fc.bias_corrected_f = float(high_so_far_val)
                        fc.bias_corrected_c = (float(high_so_far_val) - 32.0) * 5.0 / 9.0
                        fc.raw_high_f = max(fc.raw_high_f, float(high_so_far_val))
                else:
                    if fc.bias_corrected_c < high_so_far_val:
                        fc.bias_corrected_c = float(high_so_far_val)
                        fc.bias_corrected_f = float(high_so_far_val) * 9.0 / 5.0 + 32.0
                        fc.raw_high_c = max(fc.raw_high_c, float(high_so_far_val))
                        
            if ensemble_members:
                ensemble_members = [max(m, float(high_so_far_val)) for m in ensemble_members]

        # 5. Calculate Predicted Daily High
        predicted_high = await models_mod.calculate_daily_high(
            models_data, station_metar, metar_trend, ensemble_members, now_local.hour, icao, unit
        )
        
        # --- CALCULATE UNCERTAINTY SIGMA FOR PDF ---
        import numpy as np
        all_act = [k for k, v in models_data.items() if v]
        val_mods = [models_data[k] for k in all_act if models_data.get(k)]
        
        # FIX 5 (Updated): Use FULL model spread but EXCLUDE extreme outliers
        all_temps_raw = [(f.bias_corrected_f if unit == "F" else f.bias_corrected_c) 
                         for f in val_mods if f and getattr(f, 'bias_corrected_f' if unit == 'F' else 'bias_corrected_c', None)]
        
        if all_temps_raw:
            median_t = sorted(all_temps_raw)[len(all_temps_raw) // 2]
            outlier_threshold = 10.0 if unit == "F" else 5.5
            all_temps_list = [t for t in all_temps_raw if abs(t - median_t) <= outlier_threshold]
        else:
            all_temps_list = []
        
        trusted_t = []
        for f in val_mods:
            temp = f.bias_corrected_f if unit == "F" else f.bias_corrected_c
            diff = abs(temp - predicted_high)
            if getattr(f, "weight", 1.0) > 0.1 and diff < 3.5:
                trusted_t.append(temp)
                
        if trusted_t and len(trusted_t) >= 3:
            std_dev = np.std(trusted_t)
            uncertainty = std_dev * 1.5
            
            # FIX 5: Wider Sigma (Sigma should be at least half the model spread, min 1.5°F)
            if all_temps_list:
                full_range = max(all_temps_list) - min(all_temps_list)
                min_sigma = 1.5 if unit == "F" else 0.8
                uncertainty = max(min_sigma, full_range / 2.0, uncertainty)
            
            # ── P2 FIX: Detect bimodal distribution ──
            # When models split into two camps (e.g., 8 say 40-41°F, 5 say 48-50°F),
            # a single bell curve centered on the weighted average misrepresents
            # the actual probability landscape. Widen sigma to cover both camps.
            if all_temps_list and len(all_temps_list) >= 6:
                sorted_at = sorted(all_temps_list)
                mid = len(sorted_at) // 2
                lower_half = sorted_at[:mid]
                upper_half = sorted_at[mid:]
                lower_mean = sum(lower_half) / len(lower_half)
                upper_mean = sum(upper_half) / len(upper_half)
                gap = upper_mean - lower_mean
                bimodal_threshold = 4.0 if unit == "F" else 2.2
                if gap > bimodal_threshold:
                    uncertainty = max(uncertainty, gap * 0.75)
                    logger.info("BIMODAL detected: lower camp=%.1f, upper camp=%.1f, gap=%.1f, new sigma=%.1f",
                                lower_mean, upper_mean, gap, uncertainty)
        else:
            uncertainty = 3.0 if unit == "F" else 1.5

        m_high_floor = "No"
        if high_so_far_val and predicted_high == high_so_far_val:
            m_high_floor = "Yes"

        # 6. Distribution / Edge Scan
        import distribution
        import signals
        probs = distribution.calculate_bin_probabilities(
            market_group.bins if market_group else [], 
            predicted_high,
            uncertainty,
            high_so_far_val
        )
        
        # Compute confidence label early (needed by analyze_market)
        within_2_early = sum(1 for f in val_mods 
                            if abs((f.bias_corrected_f if unit == "F" else f.bias_corrected_c) - predicted_high) <= 2.0)
        conf_frac_early = within_2_early / max(1, len(val_mods))
        if conf_frac_early >= 0.85: conf_lbl = "HIGH"
        elif conf_frac_early >= 0.60: conf_lbl = "MEDIUM"
        else: conf_lbl = "LOW"
        
        # 8. Signals & Trades
        ws = await wallet_mod.get_capital_summary()
        total_cap = ws["total_value"]
        reserve = total_cap * 0.15
        
        # ── P6 FIX: Global portfolio state — cap deployment across all cities ──
        global _session_deployed, _session_deployed_reset_date
        today_date = now_local.date()
        if _session_deployed_reset_date != today_date:
            _session_deployed = 0.0
            _session_deployed_reset_date = today_date
        remaining_budget = max(0, (total_cap * 0.25) - _session_deployed)
        effective_cap = min(total_cap, remaining_budget + reserve)  # Never exceed 25% daily deployment
        
        if market_group and market_group.bins:
            trade_instructions, deployed = signals.analyze_market(
                market_group, probs, effective_cap, now_local.hour,
                predicted_high=predicted_high,
                models_data=models_data,
                metar_high=high_so_far_val,
                confidence=conf_lbl
            )
            _session_deployed += deployed  # Track cumulative deployment
            deployable = max(0, effective_cap - reserve)
            remaining = max(0, deployable - deployed)
        else:
            trade_instructions, deployed, deployable, remaining = {}, 0, 0, 0
            
        # --- BUILD FINAL DASHBOARD ---
        
        # Peak estimation
        current_month = now_local.month
        PEAK_HOURS = {
            "KMIA": {1: 15, 2: 15, 3: 15, 4: 15, 5: 14, 6: 14, 7: 14, 8: 14, 9: 15, 10: 15, 11: 15, 12: 15},
            "KLGA": {1: 14, 2: 14, 3: 15, 4: 15, 5: 15, 6: 16, 7: 16, 8: 16, 9: 15, 10: 15, 11: 14, 12: 14},
            "KORD": {1: 14, 2: 14, 3: 15, 4: 15, 5: 15, 6: 16, 7: 16, 8: 15, 9: 15, 10: 15, 11: 14, 12: 14},
            "RKSI": {1: 14, 2: 14, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 15, 9: 15, 10: 14, 11: 14, 12: 14},
            "EGLC": {1: 14, 2: 14, 3: 15, 4: 15, 5: 16, 6: 16, 7: 16, 8: 16, 9: 15, 10: 14, 11: 14, 12: 14},
        }
        peak_hour = PEAK_HOURS.get(icao, {}).get(current_month, 15)
        
        peak_start_local = now_local.replace(hour=peak_hour-1, minute=0, second=0, microsecond=0)
        peak_end_local = now_local.replace(hour=peak_hour+1, minute=0, second=0, microsecond=0)
        ist_tz = pytz.timezone("Asia/Kolkata")
        peak_start_ist = peak_start_local.astimezone(ist_tz)
        peak_end_ist = peak_end_local.astimezone(ist_tz)
        
        target_dt_str = target_date.strftime('%b %-d, %Y')
        now_ist_str = datetime.now(ist_tz).strftime('%I:%M %p')
        
        dashboard_msg = [
            f"🌡️ {city_name.upper()} ({icao}) — Forecast for: {target_dt_str} | {now_ist_str} IST",
            "",
            "━━ LIVE STATUS ━━",
            f"🌡️ Current temp: {now_temp_str} (METAR recently)",
            f"📈 Today's high so far: {high_so_far_val:.0f}°{unit}{high_so_far_str}" if high_so_far_val else f"📈 Today's high so far: —",
            f"🕐 Local time: {now_local.strftime('%I:%M %p %Z')}",
            f"   High likely to hit: {peak_start_local.strftime('%I:%M')} - {peak_end_local.strftime('%I:%M %p %Z')} ({peak_start_ist.strftime('%I:%M')} - {peak_end_ist.strftime('%I:%M %p')} IST)",
            "",
            f"━━ MODEL FORECASTS ({target_dt_str} High) ━━"
        ]

        ABBR = {
            "gfs": "GFS", "ecmwf": "ECMWF", "icon": "ICON", "gem": "GEM", 
            "jma": "JMA", "hrrr": "HRRR", "nbm": "NBM", "arpege": "ARP",
            "ukmo": "UKMO", "bom": "BOM", "nws": "NWS", "noaa_mos": "MOS", 
            "visual_crossing": "VC", "ensemble": "ENS", "tomorrow": "TMRW"
        }
        
        m1, m2, m3 = [], [], []
        
        # Determine valid model sets based on what actually came back and what the city supports
        supported_models = city_config.get("models", ["gfs", "ecmwf", "icon", "gem", "jma"])
        
        m1_keys = [m for m in ["gfs", "ecmwf", "icon", "gem", "jma"] if m in supported_models]
        m2_keys = [m for m in ["nws", "noaa_mos", "hrrr", "nbm", "arpege", "ukmo", "bom", "visual_crossing"] if m in supported_models]
        
        # Always allow NWS, MOS, VC, Tomorrow if they are present for US cities, otherwise VC/Tomorrow
        is_us = city_config.get("country", "US") == "US"
        m3_keys = ["nws", "noaa_mos", "visual_crossing", "tomorrow"] if is_us else ["visual_crossing", "tomorrow"]
        
        # Consolidate into 3 actual rows for display formatting based on the total active set
        all_active_keys = [k for k, v in models_data.items() if v]
        
        display_m1 = [k for k in ["gfs", "ecmwf", "icon", "gem", "jma"] if k in all_active_keys]
        display_m2 = [k for k in ["hrrr", "nbm", "arpege", "ukmo", "bom"] if k in all_active_keys]
        display_m3 = [k for k in ["nws", "noaa_mos", "visual_crossing", "tomorrow"] if k in all_active_keys]

        def fmt_temp(mn):
            fc = models_data.get(mn)
            if not fc: return "—"
            if mn in ["nws", "noaa_mos", "visual_crossing", "tomorrow"] and is_us:
                return f"{fc.bias_corrected_f:.0f}°F"
            if unit == "C":
                return f"{fc.bias_corrected_c:.0f}°C"
            return f"{fc.bias_corrected_f:.0f}°F"

        for mn in ["gfs", "ecmwf", "icon", "gem", "jma"]:
            if mn in supported_models:
                m1.append(f"{ABBR.get(mn, mn)}: {fmt_temp(mn)}")
                
        for mn in ["hrrr", "nbm", "arpege", "ukmo", "bom"]:
            if mn in supported_models:
                m2.append(f"{ABBR.get(mn, mn)}: {fmt_temp(mn)}")
                
        for mn in ["nws", "noaa_mos", "visual_crossing", "tomorrow"]:
            if mn in m3_keys:  # m3_keys handles the US check implicitly
                m3.append(f"{ABBR.get(mn, mn)}: {fmt_temp(mn)}")
            
        dashboard_msg.append("  " + " | ".join(m1))
        if m2:
            dashboard_msg.append("  " + " | ".join(m2))
        dashboard_msg.append("  " + " | ".join(m3))
            
        # Tally the total expected models dynamically based on country
        base_om_count = len(m1_keys) + len(m2_keys)
        total_models_expected = base_om_count + len(m3_keys)
        
        reporting_count = len(display_m1) + len(display_m2) + len(display_m3)
        valid_models = [models_data[k] for k in all_active_keys if models_data.get(k)]
        
        # Calculate expected based on intl vs US
        if city_config.get("country", "US") == "US":
            total_models = len(m1_keys) + len(m2_keys) + len(m3_keys)
        else:
            # HRRR and NBM are missing for intl
            total_models = len(m1_keys) + (len(m2_keys)-2) + len(m3_keys)
            
        dashboard_msg.append(f"  ✅ {reporting_count}/{total_models} models reporting")
        
        # Confidence label & Model Spread (Standard Deviation Cluster Analysis)
        import numpy as np
        
        within_2 = 0
        trusted_temps = []
        
        for f in valid_models:
            temp = f.bias_corrected_f if unit == "F" else f.bias_corrected_c
            
            # Confidence metric (within 2 degrees of Bayesian Final)
            diff = abs(temp - predicted_high)
            if diff <= 2.0:
                within_2 += 1
                # (Uncertainty calculation mathematically hoisted above)
            
        # Enforce physical reality limits for the UI String
        current_high = high_so_far_val
            
        if current_high:
            local_hour = now_local.hour
            hours_left = max(0, 24 - local_hour)
            
            # Only apply ceiling-based clamping AFTER peak hour.
            # Before peak, morning warming can be 3-5°F/hr — a 0.5°F/hr
            # ceiling collapses the range to a single point (the bug).
            if local_hour >= 14:
                max_warming = 0.5 if unit == "F" else 0.3
                ceiling = current_high + (max_warming * hours_left)
                
                # --- METAR Velocity Boost (Aggressive Afternoon Tightening) ---
                if station_metar and hasattr(station_metar, 'velocity') and station_metar.velocity:
                    vel = station_metar.velocity.velocity_1h_f if unit == "F" else station_metar.velocity.velocity_1h
                    trend_hours = station_metar.velocity.trend_hours
                    if vel is not None and vel < -0.6 and trend_hours >= 3.0:
                        ceiling_reduction = abs(vel) * (hours_left * 0.65)
                        ceiling -= ceiling_reduction
                        ceiling = max(ceiling, current_high)
                
                range_low = max(current_high, predicted_high - uncertainty)
                range_high = min(ceiling, predicted_high + uncertainty)
                range_high = max(range_high, range_low)
            else:
                # Before peak: use METAR floor but don't cap the ceiling
                range_low = max(current_high, predicted_high - uncertainty)
                range_high = predicted_high + uncertainty
                range_high = max(range_high, range_low)
            
            final_range = f"{range_low:.1f}–{range_high:.1f}°{unit} (±{uncertainty:.1f}°)"
        else:
            final_range = f"{(predicted_high - uncertainty):.1f}–{(predicted_high + uncertainty):.1f}°{unit} (±{uncertainty:.1f}°)"
                
        conf_frac = within_2 / max(1, reporting_count)
        if conf_frac >= 0.85: conf_lbl = "HIGH"
        elif conf_frac >= 0.60: conf_lbl = "MEDIUM"
        else: conf_lbl = "LOW"
        detail = f"{within_2}/{reporting_count} models within 2°{unit}"
        
        dashboard_msg.extend([
            "",
            f"━━ PREDICTED DAILY HIGH: {predicted_high:.1f}°{unit} | {final_range} ━━",
            f"  Bin: {market_link} | Confidence: {conf_lbl} ({detail})"
        ])
        
        if metar_trend and metar_trend.get("projected_high"):
            trend_val = metar_trend["projected_high"]
            trend_dir = "📈 Rising" if metar_trend.get("is_rising") else "📉 Falling"
            vel = metar_trend.get("velocity", 0)
            dashboard_msg.append(f"  METAR Trend: {trend_val:.1f}°{unit} ({trend_dir} at {vel:+.2f}°/hr)")
            
        dashboard_msg.extend([
            "",
            f"━━ MARKET EDGE SCAN ━━",
            f"  Bin        Price   Our Prob   Edge     Signal"
        ])
        
        if not market_group or not market_group.bins:
            dashboard_msg.append(f"  No active Polymarket market found for {city_name} today.")
            dashboard_msg.append(f"  Models predict {predicted_high}°{unit} — watch for market creation.")
        else:
            for b in market_group.bins:
                lbl = b.bin.label
                model_pct = probs.get(lbl, 0)
                mkt_pct = b.yes_price
                edge = model_pct - mkt_pct
                edge_str = f"{edge*100:+.0f}%"
                
                sig = "—"
                if edge > 0.10: sig = "🟢 BUY YES"
                elif edge > 0.05: sig = "🟡 WEAK YES"
                elif edge < -0.10 and (1-model_pct) - (1-mkt_pct) > 0.08: sig = "🔴 BUY NO"
                
                dashboard_msg.append(f"  {lbl:<10}  {mkt_pct*100:2.0f}¢    {model_pct*100:2.0f}%     {edge_str:<6}   {sig}")

        dashboard_msg.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📍 RECOMMENDED TRADES",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 Your balance: ${total_cap:.2f} | Reserve (15%): ${reserve:.2f}",
            f"   Deployable: ${deployable:.2f}"
        ])
        
        for idx, (t_type, t_cfg) in enumerate(trade_instructions.items()):
            if not t_cfg.get("valid"):
                dashboard_msg.extend([
                    f"TRADE {idx+1} — {t_cfg.get('label', t_type)}",
                    f"  {t_cfg.get('skip_reason', 'Skipped')}",
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                 ])
            else:
                dashboard_msg.extend([
                    f"TRADE {idx+1} — {t_cfg['label']}",
                    f"  {t_cfg['action_emoji']} {t_cfg['action']} \"{t_cfg['bin_label']}\" at {t_cfg['price']}¢",
                    f"  Allocation: {t_cfg['alloc_pct']}% = ${t_cfg['alloc_amount']:.2f}",
                    f"  Shares: {t_cfg['shares']:.2f} | Cost: ${t_cfg['cost']:.2f}",
                    f"  If win: ${t_cfg['payout']:.2f} payout → +${t_cfg['profit']:.2f} profit",
                    f"  If lose: -${t_cfg['cost']:.2f}",
                    f"  EV: ({t_cfg['win_prob']}% × ${t_cfg['profit']:.2f}) - ({t_cfg['lose_prob']}% × ${t_cfg['cost']:.2f}) = +${t_cfg['ev']:.2f}",
                    f"  Edge: +{t_cfg['edge']}% | Win prob: {t_cfg['win_prob']}%",
                    "",
                    "  HOW TO EXECUTE:",
                    f"  1. Open: {market_link}",
                    f"  2. Find bin \"{t_cfg['bin_label']}\"",
                    f"  3. Click {t_cfg['side']} → set price to {t_cfg['price']}¢ or lower",
                    f"  4. Enter ${t_cfg['cost']:.2f} or {t_cfg['shares']:.0f} shares",
                    "  5. Click \"Buy\"",
                    "",
                    f"⏰ TIMING: {t_cfg['timing_advice']}",
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                ])
                
        dashboard_msg.extend([
            "━━ CAPITAL DEPLOYMENT ━━",
            f"  Available:     ${deployable:.2f}"
        ])
        for idx, (t_type, t_cfg) in enumerate(trade_instructions.items()):
            if t_cfg.get("valid"): dashboard_msg.append(f"  Trade {idx+1} ({t_cfg['side']}): -${t_cfg['cost']:.2f}")
            else: dashboard_msg.append(f"  Trade {idx+1} ({t_cfg.get('label', 'N/A')[:4]}): SKIPPED")
        
        dashboard_msg.extend([
            "  ─────────────",
            f"  Remaining:     ${remaining:.2f}",
            "",
            "━━ RISK NOTES ━━"
        ])
        
        if city_config.get("is_coastal"):
            dashboard_msg.append("⚠️ Coastal City - Sea breeze / Humidity volatility.")
            
        if high_so_far_val and high_so_far_val > predicted_high + 1:
            dashboard_msg.append(f"⚠️ METAR already {high_so_far_val:.0f}°{unit}, crushing model forecasts.")
        else:
            dashboard_msg.append("📊 METAR behaving as models expect.")
            
        # Send full dashboard
        # Join exactly to respect markdown and newlines
        await update.message.reply_text("\n".join(dashboard_msg))

    except Exception as e:
        logger.error("/%s error: %s", cmd, e)
        await update.message.reply_text(f"❌ Error generating dashboard: {e}")


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------
async def cmd_forcesnipe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Hidden override command to execute a minimum $1.00 micro-trade to verify L2 API cryptographic derivation."""
    await update.message.reply_text("🔓 **FORCE SNIPE INITIATED**\nBypassing clocks. Engaging Polygon L2 Crypto-Derivation Engine to find a dead test bin...", parse_mode="Markdown")
    try:
        from markets import fetch_city_market, CITIES
        from wallet import place_clob_order
        from datetime import datetime, date
        import pytz
        
        # Scan all cities to find any mathematically dead YES share
        target_bin = None
        target_price = 0.0
        
        for city_key, city_data in CITIES.items():
            tz = pytz.timezone(city_data["tz"])
            today = datetime.now(tz).date()
            Group = await fetch_city_market(city_key, today)
            
            if Group:
                for m in Group.bins:
                    if m.yes_price > 0 and m.yes_price <= 0.05 and m.token_id:
                        target_bin = m
                        target_price = m.yes_price
                        break
            
            if target_bin:
                break
                
        if not target_bin:
            await update.message.reply_text("❌ Could not find a test bin cheaper than 5 cents. Aborting to protect capital.")
            return
            
        # Polymarket strictly requires a minimum order size of $1.00. 
        # We calculate the exact number of shares needed to barely clear the $1.00 hurdle.
        shares_to_buy = max(1, int(1.05 / target_price))
        total_risk = shares_to_buy * target_price
            
        await update.message.reply_text(f"🎯 **TARGET ACQUIRED**\n`{target_bin.bin.label}`\nPolymarket minimum order is $1.00.\nAttempting to buy **{shares_to_buy}** YES shares @ {target_price*100:.1f}¢ (Total Risk: ${total_risk:.2f})...", parse_mode="Markdown")
        
        success, detail = await place_clob_order(target_bin.token_id, "YES", shares_to_buy, target_price)
        
        if success:
            await update.message.reply_text(f"✅ **TEST TRADE CONFIRMED**\nThe Level 2 Passwords were successfully synthesized and the Block was minted!\nCapital Deployed: ${total_risk:.2f}\nCLOB Response: `{detail[:200]}`\n\nThe Auto-Sniper is unequivocally lethal and armed.", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ **TEST TRADE FAILED**\nExecution engine failed to mint the transaction.\n\n**Reason:** `{detail}`", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ **FATAL ERROR**: {str(e)}")

def register_handlers(app: Application):
    """Register all command handlers with the Telegram application."""
    app.add_handler(CommandHandler("forcesnipe", cmd_forcesnipe))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("stations", cmd_stations))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("ledger", cmd_ledger))
    app.add_handler(CommandHandler("data", cmd_data))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("setcapital", cmd_setcapital))
    app.add_handler(CommandHandler("took", cmd_took))
    app.add_handler(CommandHandler("resolve", cmd_resolve))
    # Register 15 City Commands
    for city_cmd in markets.CITIES.keys():
        app.add_handler(CommandHandler(city_cmd, cmd_city_analysis))
    logger.info("Telegram command handlers registered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ist_now() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%I:%M %p IST")
