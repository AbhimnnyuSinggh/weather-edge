"""
scheduler.py — Model Release Reminders, Scan Frequency Control

Manages time-based events: model release window detection, scan frequency
control (5min ↔ 2min), daily summary, weekly report, historical stats update.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

import pytz

import alerts
import probability
import tracker

logger = logging.getLogger("scheduler")

# Track what we've already sent today to avoid duplicates
_sent_today: dict = {}
_last_sent_date: Optional[date] = None


def _reset_if_new_day():
    global _sent_today, _last_sent_date
    today = date.today()
    if _last_sent_date != today:
        _sent_today = {}
        _last_sent_date = today


# ---------------------------------------------------------------------------
# Scan frequency control
# ---------------------------------------------------------------------------
def check_if_model_release_window(config: dict) -> int:
    """
    Check if we're in a model release window.
    Returns scan interval: 120 (fast) or 300 (normal) seconds.
    """
    now_utc = datetime.utcnow()
    releases = config.get("model_releases", {})
    fast_duration = config.get("bot", {}).get("fast_scan_duration_minutes", 30)
    fast_interval = config.get("bot", {}).get("fast_scan_interval_seconds", 120)
    normal_interval = config.get("bot", {}).get("scan_interval_seconds", 300)

    for release_name, release_cfg in releases.items():
        release_hour = release_cfg.get("utc_hour", 0)
        release_minute = release_cfg.get("utc_minute", 0)

        # Construct today's release time
        release_time = now_utc.replace(
            hour=release_hour, minute=release_minute, second=0, microsecond=0
        )

        # Check if within window
        diff_minutes = (now_utc - release_time).total_seconds() / 60.0
        if 0 <= diff_minutes <= fast_duration:
            logger.info(
                "Fast scan mode: %s released %d min ago",
                release_name, int(diff_minutes),
            )
            return fast_interval

    return normal_interval


# ---------------------------------------------------------------------------
# Scheduled events
# ---------------------------------------------------------------------------
async def check_scheduled_events(config: dict, wallet_state, stations_cfg: dict):
    """
    Check for time-based events. Called every scan cycle.
    """
    _reset_if_new_day()
    now_utc = datetime.utcnow()
    ist_tz = pytz.timezone(config.get("bot", {}).get("timezone", "Asia/Kolkata"))
    now_ist = datetime.now(ist_tz)

    # --- 1. Model release reminders ---
    releases = config.get("model_releases", {})
    for release_name, release_cfg in releases.items():
        release_hour = release_cfg.get("utc_hour", 0)
        release_minute = release_cfg.get("utc_minute", 0)
        applies_to = release_cfg.get("applies_to", [])

        release_time = now_utc.replace(
            hour=release_hour, minute=release_minute, second=0, microsecond=0
        )
        diff_minutes = abs((now_utc - release_time).total_seconds() / 60.0)

        reminder_key = f"release_{release_name}"
        if diff_minutes <= 2 and reminder_key not in _sent_today:
            # Filter to only active stations
            active = [s for s in applies_to if s in stations_cfg]
            if active:
                await alerts.send_model_release_reminder(release_name, active)
                _sent_today[reminder_key] = True
                logger.info("Sent model release reminder: %s", release_name)

    # --- 2. Daily summary (midnight IST) ---
    if now_ist.hour == 0 and 0 <= now_ist.minute <= 5:
        if "daily_summary" not in _sent_today:
            try:
                summary = await tracker.generate_daily_summary_text(
                    {"total_value": wallet_state.total_value
                     if hasattr(wallet_state, "total_value")
                     else wallet_state.get("total_value", 0)}
                )
                await alerts.send_daily_summary(summary)
                _sent_today["daily_summary"] = True
                logger.info("Sent daily summary")
            except Exception as e:
                logger.error("Daily summary error: %s", e)

    # --- 3. Weekly report (Sunday 10 AM IST) ---
    if now_ist.weekday() == 6 and now_ist.hour == 10 and 0 <= now_ist.minute <= 5:
        if "weekly_report" not in _sent_today:
            try:
                report = await tracker.generate_weekly_report()
                await alerts.send_weekly_report(report)
                _sent_today["weekly_report"] = True
                logger.info("Sent weekly report")
            except Exception as e:
                logger.error("Weekly report error: %s", e)

    # --- 4. Historical stats update (1 AM IST daily) ---
    if now_ist.hour == 1 and 0 <= now_ist.minute <= 5:
        if "stats_update" not in _sent_today:
            try:
                for icao, cfg in stations_cfg.items():
                    await probability.populate_historical_stats(
                        icao, cfg["timezone"]
                    )
                _sent_today["stats_update"] = True
                logger.info("Historical stats updated")
            except Exception as e:
                logger.error("Stats update error: %s", e)


# ---------------------------------------------------------------------------
# Model fetch timing
# ---------------------------------------------------------------------------
_last_model_fetch: Optional[datetime] = None


def should_fetch_models(config: dict) -> bool:
    """
    Determine if we should fetch fresh model data this cycle.
    True if: >30 min since last fetch OR we're in a model release window.
    """
    global _last_model_fetch

    if _last_model_fetch is None:
        return True

    elapsed = (datetime.utcnow() - _last_model_fetch).total_seconds() / 60.0

    # Always fetch if >30 minutes
    if elapsed > 30:
        return True

    # Fetch if in model release window
    interval = check_if_model_release_window(config)
    if interval < 300:  # fast mode = model just released
        return True

    return False


def mark_models_fetched():
    """Record that we just fetched model data."""
    global _last_model_fetch
    _last_model_fetch = datetime.utcnow()


# ---------------------------------------------------------------------------
# End of trading day check
# ---------------------------------------------------------------------------
def is_end_of_trading_day(stations_cfg: dict) -> bool:
    """
    Check if it's past the latest station's trading hours.
    Used to trigger silent day message if no alerts were sent.
    """
    now_utc = datetime.utcnow()
    latest_end = 0

    for icao, cfg in stations_cfg.items():
        tz = pytz.timezone(cfg["timezone"])
        now_local = datetime.now(tz)
        # Consider end of trading as 10 PM local
        if now_local.hour >= 22:
            latest_end += 1

    # If all stations are past 10 PM
    return latest_end == len(stations_cfg)
