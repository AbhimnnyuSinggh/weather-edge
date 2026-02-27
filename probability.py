"""
probability.py — Bin Decay, High-Already-Set, Momentum Divergence

The math engine. Physics-based probabilities:
1. "High Already Set" — probability the day's high won't be exceeded
2. Bin Probability Decay — which bins are mathematically impossible
3. Historical stats population from METAR history
"""

import logging
import math
from datetime import datetime, date
from typing import Dict, List, Optional

import aiohttp
import pytz

import tracker
from metar import StationMETAR, detect_wind_shift

logger = logging.getLogger("probability")

AVIATION_WX_URL = "https://aviationweather.gov/api/data/metar"


# ---------------------------------------------------------------------------
# High Already Set
# ---------------------------------------------------------------------------
async def calculate_high_already_set(station: str, metar_data: StationMETAR,
                                       current_local_hour: int,
                                       is_coastal: bool,
                                       recent_readings: List[StationMETAR] = None
                                       ) -> float:
    """
    Combine three signals to estimate P(current high = final high):
    1. Historical time-of-high distribution
    2. Current METAR velocity
    3. Acceleration

    Returns float 0.0–1.0
    """
    month = datetime.utcnow().month

    # --- Signal 1: Historical time-of-high ---
    stats = await tracker.get_time_of_high(station, month)
    p_historical = 0.5  # default if no data
    if stats:
        # Sum frequency for hours <= current hour
        cumulative = sum(
            float(s["frequency"]) for s in stats
            if s["hour_local"] <= current_local_hour
        )
        p_historical = min(1.0, cumulative)

    # --- Signal 2: METAR velocity ---
    p_velocity = 0.0
    if metar_data.velocity and metar_data.velocity.trend == "falling":
        vel_mag = abs(metar_data.velocity.velocity)
        hours_falling = metar_data.velocity.hours_since_high
        # P = 1 - exp(-velocity_magnitude * hours_falling / 2.0)
        p_velocity = 1.0 - math.exp(-vel_mag * max(0, hours_falling) / 2.0)
        p_velocity = min(1.0, max(0.0, p_velocity))

    # --- Signal 3: Acceleration ---
    acceleration_factor = 1.0
    if metar_data.velocity:
        acc = metar_data.velocity.acceleration
        # Negative acceleration = decline getting steeper = boost
        # Positive acceleration = decline slowing = slight reduce
        acceleration_factor = 1.0 + (acc * 0.1)
        acceleration_factor = max(0.8, min(1.2, acceleration_factor))

    # --- Combined: noisy-OR ---
    # P_high_set = 1 - (1-P_hist) * (1-P_vel) * accel_factor
    p_combined = 1.0 - (1.0 - p_historical) * (1.0 - p_velocity) * acceleration_factor
    p_combined = max(0.0, min(1.0, p_combined))

    # --- Coastal penalty ---
    if is_coastal and current_local_hour < 16:
        if recent_readings and detect_wind_shift(recent_readings):
            p_combined *= 0.85  # 15% reduction for secondary peak risk
            logger.info("%s: Coastal wind shift penalty applied (P=%.3f)", station, p_combined)

    return round(p_combined, 4)


# ---------------------------------------------------------------------------
# Bin Probability Decay
# ---------------------------------------------------------------------------
async def calculate_bin_decay(station: str, metar_data: StationMETAR,
                                bins: list,
                                current_local_hour: int,
                                unit: str = "C") -> Dict[str, float]:
    """
    For each bin above the current day's high, calculate probability
    that temperature could still reach that bin before midnight.

    Returns: {'15-16°C': 0.001, '14-15°C': 0.03, ...}
    """
    month = datetime.utcnow().month
    warming = await tracker.get_warming_rate(station, month)

    # Default warming rates if no data (conservative estimates)
    if warming:
        p95_rate = float(warming["p95_warming_rate_c_per_hr"] or 2.0)
    else:
        p95_rate = 2.0  # °C/hr — conservative default

    # Convert if needed
    if unit == "F":
        p95_rate *= 1.8  # °C/hr → °F/hr

    hours_remaining = max(0, 24 - current_local_hour)

    # Get day high
    if not metar_data.velocity:
        return {}

    day_high = metar_data.velocity.day_high
    current_temp = metar_data.temp_c if unit == "C" else metar_data.temp_f

    decay: Dict[str, float] = {}

    for mbin in bins:
        bin_info = mbin if hasattr(mbin, "low") else mbin.get("bin", mbin)
        if hasattr(bin_info, "low"):
            bin_low = bin_info.low
            label = bin_info.label
        else:
            bin_low = bin_info.get("low", 0)
            label = bin_info.get("label", "")

        if bin_low <= day_high:
            # Bin is at or below current high — not a tail
            continue

        gap = bin_low - day_high  # how much warming needed above established high
        max_possible_rise = p95_rate * hours_remaining

        if max_possible_rise <= 0 or gap > max_possible_rise:
            p_bin = 0.0  # physically impossible
        else:
            ratio = gap / max_possible_rise
            # Velocity penalty: if currently falling, harder to reverse
            velocity_penalty = 0.0
            if metar_data.velocity and metar_data.velocity.velocity < 0:
                velocity_penalty = abs(metar_data.velocity.velocity) / 2.0

            p_bin = max(0, (1.0 - ratio) ** 2 * math.exp(-velocity_penalty))

        decay[label] = min(1.0, max(0.0, round(p_bin, 5)))

    return decay


# ---------------------------------------------------------------------------
# Populate historical stats (bootstrap on first run)
# ---------------------------------------------------------------------------
async def populate_historical_stats(station: str, timezone_str: str):
    """
    Fetch recent METAR history and calculate:
    1. time_of_high_stats: what hour the daily high typically occurs
    2. warming_rates: max and percentile warming rates
    """
    try:
        url = f"{AVIATION_WX_URL}?ids={station}&format=json&hours=168"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"User-Agent": "WeatherEdgeBot/1.0"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.error("Historical METAR fetch failed: HTTP %d", resp.status)
                    return
                data = await resp.json()
    except Exception as e:
        logger.error("Historical METAR fetch error: %s", e)
        return

    if not isinstance(data, list):
        return

    tz = pytz.timezone(timezone_str)

    # Organise readings by date (local)
    by_date: Dict[str, list] = {}
    for item in data:
        obs_time_str = item.get("obsTime", "")
        temp = item.get("temp")
        if temp is None or not obs_time_str:
            continue
        try:
            obs_utc = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
            obs_local = obs_utc.astimezone(tz)
            day_key = obs_local.strftime("%Y-%m-%d")
            by_date.setdefault(day_key, []).append({
                "hour": obs_local.hour,
                "temp_c": float(temp),
                "time": obs_local,
            })
        except Exception:
            continue

    if not by_date:
        return

    month = datetime.utcnow().month

    # --- Time of high stats ---
    high_hours: Dict[int, int] = {}
    total_days = 0

    for day_key, readings in by_date.items():
        if len(readings) < 4:
            continue  # need enough readings
        total_days += 1
        max_temp = max(r["temp_c"] for r in readings)
        for r in readings:
            if r["temp_c"] == max_temp:
                h = r["hour"]
                high_hours[h] = high_hours.get(h, 0) + 1
                break

    if total_days > 0:
        for hour in range(24):
            freq = high_hours.get(hour, 0) / total_days
            await tracker.upsert_time_of_high(
                station, month, hour, freq, total_days,
            )

    # --- Warming rates ---
    hourly_changes: list = []
    for day_key, readings in by_date.items():
        readings.sort(key=lambda r: r["time"])
        for i in range(1, len(readings)):
            dt_hours = (readings[i]["time"] - readings[i - 1]["time"]).total_seconds() / 3600.0
            if dt_hours <= 0 or dt_hours > 2:
                continue  # skip gaps
            temp_change = readings[i]["temp_c"] - readings[i - 1]["temp_c"]
            rate = temp_change / dt_hours  # °C/hr
            if rate > 0:  # only warming rates
                hourly_changes.append(rate)

    if hourly_changes:
        hourly_changes.sort()
        max_rate = max(hourly_changes)
        p95_idx = int(len(hourly_changes) * 0.95)
        p75_idx = int(len(hourly_changes) * 0.75)
        p95_rate = hourly_changes[min(p95_idx, len(hourly_changes) - 1)]
        p75_rate = hourly_changes[min(p75_idx, len(hourly_changes) - 1)]

        await tracker.upsert_warming_rate(
            station, month, max_rate, p95_rate, p75_rate, total_days,
        )

    logger.info(
        "Historical stats populated for %s: %d days, %d warming samples",
        station, total_days, len(hourly_changes),
    )


# ---------------------------------------------------------------------------
# Calculate all probabilities for a scan cycle
# ---------------------------------------------------------------------------
async def calculate_all(metar_data: dict, model_data: dict,
                         market_data: dict, stations_cfg: dict) -> dict:
    """
    Run all probability calculations for all stations.
    Returns: {station: {high_set_prob, bin_decay: {label: prob}, ...}}
    """
    results = {}

    for icao, cfg in stations_cfg.items():
        metar = metar_data.get(icao)
        if not metar:
            continue

        tz = pytz.timezone(cfg["timezone"])
        now_local = datetime.now(tz)
        local_hour = now_local.hour
        is_coastal = cfg.get("is_coastal", False)
        unit = cfg.get("unit", "C")

        # High Already Set
        high_set_prob = await calculate_high_already_set(
            icao, metar, local_hour, is_coastal,
        )

        # Bin Decay — find bins from market data for this station
        station_bins = []
        for key, group in market_data.items():
            if isinstance(group, dict):
                if group.get("station") == icao:
                    station_bins.extend(group.get("bins", []))
            elif hasattr(group, "station") and group.station == icao:
                station_bins.extend(group.bins)

        bin_decay = await calculate_bin_decay(
            icao, metar, station_bins, local_hour, unit,
        )

        results[icao] = {
            "high_set_prob": high_set_prob,
            "bin_decay": bin_decay,
            "local_hour": local_hour,
            "is_coastal": is_coastal,
        }

    return results
