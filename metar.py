"""
metar.py — METAR Fetch, Parse, Velocity, T-Group

Fetches METAR observations from aviationweather.gov for all active stations.
Parses temperature, wind, dewpoint. Calculates velocity and acceleration
from recent readings. Extracts T-group for rounding edge detection.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pytz

import tracker

logger = logging.getLogger("metar")

AVIATION_WX_URL = "https://aviationweather.gov/api/data/metar"
USER_AGENT = "WeatherEdgeBot/1.0"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class RoundingEdge:
    exact_bin: str
    is_near_boundary: bool
    confidence_boost: int  # 5 or 10


@dataclass
class VelocityData:
    velocity: float          # °C/hr (or °F/hr depending on unit)
    acceleration: float      # change in velocity
    trend: str               # 'rising', 'falling', 'flat'
    consecutive_falling: int
    day_high: float          # max temp recorded today
    day_high_f: float
    high_time: Optional[datetime] = None
    hours_since_high: float = 0.0


@dataclass
class StationMETAR:
    icao: str
    observed_at: datetime
    temp_c: float
    temp_f: float
    dewpoint_c: float
    wind_dir: int
    wind_speed_kt: int
    wind_gust_kt: Optional[int]
    visibility_m: Optional[float]
    t_group_temp_c: Optional[float]
    raw: str
    velocity: Optional[VelocityData] = None
    rounding_edge: Optional[RoundingEdge] = None


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
async def fetch_all_stations(station_ids: List[str]) -> Dict[str, StationMETAR]:
    """
    Fetch latest METAR for all stations in one API call.
    aviationweather.gov supports comma-separated IDs.
    """
    ids_str = ",".join(station_ids)
    url = f"{AVIATION_WX_URL}?ids={ids_str}&format=json"
    result: Dict[str, StationMETAR] = {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.error("METAR fetch failed: HTTP %d", resp.status)
                    return result
                data = await resp.json()
    except Exception as e:
        logger.error("METAR fetch error: %s", e)
        return result

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    for item in data:
        try:
            parsed = parse_metar(item)
            if parsed:
                result[parsed.icao] = parsed
                # Store in database
                await tracker.store_metar(
                    station=parsed.icao,
                    observed_at=parsed.observed_at,
                    temp_c=parsed.temp_c,
                    temp_f=parsed.temp_f,
                    dewpoint_c=parsed.dewpoint_c,
                    wind_dir=parsed.wind_dir,
                    wind_speed_kt=parsed.wind_speed_kt,
                    wind_gust_kt=parsed.wind_gust_kt,
                    visibility_m=parsed.visibility_m,
                    t_group_temp_c=parsed.t_group_temp_c,
                    raw_metar=parsed.raw,
                )
        except Exception as e:
            logger.error("METAR parse error for %s: %s", item.get("icaoId", "?"), e)

    return result


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
def parse_metar(raw: dict) -> Optional[StationMETAR]:
    """Parse aviationweather.gov JSON METAR into StationMETAR."""
    icao = raw.get("icaoId")
    if not icao:
        return None

    obs_time_raw = raw.get("obsTime")
    if obs_time_raw:
        if isinstance(obs_time_raw, (int, float)):
            # Unix timestamp (seconds since epoch)
            observed_at = datetime.utcfromtimestamp(obs_time_raw)
        elif isinstance(obs_time_raw, str):
            observed_at = datetime.fromisoformat(obs_time_raw.replace("Z", "+00:00"))
            observed_at = observed_at.replace(tzinfo=None)
        else:
            observed_at = datetime.utcnow()
    else:
        observed_at = datetime.utcnow()

    temp_c = float(raw.get("temp", 0))
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    dewpoint_c = float(raw.get("dewp", 0))

    wind_dir = int(raw.get("wdir", 0)) if raw.get("wdir") not in (None, "VRB") else 0
    wind_speed = int(raw.get("wspd", 0))
    wind_gust = int(raw.get("wgst")) if raw.get("wgst") else None

    visib_raw = raw.get("visib")
    visibility_m = None
    if visib_raw:
        try:
            v = str(visib_raw).replace("+", "")
            visibility_m = float(v) * 1609.34  # statute miles → meters
        except (ValueError, TypeError):
            pass

    raw_ob = raw.get("rawOb", "")
    t_group_temp_c = extract_t_group(raw_ob)

    return StationMETAR(
        icao=icao,
        observed_at=observed_at,
        temp_c=temp_c,
        temp_f=temp_f,
        dewpoint_c=dewpoint_c,
        wind_dir=wind_dir,
        wind_speed_kt=wind_speed,
        wind_gust_kt=wind_gust,
        visibility_m=visibility_m,
        t_group_temp_c=t_group_temp_c,
        raw=raw_ob,
    )


# ---------------------------------------------------------------------------
# T-Group extraction
# ---------------------------------------------------------------------------
def extract_t_group(raw_metar: str) -> Optional[float]:
    """
    Extract T-group from METAR remarks for tenths-precision temperature.
    Format: T{sign}{temp3digits}{sign}{dewp3digits}
    Example: T00940022 → temp = +09.4°C
    Sign digit: 0 = positive, 1 = negative
    """
    match = re.search(r"\bT(\d)(\d{3})(\d)(\d{3})\b", raw_metar)
    if not match:
        return None

    temp_sign = -1 if match.group(1) == "1" else 1
    temp_raw = int(match.group(2))
    temp_c = temp_sign * temp_raw / 10.0
    return temp_c


# ---------------------------------------------------------------------------
# Velocity & acceleration
# ---------------------------------------------------------------------------
async def calculate_velocity(station: str, station_tz: str,
                              unit: str = "C") -> Optional[VelocityData]:
    """
    Pull today's METAR readings and compute velocity, acceleration, trend.
    Uses numpy linear regression on (hours_elapsed, temperature) pairs.
    """
    readings = await tracker.get_today_metar(station, station_tz)
    if len(readings) < 2:
        return None

    # Build (time_hours, temp) arrays
    temps_c = [float(r["temp_c"]) for r in readings if r["temp_c"] is not None]
    temps_f = [float(r["temp_f"]) for r in readings if r["temp_f"] is not None]
    times = [r["observed_at"] for r in readings if r["temp_c"] is not None]

    if len(temps_c) < 2:
        return None

    t0 = times[0]
    hours = [(t - t0).total_seconds() / 3600.0 for t in times]

    # Use the station's preferred unit for velocity
    temps = temps_c if unit == "C" else temps_f

    # Linear regression for velocity
    x = np.array(hours)
    y = np.array(temps)
    if len(x) < 2 or x[-1] - x[0] < 0.1:
        return None

    coeffs = np.polyfit(x, y, 1)
    velocity = float(coeffs[0])  # slope = °/hr

    # Acceleration: compare slope of first half vs second half
    mid = len(x) // 2
    if mid >= 2 and len(x) - mid >= 2:
        slope1 = float(np.polyfit(x[:mid], y[:mid], 1)[0])
        slope2 = float(np.polyfit(x[mid:], y[mid:], 1)[0])
        acceleration = slope2 - slope1
    else:
        acceleration = 0.0

    # Trend
    if velocity > 0.2:
        trend = "rising"
    elif velocity < -0.2:
        trend = "falling"
    else:
        trend = "flat"

    # Consecutive falling: count from most recent backwards
    consecutive_falling = 0
    for i in range(len(temps) - 1, 0, -1):
        if temps[i] < temps[i - 1]:
            consecutive_falling += 1
        else:
            break

    # Day high
    day_high_c = max(temps_c)
    day_high_f = max(temps_f) if temps_f else day_high_c * 9 / 5 + 32
    high_idx = temps_c.index(max(temps_c))
    high_time = times[high_idx]
    now_utc = datetime.utcnow()
    hours_since_high = (now_utc - high_time).total_seconds() / 3600.0

    return VelocityData(
        velocity=round(velocity, 2),
        acceleration=round(acceleration, 3),
        trend=trend,
        consecutive_falling=consecutive_falling,
        day_high=day_high_c if unit == "C" else day_high_f,
        day_high_f=day_high_f,
        high_time=high_time,
        hours_since_high=round(hours_since_high, 2),
    )


# ---------------------------------------------------------------------------
# Wind shift detection (coastal secondary peak)
# ---------------------------------------------------------------------------
def detect_wind_shift(readings: List[StationMETAR]) -> bool:
    """
    Check if wind direction changed by >45° in the last 2 readings.
    A wind shift at coastal stations can signal sea breeze onset/reversal.
    """
    if len(readings) < 2:
        return False

    dir1 = readings[-2].wind_dir
    dir2 = readings[-1].wind_dir
    if dir1 == 0 or dir2 == 0:
        return False

    diff = abs(dir2 - dir1)
    if diff > 180:
        diff = 360 - diff
    return diff > 45


# ---------------------------------------------------------------------------
# Rounding edge extraction
# ---------------------------------------------------------------------------
def extract_rounding_edge(t_group_temp_c: float, unit: str) -> Optional[RoundingEdge]:
    """
    When we have T-group tenths precision, determine the exact bin.
    Check if temp is within 0.3° of a bin boundary → high confidence.
    """
    if unit == "F":
        temp_f = t_group_temp_c * 9.0 / 5.0 + 32.0
        rounded = round(temp_f)
        bin_low = rounded if rounded % 2 == 0 else rounded - 1
        bin_high = bin_low + 1
        exact_bin = f"{bin_low}-{bin_high}°F"
        # Check proximity to boundary
        dist_to_lower = abs(temp_f - bin_low)
        dist_to_upper = abs(temp_f - (bin_high + 1))
        is_near = min(dist_to_lower, dist_to_upper) < 0.3
    else:
        rounded = round(t_group_temp_c)
        bin_low = rounded
        bin_high = rounded + 1
        exact_bin = f"{bin_low}-{bin_high}°C"
        dist_to_lower = abs(t_group_temp_c - bin_low)
        dist_to_upper = abs(t_group_temp_c - bin_high)
        is_near = min(dist_to_lower, dist_to_upper) < 0.3

    confidence_boost = 10 if not is_near else 5  # definitively in bin vs near boundary

    return RoundingEdge(
        exact_bin=exact_bin,
        is_near_boundary=is_near,
        confidence_boost=confidence_boost,
    )


# ---------------------------------------------------------------------------
# Enrich METAR with velocity + rounding edge
# ---------------------------------------------------------------------------
async def enrich_metar(station_metar: StationMETAR, station_tz: str,
                        unit: str) -> StationMETAR:
    """Add velocity data and rounding edge to a parsed METAR."""
    velocity = await calculate_velocity(station_metar.icao, station_tz, unit)
    station_metar.velocity = velocity

    if station_metar.t_group_temp_c is not None:
        station_metar.rounding_edge = extract_rounding_edge(
            station_metar.t_group_temp_c, unit
        )

    return station_metar
