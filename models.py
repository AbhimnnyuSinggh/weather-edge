"""
models.py — Model Forecasts, Bias Correction, Dynamic Weighting

Fetches weather model forecasts from multiple sources:
- Open-Meteo (GFS, ECMWF, ICON) — free, no key
- NWS (US stations only) — free, no key
- Tomorrow.io — 500 free calls/day
- OpenWeather — 1000 free calls/day
- Weatherbit — 50 free calls/day
- NOAA MOS — free, no key, US only
- Open-Meteo Ensemble (31 GFS members) — free, no key
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import ssl

import aiohttp
import asyncio
import pytz

import tracker

logger = logging.getLogger("models")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
NWS_POINTS_URL = "https://api.weather.gov/points"
NOAA_MOS_URL = "https://aviationweather.gov/cgi-bin/data/mos.php"
VISUAL_CROSSING_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
VISUAL_CROSSING_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
USER_AGENT = "WeatherEdgeBot/1.0"
EWMA_ALPHA = 0.15  # ~90% weight to last 13 data points

# Default MAE per model (°C) — used for distribution calculations
DEFAULT_MAE = {
    "gfs": 1.8, "ecmwf": 1.5, "icon": 2.0, "gem": 2.2, "jma": 2.0,
    "nws": 1.5, "noaa_mos": 1.3,
    "visual_crossing": 2.0, "ensemble": 1.0,
}

# Rate limit tracking (resets per scan cycle)
_rate_limit_status: Dict[str, str] = {}  # source -> "ok" | "limited" | "no_key"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ModelForecast:
    station: str
    model_name: str
    target_date: date
    raw_high_c: float
    raw_high_f: float
    bias_corrected_c: float
    bias_corrected_f: float
    weight: float
    ensemble_members: List[float] = field(default_factory=list)
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.utcnow()


# ---------------------------------------------------------------------------
# Fetch all stations
# ---------------------------------------------------------------------------
def get_rate_limit_status() -> Dict[str, str]:
    """Return current rate limit status for all sources."""
    return dict(_rate_limit_status)


def reset_rate_limits():
    """Reset rate limits at the start of each scan cycle."""
    global _rate_limit_status
    _rate_limit_status = {}


async def fetch_all_stations(stations_cfg: dict) -> Dict[str, Dict[str, ModelForecast]]:
    """
    For each active station, fetch forecasts from ALL available sources.
    Returns nested dict: {station: {model_name: ModelForecast}}
    Sources used/skipped are logged at the end.
    """
    global _rate_limit_status
    reset_rate_limits()
    results: Dict[str, Dict[str, ModelForecast]] = {}

    for icao, cfg in stations_cfg.items():
        results[icao] = {}
        model_list = cfg.get("models", ["gfs", "ecmwf", "icon", "gem", "jma"])
        lat, lon = cfg["lat"], cfg["lon"]

        # 1) Open-Meteo deterministic (GFS, ECMWF, ICON, GEM, JMA)
        open_meteo_models = [m for m in model_list if m in ("gfs", "ecmwf", "icon", "gem", "jma")]
        if open_meteo_models:
            om_results = await _fetch_open_meteo(icao, lat, lon, open_meteo_models, cfg)
            results[icao].update(om_results)

        # 2) NWS (US stations only)
        if cfg.get("country") == "US":
            nws_result = await _fetch_nws(icao, lat, lon, cfg)
            if nws_result:
                results[icao]["nws"] = nws_result

        # 3) Visual Crossing
        if _rate_limit_status.get("visual_crossing") != "limited":
            vc_result = await _fetch_visual_crossing(icao, lat, lon, cfg)
            if vc_result:
                results[icao]["visual_crossing"] = vc_result

        # 4) NOAA MOS (US stations only)
        if cfg.get("country") == "US":
            mos_result = await _fetch_noaa_mos(icao, cfg)
            if mos_result:
                results[icao]["noaa_mos"] = mos_result

        # 5) Open-Meteo Ensemble
        ensemble_result = await _fetch_open_meteo_ensemble(icao, lat, lon, cfg)
        if ensemble_result:
            results[icao]["ensemble"] = ensemble_result

    # Log source summary
    sources_used = set()
    for icao_data in results.values():
        sources_used.update(icao_data.keys())
    sources_limited = {k: v for k, v in _rate_limit_status.items() if v != "ok"}
    logger.info("Forecasts collected: %s | Limited: %s",
                list(sources_used), sources_limited if sources_limited else "none")

    return results


# ---------------------------------------------------------------------------
# Open-Meteo fetch
# ---------------------------------------------------------------------------
async def _fetch_open_meteo(station: str, lat: float, lon: float,
                             model_list: List[str],
                             station_cfg: dict) -> Dict[str, ModelForecast]:
    """Fetch forecasts from Open-Meteo for multiple models in one call."""
    model_map = {
        "gfs": "gfs_seamless",
        "ecmwf": "ecmwf_ifs025",
        "icon": "icon_seamless",
        "gem": "gem_seamless",
        "jma": "jma_seamless",
    }
    api_models = ",".join(model_map[m] for m in model_list if m in model_map)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "models": api_models,
        "timezone": "auto",
        "forecast_days": 2,
    }

    api_key = os.environ.get("OPEN_METEO_API_KEY")
    fetch_url = "https://customer-api.open-meteo.com/v1/forecast" if api_key else OPEN_METEO_URL
    if api_key:
        params["apikey"] = api_key

    results: Dict[str, ModelForecast] = {}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.get(
                    fetch_url, params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 429:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning("Open-Meteo 429 for %s. Retrying in %ds...", station, wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning("Open-Meteo HTTP 429: Rate limit exhausted for %s", station)
                            return results
                    if resp.status in (400, 401, 403) and "customer" in fetch_url:
                        logger.warning("Open-Meteo API key unauthorized for Customer-API. Falling back to public endpoint.")
                        fallback_url = fetch_url.replace("customer-api.open-meteo.com", "api.open-meteo.com")
                        if "apikey" in params:
                            del params["apikey"]
                        async with session.get(fallback_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as fallback_resp:
                            if fallback_resp.status != 200:
                                logger.error("Open-Meteo fallback HTTP %d for %s", fallback_resp.status, station)
                                return results
                            data = await fallback_resp.json()
                            break # Break the retry loop as fallback was attempted
                    if resp.status != 200:
                        logger.error("Open-Meteo HTTP %d for %s", resp.status, station)
                        return results
                    data = await resp.json()
                    break # Break the retry loop on success
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("Open-Meteo error for %s (attempt %d): %s, retrying...", station, attempt+1, e)
                await asyncio.sleep(1)
            else:
                logger.error("Open-Meteo final error for %s: %s", station, e)
                return results

    daily = data.get("daily", {})
    if not daily:
        logger.error("Open-Meteo returned no daily data for %s", station)
        return results

    tz_name = station_cfg.get("tz", "UTC")
    today_local = datetime.now(pytz.timezone(tz_name)).date()
    tomorrow_local = today_local + timedelta(days=1)

    for model_key, api_name in model_map.items():
        if model_key not in model_list:
            continue

        model_specific_key = f"temperature_2m_max_{api_name}"
        if model_specific_key in daily:
            model_daily = daily[model_specific_key]
        elif "temperature_2m_max" in daily:
            model_daily = daily["temperature_2m_max"]
        else:
            model_daily = None

        if not model_daily or not daily.get("time"):
            logger.debug("No data for model %s (%s) at station %s", model_key, api_name, station)
            continue

        dates = daily["time"]
        for i, date_str in enumerate(dates):
            if i >= len(model_daily) or model_daily[i] is None:
                continue

            target_date = date.fromisoformat(date_str)
            raw_high_c = float(model_daily[i])
            raw_high_f = raw_high_c * 9.0 / 5.0 + 32.0

            # Apply bias correction
            bias_c = await _get_bias(station, model_key, station_cfg)
            corrected_c = raw_high_c - bias_c
            corrected_f = corrected_c * 9.0 / 5.0 + 32.0

            # Get dynamic weight
            weight = await _get_weight(station, model_key)

            forecast = ModelForecast(
                station=station,
                model_name=model_key,
                target_date=target_date,
                raw_high_c=raw_high_c,
                raw_high_f=raw_high_f,
                bias_corrected_c=corrected_c,
                bias_corrected_f=corrected_f,
                weight=weight,
            )
            
            # Prioritize today's forecast, but allow tomorrow's if today's is missing (useful for Asian markets at night UTC)
            if station_cfg.get("target_date"): # Check if the function was called with a specific target_date
                if target_date == station_cfg["target_date"]: # Only append if the current forecast date matches the requested date
                    results[model_key] = forecast
            else: # Fallback to old logic if no specific target_date was requested
                if model_key not in results or target_date in (today_local, tomorrow_local):
                    # If we already have a forecast and it's for today, don't overwrite it with tomorrow
                    if model_key in results and results[model_key].target_date == today_local and target_date == tomorrow_local:
                        pass
                    else:
                        results[model_key] = forecast

            # Store in database (protected against missing stations by try/except inside store_forecast)
            await tracker.store_forecast(
                station, target_date, model_key,
                raw_high_c, raw_high_f, corrected_c, corrected_f,
            )

    return results


# ---------------------------------------------------------------------------
# NWS fetch (US stations only)
# ---------------------------------------------------------------------------
async def _fetch_nws(station: str, lat: float, lon: float,
                      station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch NWS point forecast for US stations."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            # Step 1: Get forecast URL from points endpoint
            points_url = f"{NWS_POINTS_URL}/{lat},{lon}"
            async with session.get(
                points_url,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("NWS points HTTP %d for %s", resp.status, station)
                    return None
                points_data = await resp.json()

            forecast_url = points_data.get("properties", {}).get("forecast")
            if not forecast_url:
                return None

            # Step 2: Get the actual forecast
            async with session.get(
                forecast_url,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("NWS forecast HTTP %d for %s", resp.status, station)
                    return None
                forecast_data = await resp.json()

    except Exception as e:
        logger.error("NWS error for %s: %s", station, e)
        return None

    # Parse periods — find daytime period with high temperature
    periods = forecast_data.get("properties", {}).get("periods", [])
    for period in periods:
        if period.get("isDaytime"):
            temp_f = float(period.get("temperature", 0))
            temp_unit = period.get("temperatureUnit", "F")
            if temp_unit == "C":
                temp_c = temp_f
                temp_f = temp_c * 9.0 / 5.0 + 32.0
            else:
                temp_c = (temp_f - 32.0) * 5.0 / 9.0

            # Parse date from period
            start_time = period.get("startTime", "")
            try:
                target_date = date.fromisoformat(start_time[:10])
            except (ValueError, IndexError):
                tz_name = station_cfg.get("tz", "UTC")
                target_date = datetime.now(pytz.timezone(tz_name)).date()

            bias_c = await _get_bias(station, "nws", station_cfg)
            corrected_c = temp_c - bias_c
            corrected_f = corrected_c * 9.0 / 5.0 + 32.0
            weight = await _get_weight(station, "nws")

            forecast = ModelForecast(
                station=station,
                model_name="nws",
                target_date=target_date,
                raw_high_c=temp_c,
                raw_high_f=temp_f,
                bias_corrected_c=corrected_c,
                bias_corrected_f=corrected_f,
                weight=weight,
            )

            await tracker.store_forecast(
                station, target_date, "nws",
                temp_c, temp_f, corrected_c, corrected_f,
            )
            return forecast

    return None


    return None


# ---------------------------------------------------------------------------
# NOAA Gridpoints fetch (US stations only)
# ---------------------------------------------------------------------------
async def _fetch_noaa(station: str, lat: float, lon: float,
                       station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch NOAA gridpoints forecast for US stations."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            # Step 1: Get gridpoint endpoints
            points_url = f"{NWS_POINTS_URL}/{lat},{lon}"
            async with session.get(
                points_url,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("NOAA points HTTP %d for %s", resp.status, station)
                    return None
                points_data = await resp.json()

            grid_id = points_data.get("properties", {}).get("gridId")
            grid_x = points_data.get("properties", {}).get("gridX")
            grid_y = points_data.get("properties", {}).get("gridY")
            
            if not grid_id or grid_x is None or grid_y is None:
                return None

            forecast_url = f"https://api.weather.gov/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast"

            # Step 2: Get the gridpoint forecast
            async with session.get(
                forecast_url,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("NOAA gridpoints HTTP %d for %s", resp.status, station)
                    return None
                forecast_data = await resp.json()

    except Exception as e:
        logger.error("NOAA error for %s: %s", station, e)
        return None

    # Parse periods — find daytime period with high temperature
    periods = forecast_data.get("properties", {}).get("periods", [])
    for period in periods:
        if period.get("isDaytime"):
            temp_f = float(period.get("temperature", 0))
            temp_unit = period.get("temperatureUnit", "F")
            if temp_unit == "C":
                temp_c = temp_f
                temp_f = temp_c * 9.0 / 5.0 + 32.0
            else:
                temp_c = (temp_f - 32.0) * 5.0 / 9.0

            start_time = period.get("startTime", "")
            try:
                target_date = date.fromisoformat(start_time[:10])
            except (ValueError, IndexError):
                tz_name = station_cfg.get("tz", "UTC")
                target_date = datetime.now(pytz.timezone(tz_name)).date()

            bias_c = await _get_bias(station, "noaa", station_cfg)
            corrected_c = temp_c - bias_c
            corrected_f = corrected_c * 9.0 / 5.0 + 32.0
            weight = await _get_weight(station, "noaa")

            forecast = ModelForecast(
                station=station,
                model_name="noaa",
                target_date=target_date,
                raw_high_c=temp_c,
                raw_high_f=temp_f,
                bias_corrected_c=corrected_c,
                bias_corrected_f=corrected_f,
                weight=weight,
            )

            await tracker.store_forecast(
                station, target_date, "noaa",
                temp_c, temp_f, corrected_c, corrected_f,
            )
            return forecast

    return None




# ---------------------------------------------------------------------------
# Visual Crossing fetch (requires key)
# ---------------------------------------------------------------------------
async def _fetch_visual_crossing(station: str, lat: float, lon: float,
                                 station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch daily high from Visual Crossing API."""
    import os
    api_key = os.environ.get("VISUAL_CROSSING_API_KEY")
    if not api_key:
        return None

    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today"
        params = {
            "unitGroup": "us", # always fetch F
            "include": "days",
            "key": api_key,
            "contentType": "json"
        }
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(
                url, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.debug("Visual Crossing HTTP %d for %s", resp.status, station)
                    return None
                data = await resp.json()

        days = data.get("days", [])
        if not days:
            return None
            
        raw_high_f = float(days[0].get("tempmax"))
        raw_high_c = (raw_high_f - 32.0) * 5.0 / 9.0

        tz_name = station_cfg.get("tz", "UTC")
        target_date = datetime.now(pytz.timezone(tz_name)).date()

        bias_c = await _get_bias(station, "visual_crossing", station_cfg)
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0
        weight = await _get_weight(station, "visual_crossing")

        forecast = ModelForecast(
            station=station, model_name="visual_crossing",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
        )
        await tracker.store_forecast(
            station, target_date, "visual_crossing",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )
        return forecast

    except Exception as e:
        logger.error("Visual Crossing error for %s: %s", station, e)
        return None


# ---------------------------------------------------------------------------
# NOAA MOS fetch (free, no key, US stations only)
# ---------------------------------------------------------------------------
async def _fetch_noaa_mos(station: str,
                           station_cfg: dict) -> Optional[ModelForecast]:
    """
    Parse NOAA MOS (Model Output Statistics) text for MAX temperature.
    MOS is statistically post-processed GFS — often more accurate than raw GFS.
    Only available for US stations.
    """
    try:
        url = "https://mesonet.agron.iastate.edu/mos/csv.php"
        params = {"station": station, "model": "GFS"}
        
        # IEM SSL cert chain is sometimes missing locally on MacOS, circumventing quietly.
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(
                url, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.debug("IEM MOS HTTP %d for %s", resp.status, station)
                    return None
                text = await resp.text()

        import csv
        lines = text.splitlines()
        reader = csv.DictReader(lines)
        
        tz_name = station_cfg.get("tz", "UTC")
        now_local = datetime.now(pytz.timezone(tz_name))
        
        # Look for the max temperature for the target date from hourly `tmp` metrics
        target_date_str = now_local.strftime("%Y-%m-%d")
        
        raw_high_f = -999.0
        found = False
        for row in list(reader)[:200]:
            if row.get('ftime', '').startswith(target_date_str):
                tmp_str = row.get('tmp')
                if tmp_str:
                    try:
                        t = float(tmp_str)
                        if t > raw_high_f:
                            raw_high_f = t
                            found = True
                    except ValueError:
                        pass
        
        if not found or raw_high_f == -999.0:
            logger.debug("IEM MOS: no temperature points found for %s today", station)
            return None

        raw_high_c = (raw_high_f - 32.0) * 5.0 / 9.0
        target_date = datetime.now(pytz.timezone(tz_name)).date()

        bias_c = await _get_bias(station, "noaa_mos", station_cfg)
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0
        weight = await _get_weight(station, "noaa_mos")

        forecast = ModelForecast(
            station=station, model_name="noaa_mos",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
        )
        await tracker.store_forecast(
            station, target_date, "noaa_mos",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )
        return forecast

    except Exception as e:
        logger.error("IEM MOS error for %s: %s", station, e)
        return None


# ---------------------------------------------------------------------------
# Open-Meteo Ensemble fetch (Free, 31 members)
# ---------------------------------------------------------------------------
async def _fetch_open_meteo_ensemble(station: str, lat: float, lon: float,
                                      station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch 31-member GFS ensemble from Open-Meteo."""
    try:
        api_key = os.environ.get("OPEN_METEO_API_KEY")
        fetch_url = "https://customer-ensemble-api.open-meteo.com/v1/ensemble" if api_key else OPEN_METEO_ENSEMBLE_URL

        params = {
            "latitude": lat,
            "longitude": lon,
            "models": "gfs_seamless",
            "daily": "temperature_2m_max",
            "timezone": "auto"
        }
        if api_key:
            params["apikey"] = api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(
                fetch_url, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

        daily = data.get("daily", {})
        times = daily.get("time", [])
        if not times:
            return None

        target_idx = 1 if len(times) > 1 else 0
        target_date_str = times[target_idx][:10]
        try:
            target_date = date.fromisoformat(target_date_str)
        except ValueError:
            tz_name = station_cfg.get("tz", "UTC")
            target_date = datetime.now(pytz.timezone(tz_name)).date()

        tz_name = station_cfg.get("tz", "UTC")
        today_local = datetime.now(pytz.timezone(tz_name)).date()
        tomorrow_local = today_local + timedelta(days=1)

        if target_date not in (today_local, tomorrow_local):
            return None

        members_c = []
        for i in range(1, 32):  # Members 00..31 in some models, but open-meteo usually uses member01..member31
            key = f"temperature_2m_max_member{i:02d}"
            arr = daily.get(key)
            if arr and len(arr) > target_idx and arr[target_idx] is not None:
                members_c.append(float(arr[target_idx]))

        if len(members_c) < 10:  # Not a real ensemble if too few members
            return None

        bias_c = await _get_bias(station, "ensemble", station_cfg)
        members_corrected_c = [m - bias_c for m in members_c]

        raw_high_c = sum(members_c) / len(members_c)
        raw_high_f = raw_high_c * 9.0 / 5.0 + 32.0
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0

        weight = await _get_weight(station, "ensemble")

        forecast = ModelForecast(
            station=station, model_name="ensemble",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
            ensemble_members=members_corrected_c
        )
        
        await tracker.store_forecast(
            station, target_date, "ensemble",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )

        return forecast

    except Exception as e:
        logger.error("Open-Meteo Ensemble error for %s: %s", station, e)
        return None


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------
async def _get_bias(station: str, model_name: str, station_cfg: dict) -> float:
    """
    Get current bias for this model/station.
    Uses learned EWMA if sample_count >= 5, else starting_bias from config.
    """
    acc = await tracker.get_model_accuracy(station, model_name)
    if acc and acc["sample_count"] >= 5:
        return float(acc["ewma_bias"])
    return station_cfg.get("starting_bias", {}).get(model_name, 0.0)


async def _get_weight(station: str, model_name: str) -> float:
    """Get current dynamic weight for this model at this station."""
    acc = await tracker.get_model_accuracy(station, model_name)
    if acc and acc["weight"]:
        return float(acc["weight"])
    return 0.25  # equal weight default


# ---------------------------------------------------------------------------
# Dynamic weights
# ---------------------------------------------------------------------------
async def get_dynamic_weights(station: str, model_list: List[str]) -> Dict[str, float]:
    """
    Retrieve accuracy scores and compute weights using inverse MAE.
    weight_i = (1/mae_i) / sum(1/mae_j for all j)
    If any model has < 10 data points, use equal weights.
    """
    accuracies = {}
    for model in model_list:
        acc = await tracker.get_model_accuracy(station, model)
        if acc:
            accuracies[model] = acc
        else:
            accuracies[model] = None

    # Check if all models have enough data
    min_samples = min(
        (a["sample_count"] if a else 0) for a in accuracies.values()
    )

    if min_samples < 10:
        # Equal weights
        w = 1.0 / len(model_list) if model_list else 0.25
        return {m: w for m in model_list}

    # Inverse MAE weighting
    inv_maes = {}
    for model, acc in accuracies.items():
        mae = max(0.1, float(acc["ewma_error"]))  # floor at 0.1 to avoid inf
        inv_maes[model] = 1.0 / mae

    total_inv = sum(inv_maes.values())
    return {m: v / total_inv for m, v in inv_maes.items()}


# ---------------------------------------------------------------------------
# Weighted consensus
# ---------------------------------------------------------------------------
def weighted_consensus(forecasts: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted average forecast temperature."""
    total_weight = 0.0
    weighted_sum = 0.0
    for model, temp in forecasts.items():
        w = weights.get(model, 0.25)
        weighted_sum += temp * w
        total_weight += w
    if total_weight == 0:
        return sum(forecasts.values()) / max(1, len(forecasts))
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Momentum divergence
# ---------------------------------------------------------------------------
async def check_momentum_divergence(station: str, current_high: float,
                                      station_cfg: dict) -> float:
    """
    Compare what models predicted for TODAY vs what METAR actually shows.
    Returns trust_penalty (0.0 to 0.5):
    0.0 = models were perfect, full trust
    0.5 = models were off by 5°C+, major distrust
    """
    today = date.today()
    forecasts = await tracker.get_latest_forecasts(station, today)
    if not forecasts:
        return 0.0

    # Average forecast for today
    avg_forecast = sum(float(f["bias_corrected_c"]) for f in forecasts) / len(forecasts)
    error = abs(avg_forecast - current_high)
    trust_penalty = min(1.0, error / 5.0) * 0.5
    return round(trust_penalty, 3)


# ---------------------------------------------------------------------------
# Accuracy update on resolution
# ---------------------------------------------------------------------------
async def update_accuracy_on_resolution(station: str, target_date: date,
                                          actual_high_c: float):
    """
    Called when a market resolves.
    Update EWMA error, bias, and recalculate weights for all models.
    """
    forecasts = await tracker.get_latest_forecasts(station, target_date)
    if not forecasts:
        return

    model_errors = {}
    for f in forecasts:
        model_name = f["model_name"]
        predicted = float(f["bias_corrected_c"])
        error = predicted - actual_high_c
        abs_error = abs(error)

        # Get current accuracy record
        acc = await tracker.get_model_accuracy(station, model_name)
        if acc:
            old_ewma_error = float(acc["ewma_error"])
            old_ewma_bias = float(acc["ewma_bias"])
            sample_count = acc["sample_count"] + 1
        else:
            old_ewma_error = abs_error
            old_ewma_bias = error
            sample_count = 1

        # EWMA update
        new_ewma_error = EWMA_ALPHA * abs_error + (1 - EWMA_ALPHA) * old_ewma_error
        new_ewma_bias = EWMA_ALPHA * error + (1 - EWMA_ALPHA) * old_ewma_bias

        model_errors[model_name] = new_ewma_error

        await tracker.upsert_model_accuracy(
            station, model_name,
            ewma_error=new_ewma_error,
            ewma_bias=new_ewma_bias,
            sample_count=sample_count,
            weight=0.25,  # placeholder — recalculated below
        )

    # Recalculate normalized weights using inverse MAE
    if model_errors:
        inv_maes = {m: 1.0 / max(0.1, e) for m, e in model_errors.items()}
        total_inv = sum(inv_maes.values())
        for model_name, inv_mae in inv_maes.items():
            weight = inv_mae / total_inv
            acc = await tracker.get_model_accuracy(station, model_name)
            if acc:
                await tracker.upsert_model_accuracy(
                    station, model_name,
                    ewma_error=float(acc["ewma_error"]),
                    ewma_bias=float(acc["ewma_bias"]),
                    sample_count=acc["sample_count"],
                    weight=weight,
                )

    logger.info("Model accuracy updated for %s/%s: %s", station, target_date, model_errors)


# ---------------------------------------------------------------------------
# Get latest from DB (when fresh fetch not needed)
# ---------------------------------------------------------------------------
async def get_latest_from_db(stations_cfg: dict) -> Dict[str, Dict[str, ModelForecast]]:
    """Retrieve the most recent forecasts from database."""
    results: Dict[str, Dict[str, ModelForecast]] = {}
    today = date.today()

    for icao, cfg in stations_cfg.items():
        results[icao] = {}
        for target in [today, today]:
            rows = await tracker.get_latest_forecasts(icao, target)
            for r in rows:
                weight = await _get_weight(icao, r["model_name"])
                results[icao][r["model_name"]] = ModelForecast(
                    station=icao,
                    model_name=r["model_name"],
                    target_date=r["target_date"],
                    raw_high_c=float(r["raw_high_c"]) if r["raw_high_c"] else 0,
                    raw_high_f=float(r["raw_high_f"]) if r["raw_high_f"] else 0,
                    bias_corrected_c=float(r["bias_corrected_c"]) if r["bias_corrected_c"] else 0,
                    bias_corrected_f=float(r["bias_corrected_f"]) if r["bias_corrected_f"] else 0,
                    weight=weight,
                    fetched_at=r["fetched_at"],
                )
    return results

# ---------------------------------------------------------------------------
# Daily High Prediction (3-step Bayesian Blend)
# ---------------------------------------------------------------------------
async def calculate_daily_high(models_data: Dict[str, ModelForecast], metar: Optional[Any], local_hour: int, station: str, unit: str = "C") -> float:
    """
    3-step Bayesian prediction:
    Step 1: Inverse-MAE weighted average of all models
    Step 2: METAR floor (can't be lower than already observed)
    Step 3: Time-of-day blend (morning = models, afternoon = METAR)
    """

    # Default MAE per model (lower = more accurate = higher weight)
    DEFAULT_MAE = {
        "gfs": 1.8, "ecmwf": 1.5, "icon": 2.0, "gem": 2.2, "jma": 2.0,
        "nws": 1.5, "noaa_mos": 1.3, "visual_crossing": 2.0,
    }

    # Step 1: Inverse-MAE weighted average
    weights = {}
    temps = {}
    for name, forecast in models_data.items():
        temp = forecast.bias_corrected_c if unit == "C" else forecast.bias_corrected_f
        if temp is None or temp == 0:
            continue
        mae = DEFAULT_MAE.get(name, 2.0)
        weights[name] = 1.0 / max(0.5, mae)
        temps[name] = temp

    if not temps:
        return 0.0

    total_w = sum(weights.values())
    model_pred = sum(temps[n] * (weights[n] / total_w) for n in temps)

    # Step 2: METAR floor
    current_high = None
    if metar and hasattr(metar, 'velocity') and metar.velocity:
        current_high = metar.velocity.day_high if unit == "C" else metar.velocity.day_high_f
        model_pred = max(model_pred, current_high)

    # Step 3: Time-of-day blend
    # Before noon local → models dominate (0.0-0.3 blend)
    # After 2PM local → METAR dominates (0.7-1.0 blend)
    if local_hour < 8:
        blend_weight = 0.0
    elif local_hour < 14:
        blend_weight = (local_hour - 8) / 12.0  # 0.0 at 8AM, 0.5 at 14
    else:
        blend_weight = 0.5 + (local_hour - 14) / 12.0  # 0.5 at 14, ~0.83 at 18
    blend_weight = max(0.0, min(1.0, blend_weight))

    if current_high is not None:
        final = (1 - blend_weight) * model_pred + blend_weight * current_high
    else:
        final = model_pred

    return round(final, 1)
