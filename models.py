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
USER_AGENT = "WeatherEdgeBot/1.0"
EWMA_ALPHA = 0.15  # ~90% weight to last 13 data points

# Default MAE per model (°C) — used for distribution calculations
DEFAULT_MAE = {
    # Existing
    "gfs": 1.8, "ecmwf": 1.5, "icon": 2.0, "gem": 2.2, "jma": 2.0,
    "nws": 1.5, "noaa_mos": 1.3,
    "visual_crossing": 2.0, "ensemble": 1.0, "tomorrow": 1.5,
    # New
    "hrrr": 1.2,
    "nbm": 1.1,
    "arpege": 2.0,
    "ukmo": 1.8,
    "bom": 2.2,
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


async def fetch_all_stations(stations_cfg: dict, use_cache_fallback: bool = True) -> Dict[str, Dict[str, ModelForecast]]:
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

        # 1) Open-Meteo deterministic
        open_meteo_models = [m for m in model_list if m in ("gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom")]
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

        # 3.5) Tomorrow.io
        if _rate_limit_status.get("tomorrow") != "limited":
            tomorrow_result = await _fetch_tomorrow_io(icao, lat, lon, cfg)
            if tomorrow_result:
                results[icao]["tomorrow"] = tomorrow_result

        # 4) NOAA MOS (US stations only)
        if cfg.get("country") == "US":
            mos_result = await _fetch_noaa_mos(icao, cfg)
            if mos_result:
                results[icao]["noaa_mos"] = mos_result

        # 5) Open-Meteo Ensemble
        ensemble_result = await _fetch_open_meteo_ensemble(icao, lat, lon, cfg)
        if ensemble_result:
            results[icao]["ensemble"] = ensemble_result

        # --- DATABASE FALLBACK ---
        # If any live API failed (e.g. Rate Limit 429), recover the last known forecast from the DB
        # Bypassed completely if use_cache_fallback is False (e.g. live dashboard commands)
        if use_cache_fallback:
            cached_db = None
            expected_models = [m for m in open_meteo_models if m not in ("hrrr", "nbm")]
            expected_models.extend(["ensemble", "visual_crossing", "tomorrow"])
            if cfg.get("country") == "US":
                expected_models.extend(["nws", "noaa_mos", "hrrr", "nbm"])
                
            for m in expected_models:
                if m not in results[icao]:
                    if cached_db is None:
                        cached_db = await get_latest_from_db({icao: cfg})
                    if icao in cached_db and m in cached_db[icao]:
                        logger.warning("Recovering missing model %s for %s from DB cache", m, icao)
                        results[icao][m] = cached_db[icao][m]
        # -------------------------


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
        "ecmwf": "ecmwf_ifs04",
        "icon": "icon_seamless",
        "gem": "gem_seamless",
        "jma": "jma_seamless",
        "hrrr": "ncep_hrrr_conus",
        "nbm": "ncep_nbm_conus",
        "arpege": "arpege_world",
        "ukmo": "ukmo_global_deterministic_10km",
        "bom": "bom_access_global",
    }
    api_models = ",".join(model_map[m] for m in model_list if m in model_map)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "hourly": "temperature_2m",
        "models": api_models,
        "timezone": "auto",
        "forecast_days": 2,
    }

    # Bypassing the Render IP rate limit by tunneling through Codetabs CORS proxy
    import urllib.parse
    import random
    qs = urllib.parse.urlencode(params)
    # Add random cachebuster
    raw_url = f"{OPEN_METEO_URL}?{qs}&cb={random.randint(1000, 99999)}"
    encoded_url = urllib.parse.quote(raw_url, safe='')
    fetch_url = f"https://api.codetabs.com/v1/proxy?quest={encoded_url}"

    results: Dict[str, ModelForecast] = {}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                logger.debug(f"OM TUNNEL URL: {fetch_url}")
                async with session.get(
                    fetch_url,
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
                            
                    if resp.status in (400, 401, 403):
                        _rate_limit_status["open_meteo"] = "limited"
                        return results

                    elif resp.status != 200:
                        logger.error("Open-Meteo HTTP %d for %s", resp.status, station)
                        return results
                    else:
                        data = await resp.json()
                        break # Primary success, break retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("Open-Meteo error for %s (attempt %d): %s, retrying...", station, attempt+1, e)
                await asyncio.sleep(1)
            else:
                logger.error("Open-Meteo final error for %s: %s", station, e)
                return results

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    if not daily and not hourly:
        logger.error("Open-Meteo returned no daily or hourly data for %s", station)
        return results

    tz_name = station_cfg.get("tz", "UTC")
    today_local = datetime.now(pytz.timezone(tz_name)).date()
    tomorrow_local = today_local + timedelta(days=1)

    for model_key, api_name in model_map.items():
        if model_key not in model_list:
            continue

        # 1. Try to compute from hourly data first
        hourly_computed = {} # date -> max_temp_c
        model_hourly_key = f"temperature_2m_{api_name}"
        if hourly and model_hourly_key in hourly and "time" in hourly:
            times = hourly["time"]
            temps = hourly[model_hourly_key]
            for i, t_str in enumerate(times):
                if i >= len(temps) or temps[i] is None:
                    continue
                d_str, h_str = t_str.split("T")
                h_val = int(h_str.split(":")[0])
                if 6 <= h_val <= 20: # 06:00 to 20:00 local
                    dt_date = date.fromisoformat(d_str)
                    if dt_date not in hourly_computed:
                        hourly_computed[dt_date] = []
                    hourly_computed[dt_date].append(float(temps[i]))
            for d in hourly_computed:
                hourly_computed[d] = max(hourly_computed[d])

        # 2. Extract from daily data as fallback
        daily_computed = {}
        model_daily_key = f"temperature_2m_max_{api_name}"
        if model_daily_key in daily:
            model_daily = daily[model_daily_key]
        elif len(model_list) == 1 and "temperature_2m_max" in daily:
            model_daily = daily["temperature_2m_max"]
        else:
            model_daily = None

        if model_daily and daily.get("time"):
            for i, d_str in enumerate(daily["time"]):
                if i >= len(model_daily) or model_daily[i] is None:
                    continue
                dt_date = date.fromisoformat(d_str)
                daily_computed[dt_date] = float(model_daily[i])

        all_dates = set(hourly_computed.keys()) | set(daily_computed.keys())
        if not all_dates:
            logger.debug("No data for model %s (%s) at station %s", model_key, api_name, station)
            continue

        for target_date in sorted(list(all_dates)):
            raw_high_c = hourly_computed.get(target_date)
            if raw_high_c is None:
                raw_high_c = daily_computed.get(target_date)

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
            if station_cfg.get("target_date"):
                requested_date = station_cfg["target_date"]
                
                # Check 1: Is this the exact date requested?
                if target_date == requested_date:
                    results[model_key] = forecast
                # Check 2: If it's a 1-day drift, only save it if we DON'T ALREADY have the exact date
                elif abs((target_date - requested_date).days) <= 1:
                    if model_key not in results or results[model_key].target_date != requested_date:
                        results[model_key] = forecast
            else:
                if model_key not in results or target_date in (today_local, tomorrow_local):
                    # If we already have a forecast and it's for today, don't overwrite it with tomorrow
                    if model_key in results and results[model_key].target_date == today_local and target_date == tomorrow_local:
                        pass
                    else:
                        results[model_key] = forecast

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
            hourly_url = points_data.get("properties", {}).get("forecastHourly")
            if not forecast_url and not hourly_url:
                return None

            temp_c_final = None
            target_date_final = None

            # Step 2: Try Hourly Aggregation First
            if hourly_url:
                async with session.get(
                    hourly_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        hourly_data = await resp.json()
                        periods = hourly_data.get("properties", {}).get("periods", [])
                        tz_name = station_cfg.get("tz", "UTC")
                        today_local = datetime.now(pytz.timezone(tz_name)).date()
                        
                        hourly_temps = []
                        for period in periods:
                            start_time = period.get("startTime", "")
                            try:
                                p_date = date.fromisoformat(start_time[:10])
                                # NWS uses proper ISO8601 with offset, so grab the hour directly
                                p_hour = int(start_time[11:13])
                                if p_date == today_local and 6 <= p_hour <= 20:
                                    t_val = float(period.get("temperature", 0))
                                    t_unit = period.get("temperatureUnit", "F")
                                    if t_unit == "F":
                                        t_val = (t_val - 32.0) * 5.0 / 9.0
                                    hourly_temps.append(t_val)
                            except Exception:
                                pass
                                
                        if hourly_temps:
                            temp_c_final = max(hourly_temps)
                            target_date_final = today_local

            # Step 3: Fallback to standard period forecast
            if temp_c_final is None and forecast_url:
                async with session.get(
                    forecast_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        forecast_data = await resp.json()
                        periods = forecast_data.get("properties", {}).get("periods", [])
                        for period in periods:
                            if period.get("isDaytime"):
                                temp_f = float(period.get("temperature", 0))
                                temp_unit = period.get("temperatureUnit", "F")
                                if temp_unit == "F":
                                    temp_c_final = (temp_f - 32.0) * 5.0 / 9.0
                                else:
                                    temp_c_final = temp_f

                                start_time = period.get("startTime", "")
                                try:
                                    target_date_final = date.fromisoformat(start_time[:10])
                                except (ValueError, IndexError):
                                    tz_name = station_cfg.get("tz", "UTC")
                                    target_date_final = datetime.now(pytz.timezone(tz_name)).date()
                                break

    except Exception as e:
        logger.error("NWS error for %s: %s", station, e)
        return None

    if temp_c_final is None or target_date_final is None:
        return None

    temp_f_final = temp_c_final * 9.0 / 5.0 + 32.0
    bias_c = await _get_bias(station, "nws", station_cfg)
    corrected_c = temp_c_final - bias_c
    corrected_f = corrected_c * 9.0 / 5.0 + 32.0
    weight = await _get_weight(station, "nws")

    forecast = ModelForecast(
        station=station,
        model_name="nws",
        target_date=target_date_final,
        raw_high_c=temp_c_final,
        raw_high_f=temp_f_final,
        bias_corrected_c=corrected_c,
        bias_corrected_f=corrected_f,
        weight=weight,
    )

    await tracker.store_forecast(
        station, target_date_final, "nws",
        temp_c_final, temp_f_final, corrected_c, corrected_f,
    )
    return forecast


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

        print("NWS PERIODS:", [p["name"] + " | " + str(p["temperature"]) for p in forecast_data.get("properties", {}).get("periods", [])[:5]])

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
            "include": "days,hours",
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

        tz_name = station_cfg.get("tz", "UTC")
        target_date = datetime.now(pytz.timezone(tz_name)).date()

        raw_high_f = None
        hours = days[0].get("hours", [])
        if hours:
            hourly_temps = []
            for h in hours:
                dt_str = h.get("datetime", "")
                try:
                    h_val = int(dt_str.split(":")[0])
                    if 6 <= h_val <= 20:
                        t_val = h.get("temp")
                        if t_val is not None:
                            hourly_temps.append(float(t_val))
                except (ValueError, IndexError):
                    pass
            
            if hourly_temps:
                raw_high_f = max(hourly_temps)

        if raw_high_f is None:
            raw_high_f = float(days[0].get("tempmax"))

        raw_high_c = (raw_high_f - 32.0) * 5.0 / 9.0

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
# Tomorrow.io fetch (requires key)
# ---------------------------------------------------------------------------
async def _fetch_tomorrow_io(station: str, lat: float, lon: float,
                                 station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch hourly data from Tomorrow.io API and compute daily high."""
    import os
    api_key = os.environ.get("TOMORROWIO_API_KEY")
    if not api_key:
        return None

    try:
        url = "https://api.tomorrow.io/v4/timelines"
        params = {
            "location": f"{lat},{lon}",
            "fields": ["temperature"],
            "timesteps": "1h",
            "units": "imperial",
            "apikey": api_key
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
                    logger.debug("Tomorrow.io HTTP %d for %s", resp.status, station)
                    if resp.status == 429:
                        _rate_limit_status["tomorrow"] = "limited"
                    return None
                data = await resp.json()

        intervals = data.get("data", {}).get("timelines", [])
        if not intervals:
            return None
        intervals = intervals[0].get("intervals", [])

        tz_name = station_cfg.get("tz", "UTC")
        target_date = station_cfg.get("target_date", datetime.now(pytz.timezone(tz_name)).date())

        hourly_temps = []
        for interval in intervals:
            dt_str = interval.get("startTime", "")
            try:
                dt_utc = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
                dt_local = dt_utc.astimezone(pytz.timezone(tz_name))
                
                if dt_local.date() == target_date:
                    if 6 <= dt_local.hour <= 20:
                        t_val = interval.get("values", {}).get("temperature")
                        if t_val is not None:
                            hourly_temps.append(float(t_val))
            except (ValueError, IndexError):
                pass
        
        if not hourly_temps:
            return None

        raw_high_f = max(hourly_temps)
        raw_high_c = (raw_high_f - 32.0) * 5.0 / 9.0

        bias_c = await _get_bias(station, "tomorrow", station_cfg)
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0
        weight = await _get_weight(station, "tomorrow")

        forecast = ModelForecast(
            station=station, model_name="tomorrow",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
        )
        await tracker.store_forecast(
            station, target_date, "tomorrow",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )
        return forecast

    except Exception as e:
        logger.error("Tomorrow.io error for %s: %s", station, e)
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
                if resp.status == 429:
                    logger.warning("Open-Meteo Ensemble HTTP 429: Rate limit hit. Defaulting to 1.0 curve.")
                    return None
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
            if not rows: continue
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
# ---------------------------------------------------------------------------
# 31-Member Open-Meteo GFS Ensemble
# ---------------------------------------------------------------------------
async def fetch_ensemble(lat: float, lon: float, unit: str = "F") -> List[float]:
    """Fetch 31 iterations of the GFS Atmosphere Simulation representing a direct probability matrix."""
    api_key = os.environ.get("OPEN_METEO_API_KEY")
    url = "https://customer-ensemble-api.open-meteo.com/v1/ensemble" if api_key else "https://ensemble-api.open-meteo.com/v1/ensemble"
    
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max",
        "models": "gfs_seamless",
        "timezone": "auto",
        "forecast_days": 2,
    }
    if api_key:
        params["apikey"] = api_key

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        async with session.get(url, params=params, timeout=15) as resp:
            if resp.status in (401, 403) and api_key:
                logger.warning("Ensemble API key unauthorized. Falling back to free tier.")
                url = "https://ensemble-api.open-meteo.com/v1/ensemble"
                if "apikey" in params:
                    del params["apikey"]
                async with session.get(url, params=params, timeout=15) as resp_fallback:
                    if resp_fallback.status != 200:
                        logger.error("Ensemble API fallback failed with HTTP %d", resp_fallback.status)
                        return []
                    data = await resp_fallback.json()
            elif resp.status != 200:
                logger.error("Ensemble API fetch failed with HTTP %d", resp.status)
                return []
            else:
                data = await resp.json()
    
    daily = data.get("daily", {})
    members = []
    for i in range(1, 32):
        key = f"temperature_2m_max_member{i:02d}"
        vals = daily.get(key, [None])
        temp_c = vals[0]  # today's forecast
        if temp_c is not None:
            if unit == "F":
                members.append(round(temp_c * 9/5 + 32, 1))
            else:
                members.append(round(temp_c, 1))
    
    return members

# ---------------------------------------------------------------------------
# Dewpoint Physical Constraints
# ---------------------------------------------------------------------------
def dewpoint_adjustment(current_temp: float, dewpoint: float, predicted_high: float, unit: str = "F"):
    """
    If dewpoint spread is very narrow, cap the predicted high.
    High humidity prevents rapid warming.
    """
    spread = current_temp - dewpoint
    
    if unit == "C":
        narrow_threshold = 5.0
        max_additional_rise = 1.0
    else:
        narrow_threshold = 9.0
        max_additional_rise = 1.8
    
    if spread < narrow_threshold:
        # Very humid — cap warming potential
        capped = current_temp + max_additional_rise
        if predicted_high > capped:
            return capped, f"⚠️ High humidity (dewpoint {dewpoint:.1f}°{unit}) caps warming"
    
    return predicted_high, None

# ---------------------------------------------------------------------------
# Daily High Prediction (5-step Bayesian Blend)
# ---------------------------------------------------------------------------
async def calculate_daily_high(models_data: Dict[str, ModelForecast], metar: Optional[Any], 
                               metar_trend: Optional[Dict], ensemble_members: List[float],
                               local_hour: int, station: str, unit: str = "F") -> float:
    """
    5-step prediction combining all data sources:
    
    Step 1: Inverse-MAE weighted model average (with bias correction)
    Step 2: METAR floor (can't be lower than observed high)
    Step 3: METAR trend projection (if rising, where does it end up?)
    Step 4: Time-of-day blend
    Step 5: Dewpoint constraint
    """
    
    DEFAULT_MAE = {
        # Existing
        "gfs": 1.8, "ecmwf": 1.5, "icon": 2.0, "gem": 2.2, "jma": 2.0,
        "nws": 1.5, "noaa_mos": 1.3, "visual_crossing": 2.0,
        # New
        "hrrr": 1.2,
        "nbm": 1.1,
        "arpege": 2.0,
        "ukmo": 1.8,
        "bom": 2.2,
    }
    
    # Step 1: Inverse-MAE weighted average with bias correction
    weights = {}
    temps = {}
    for name, forecast in models_data.items():
        try:
            temp = forecast.bias_corrected_c if unit == "C" else forecast.bias_corrected_f
            if temp is None:
                continue
            
            # Apply yesterday's bias explicitly
            recent_bias = await tracker.get_recent_bias(station, name, days=7)
            temp = temp - recent_bias  # Correct for known model bias
            
            mae = DEFAULT_MAE.get(name, 2.0)
            weights[name] = 1.0 / max(0.5, mae)
            temps[name] = temp
        except AttributeError:
            continue
    
    if not temps:
        return 0.0
    
    total_w = sum(weights.values())
    model_pred = sum(temps[n] * (weights[n] / total_w) for n in temps)
    
    # Add ensemble median if available
    if ensemble_members and len(ensemble_members) >= 20:
        ensemble_median = sorted(ensemble_members)[len(ensemble_members) // 2]
        # Ensemble gets weight equivalent to best model (MAE ~1.3)
        ensemble_weight = 1.0 / 1.3
        model_pred = (model_pred * total_w + ensemble_median * ensemble_weight) / (total_w + ensemble_weight)
    
    # Step 2: METAR floor
    current_high = None
    if metar and hasattr(metar, 'velocity') and metar.velocity:
        current_high = metar.velocity.day_high if unit == "C" else metar.velocity.day_high_f
        if current_high:
            model_pred = max(model_pred, current_high)
    
    # Step 3: METAR trend projection
    if metar_trend and metar_trend.get("projected_high"):
        trend_pred = metar_trend["projected_high"]
        
        # Blend based on time of day and whether temp is rising
        if local_hour >= 12 and metar_trend.get("is_rising"):
            # Afternoon and rising → trend is very reliable
            metar_trend_weight = 3.0
        elif local_hour >= 10:
            metar_trend_weight = 1.5
        else:
            metar_trend_weight = 0.5
        
        total_w_with_trend = total_w + metar_trend_weight
        model_pred = ((model_pred * total_w) + (trend_pred * metar_trend_weight)) / total_w_with_trend
    
    # Step 4: Time-of-day blend with METAR
    if local_hour < 8:
        blend = 0.0
    elif local_hour < 14:
        blend = (local_hour - 8) / 12.0
    else:
        blend = 0.5 + (local_hour - 14) / 12.0
    blend = max(0.0, min(1.0, blend))
    
    if current_high is not None:
        final = (1 - blend) * model_pred + blend * current_high
    else:
        final = model_pred
    
    # Step 5: Dewpoint constraint
    if metar:
        dewpoint = getattr(metar, "dewpoint_f", None) if unit == "F" else getattr(metar, "dewpoint_c", None)
        if dewpoint is not None:
            current_temp = getattr(metar, "temp_f", None) if unit == "F" else getattr(metar, "temp_c", None)
            if current_temp is not None:
                final, dew_note = dewpoint_adjustment(current_temp, dewpoint, final, unit)
    
    return round(final, 1)
