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
from typing import Dict, List, Optional

import aiohttp

import tracker

logger = logging.getLogger("models")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
NWS_POINTS_URL = "https://api.weather.gov/points"
TOMORROW_IO_URL = "https://api.tomorrow.io/v4/weather/forecast"
OPENWEATHER_URL = "https://api.openweathermap.org/data/3.0/onecall"
WEATHERBIT_URL = "https://api.weatherbit.io/v2.0/forecast/daily"
NOAA_MOS_URL = "https://aviationweather.gov/cgi-bin/data/mos.php"
USER_AGENT = "WeatherEdgeBot/1.0"
EWMA_ALPHA = 0.15  # ~90% weight to last 13 data points

# Default MAE per model (°C) — used for distribution calculations
DEFAULT_MAE = {
    "gfs": 2.0, "ecmwf": 1.5, "icon": 2.0, "nws": 1.8, "noaa": 1.8,
    "tomorrow_io": 2.0, "openweather": 2.2, "weatherbit": 2.5,
    "noaa_mos": 1.8, "ensemble": 1.0,
}

# Rate limit tracking (resets per scan cycle)
_rate_limit_status: Dict[str, str] = {}  # source -> "ok" | "limited" | "no_key"
_weatherbit_calls_today: int = 0
_weatherbit_reset_date: Optional[date] = None


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
        model_list = cfg.get("models", ["gfs", "ecmwf", "icon"])
        lat, lon = cfg["lat"], cfg["lon"]

        # 1) Open-Meteo deterministic (GFS, ECMWF, ICON)
        open_meteo_models = [m for m in model_list if m in ("gfs", "ecmwf", "icon")]
        if open_meteo_models:
            om_results = await _fetch_open_meteo(icao, lat, lon, open_meteo_models, cfg)
            results[icao].update(om_results)

        # 2) NWS (US stations only)
        if cfg.get("country") == "US":
            nws_result = await _fetch_nws(icao, lat, lon, cfg)
            if nws_result:
                results[icao]["nws"] = nws_result

            noaa_result = await _fetch_noaa(icao, lat, lon, cfg)
            if noaa_result:
                results[icao]["noaa"] = noaa_result

        # 3) Tomorrow.io
        if _rate_limit_status.get("tomorrow_io") != "limited":
            tio_result = await _fetch_tomorrow_io(icao, lat, lon, cfg)
            if tio_result:
                results[icao]["tomorrow_io"] = tio_result

        # 4) OpenWeather
        if _rate_limit_status.get("openweather") != "limited":
            ow_result = await _fetch_openweather(icao, lat, lon, cfg)
            if ow_result:
                results[icao]["openweather"] = ow_result

        # 5) Weatherbit (strict 50/day limit)
        if _rate_limit_status.get("weatherbit") != "limited":
            wb_result = await _fetch_weatherbit(icao, lat, lon, cfg)
            if wb_result:
                results[icao]["weatherbit"] = wb_result

        # 6) NOAA MOS (US stations only)
        if cfg.get("country") == "US":
            mos_result = await _fetch_noaa_mos(icao, cfg)
            if mos_result:
                results[icao]["noaa_mos"] = mos_result

        # 7) Open-Meteo Ensemble
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
        "ecmwf": "ecmwf_ifs04",
        "icon": "icon_seamless",
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

    results: Dict[str, ModelForecast] = {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                OPEN_METEO_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.error("Open-Meteo HTTP %d for %s", resp.status, station)
                    return results
                data = await resp.json()
    except Exception as e:
        logger.error("Open-Meteo error for %s: %s", station, e)
        return results

    # Open-Meteo returns separate daily arrays per model
    # The response structure varies — handle both flat and per-model formats
    today = date.today()
    tomorrow = date.today() + timedelta(days=1)

    for model_key, api_name in model_map.items():
        if model_key not in model_list:
            continue

        # Try per-model format first (newer API)
        daily_key = f"temperature_2m_max"
        model_daily = None

        if "daily" in data:
            daily = data["daily"]
            # Check for model-specific key
            model_specific_key = f"temperature_2m_max_{api_name}"
            if model_specific_key in daily:
                model_daily = daily[model_specific_key]
            elif daily_key in daily:
                model_daily = daily[daily_key]

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
            results[model_key] = forecast

            # Store in database
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
    try:
        async with aiohttp.ClientSession() as session:
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
                target_date = date.today()

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
    try:
        async with aiohttp.ClientSession() as session:
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
                target_date = date.today()

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
# Tomorrow.io fetch (500 free/day)
# ---------------------------------------------------------------------------
async def _fetch_tomorrow_io(station: str, lat: float, lon: float,
                              station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch from Tomorrow.io weather API."""
    api_key = os.environ.get("TOMORROWIO_API_KEY")
    if not api_key:
        _rate_limit_status["tomorrow_io"] = "no_key"
        return None

    try:
        params = {
            "location": f"{lat},{lon}",
            "apikey": api_key,
            "timesteps": "1d",
            "units": "metric",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                TOMORROW_IO_URL, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 429:
                    _rate_limit_status["tomorrow_io"] = "limited"
                    logger.warning("Tomorrow.io rate limit reached")
                    return None
                if resp.status != 200:
                    logger.warning("Tomorrow.io HTTP %d for %s", resp.status, station)
                    return None
                data = await resp.json()

        _rate_limit_status["tomorrow_io"] = "ok"

        # Parse daily data
        timelines = data.get("timelines", {}).get("daily", [])
        if not timelines:
            return None

        # Use tomorrow's forecast (index 1) or today (index 0)
        day = timelines[1] if len(timelines) > 1 else timelines[0]
        values = day.get("values", {})
        raw_high_c = float(values.get("temperatureMax", 0))
        raw_high_f = raw_high_c * 9.0 / 5.0 + 32.0

        target_date_str = day.get("time", "")[:10]
        try:
            target_date = date.fromisoformat(target_date_str)
        except ValueError:
            target_date = date.today() + timedelta(days=1)

        bias_c = await _get_bias(station, "tomorrow_io", station_cfg)
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0
        weight = await _get_weight(station, "tomorrow_io")

        forecast = ModelForecast(
            station=station, model_name="tomorrow_io",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
        )
        await tracker.store_forecast(
            station, target_date, "tomorrow_io",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )
        return forecast

    except Exception as e:
        logger.error("Tomorrow.io error for %s: %s", station, e)
        return None


# ---------------------------------------------------------------------------
# OpenWeather fetch (1000 free/day)
# ---------------------------------------------------------------------------
async def _fetch_openweather(station: str, lat: float, lon: float,
                              station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch from OpenWeather OneCall API."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        _rate_limit_status["openweather"] = "no_key"
        return None

    try:
        params = {
            "lat": lat, "lon": lon,
            "appid": api_key,
            "units": "metric",
            "exclude": "minutely,hourly,alerts",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                OPENWEATHER_URL, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 429:
                    _rate_limit_status["openweather"] = "limited"
                    logger.warning("OpenWeather rate limit reached")
                    return None
                if resp.status != 200:
                    logger.warning("OpenWeather HTTP %d for %s", resp.status, station)
                    return None
                data = await resp.json()

        _rate_limit_status["openweather"] = "ok"

        daily = data.get("daily", [])
        if len(daily) < 2:
            return None

        day = daily[1]  # Tomorrow
        raw_high_c = float(day.get("temp", {}).get("max", 0))
        raw_high_f = raw_high_c * 9.0 / 5.0 + 32.0
        target_date = date.today() + timedelta(days=1)

        bias_c = await _get_bias(station, "openweather", station_cfg)
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0
        weight = await _get_weight(station, "openweather")

        forecast = ModelForecast(
            station=station, model_name="openweather",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
        )
        await tracker.store_forecast(
            station, target_date, "openweather",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )
        return forecast

    except Exception as e:
        logger.error("OpenWeather error for %s: %s", station, e)
        return None


# ---------------------------------------------------------------------------
# Weatherbit fetch (50 free/day — strict limit)
# ---------------------------------------------------------------------------
async def _fetch_weatherbit(station: str, lat: float, lon: float,
                             station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch from Weatherbit daily forecast. Tracks call count."""
    global _weatherbit_calls_today, _weatherbit_reset_date

    api_key = os.environ.get("WEATHERBIT_API_KEY")
    if not api_key:
        _rate_limit_status["weatherbit"] = "no_key"
        return None

    # Reset daily counter
    today = date.today()
    if _weatherbit_reset_date != today:
        _weatherbit_calls_today = 0
        _weatherbit_reset_date = today

    if _weatherbit_calls_today >= 45:  # Buffer of 5 from 50 limit
        _rate_limit_status["weatherbit"] = "limited"
        logger.debug("Weatherbit daily limit approaching (%d/50)", _weatherbit_calls_today)
        return None

    try:
        params = {
            "lat": lat, "lon": lon,
            "key": api_key,
            "days": 2,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                WEATHERBIT_URL, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                _weatherbit_calls_today += 1
                if resp.status == 429:
                    _rate_limit_status["weatherbit"] = "limited"
                    logger.warning("Weatherbit rate limit reached")
                    return None
                if resp.status != 200:
                    logger.warning("Weatherbit HTTP %d for %s", resp.status, station)
                    return None
                data = await resp.json()

        _rate_limit_status["weatherbit"] = "ok"

        days = data.get("data", [])
        if len(days) < 2:
            return None

        day = days[1]  # Tomorrow
        raw_high_c = float(day.get("max_temp", 0))
        raw_high_f = raw_high_c * 9.0 / 5.0 + 32.0

        date_str = day.get("datetime", "")
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            target_date = date.today() + timedelta(days=1)

        bias_c = await _get_bias(station, "weatherbit", station_cfg)
        corrected_c = raw_high_c - bias_c
        corrected_f = corrected_c * 9.0 / 5.0 + 32.0
        weight = await _get_weight(station, "weatherbit")

        forecast = ModelForecast(
            station=station, model_name="weatherbit",
            target_date=target_date,
            raw_high_c=raw_high_c, raw_high_f=raw_high_f,
            bias_corrected_c=corrected_c, bias_corrected_f=corrected_f,
            weight=weight,
        )
        await tracker.store_forecast(
            station, target_date, "weatherbit",
            raw_high_c, raw_high_f, corrected_c, corrected_f,
        )
        return forecast

    except Exception as e:
        logger.error("Weatherbit error for %s: %s", station, e)
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
        params = {"ids": station, "type": "mav"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                NOAA_MOS_URL, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.debug("NOAA MOS HTTP %d for %s", resp.status, station)
                    return None
                text = await resp.text()

        if not text or station not in text:
            return None

        # Parse MAX line from MOS text
        max_temp_f = _parse_mos_max_temp(text)
        if max_temp_f is None:
            logger.debug("NOAA MOS: no MAX temp found for %s", station)
            return None

        raw_high_f = float(max_temp_f)
        raw_high_c = (raw_high_f - 32.0) * 5.0 / 9.0
        target_date = date.today()  # MOS MAX is for today

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
        logger.error("NOAA MOS error for %s: %s", station, e)
        return None


def _parse_mos_max_temp(text: str) -> Optional[float]:
    """Extract MAX temperature from MOS text output."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("X/N") or stripped.startswith("N/X"):
            # Format: X/N or N/X followed by values
            parts = stripped.split()
            for part in parts[1:]:
                try:
                    val = int(part)
                    if -40 <= val <= 130:  # Reasonable F range
                        return float(val)
                except ValueError:
                    continue
    return None


# ---------------------------------------------------------------------------
# Open-Meteo Ensemble fetch (Free, 31 members)
# ---------------------------------------------------------------------------
async def _fetch_open_meteo_ensemble(station: str, lat: float, lon: float,
                                      station_cfg: dict) -> Optional[ModelForecast]:
    """Fetch 31-member GFS ensemble from Open-Meteo."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "models": "gfs_seamless",
            "daily": "temperature_2m_max",
            "timezone": "auto"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                OPEN_METEO_ENSEMBLE_URL, params=params,
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
            target_date = date.today() + timedelta(days=1)

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
# Confluence logic
# ---------------------------------------------------------------------------
def get_model_confluence(models_data: Dict[str, ModelForecast], unit: str = "F") -> str:
    """Generate a confluence report card for all available models."""
    if not models_data:
        return ""
        
    temps = []
    for model_name, forecast in models_data.items():
        if unit.upper() == "F":
            temps.append(forecast.bias_corrected_f)
        else:
            temps.append(forecast.bias_corrected_c)
            
    if not temps:
        return ""
        
    avg_temp = sum(temps) / len(temps)
    
    # 1°C is equivalent to 1.8°F
    threshold = 1.0 if unit.upper() == "C" else 1.8
    
    agree_count = sum(1 for t in temps if abs(t - avg_temp) <= threshold)
    agreement_pct = (agree_count / len(temps)) * 100
    
    return f"🌡️ {len(temps)} Models: Avg {avg_temp:.1f}°{unit} (±{threshold}°{unit}) | {agreement_pct:.0f}% Strong Agreement"
