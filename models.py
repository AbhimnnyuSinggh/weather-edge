"""
models.py — Model Forecasts, Bias Correction, Dynamic Weighting

Fetches weather model forecasts from Open-Meteo (GFS, ECMWF, ICON) and
NWS (US stations only). Applies station bias correction. Maintains dynamic
model weights based on historical accuracy. Detects intra-day momentum
divergence.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import aiohttp

import tracker

logger = logging.getLogger("models")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NWS_POINTS_URL = "https://api.weather.gov/points"
USER_AGENT = "WeatherEdgeBot/1.0"
EWMA_ALPHA = 0.15  # ~90% weight to last 13 data points


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
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.utcnow()


# ---------------------------------------------------------------------------
# Fetch all stations
# ---------------------------------------------------------------------------
async def fetch_all_stations(stations_cfg: dict) -> Dict[str, Dict[str, ModelForecast]]:
    """
    For each active station, fetch forecasts from all configured models.
    Returns nested dict: {station: {model_name: ModelForecast}}
    """
    results: Dict[str, Dict[str, ModelForecast]] = {}

    for icao, cfg in stations_cfg.items():
        results[icao] = {}
        model_list = cfg.get("models", ["gfs", "ecmwf", "icon"])

        # Open-Meteo models
        open_meteo_models = [m for m in model_list if m in ("gfs", "ecmwf", "icon")]
        if open_meteo_models:
            om_results = await _fetch_open_meteo(
                icao, cfg["lat"], cfg["lon"], open_meteo_models, cfg
            )
            results[icao].update(om_results)

        # NWS (US stations only)
        if "nws" in model_list and cfg.get("country") == "US":
            nws_result = await _fetch_nws(icao, cfg["lat"], cfg["lon"], cfg)
            if nws_result:
                results[icao]["nws"] = nws_result

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
    tomorrow = date.today()

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
