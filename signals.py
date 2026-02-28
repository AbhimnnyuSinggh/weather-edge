"""
signals.py — 4 Trade Types, Confidence Scoring, EV Calculation

The brain of the bot. Takes all data (METAR, models, markets,
probabilities, wallet) and generates trade signals with confidence
scores and expected values.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import pytz

import models as models_mod
import tracker
from markets import BinInfo, MarketBin, MarketGroup
from metar import StationMETAR

logger = logging.getLogger("signals")


# ---------------------------------------------------------------------------
# Signal data model
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    trade_type: str       # 'lockin_yes', 'no_tail', 'forecast_yes', 'ladder'
    station: str
    target_date: date
    side: str             # 'YES' or 'NO'
    bin_label: str
    bins: List[dict] = field(default_factory=list)   # for ladders
    entry_price: float = 0.0
    confidence_score: int = 0
    confidence_components: dict = field(default_factory=dict)
    ev: float = 0.0
    win_probability: float = 0.0
    profit_if_win: float = 0.0
    loss_if_lose: float = 0.0
    market_id: str = ""
    polymarket_url: str = ""
    book_depth: int = 0
    metar_summary: str = ""
    model_summary: str = ""
    shares: float = 0.0
    cost: float = 0.0
    city: str = ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def generate_all(metar_data: dict, model_data: dict,
                        market_data: dict, prob_data: dict,
                        wallet_state, config: dict,
                        stations_cfg: dict) -> List[Signal]:
    """
    For each station/date combo, generate all qualifying signals.
    """
    all_signals: List[Signal] = []
    min_conf = config.get("trading", {}).get("min_confidence", 40)
    trade_rules = config.get("trade_types", {})

    for market_key, group in market_data.items():
        station = group.get("station") if isinstance(group, dict) else group.station
        if not station or station == "UNKNOWN":
            continue

        cfg = stations_cfg.get(station)
        if not cfg:
            continue

        target_date_val = group.get("target_date") if isinstance(group, dict) else group.target_date
        city = group.get("city", cfg.get("city", "")) if isinstance(group, dict) else group.city
        bins_list = group.get("bins", []) if isinstance(group, dict) else group.bins

        metar = metar_data.get(station)
        prob = prob_data.get(station, {})
        models = model_data.get(station, {})

        tz = pytz.timezone(cfg["timezone"])
        now_local = datetime.now(tz)
        local_hour = now_local.hour
        is_coastal = cfg.get("is_coastal", False)
        unit = cfg.get("unit", "C")

        # Determine if this is an afternoon scenario
        min_hour = (trade_rules.get("lockin_yes", {}).get("coastal_min_local_hour", 15)
                    if is_coastal
                    else trade_rules.get("lockin_yes", {}).get("non_coastal_min_local_hour", 14))

        is_afternoon = local_hour >= min_hour
        station_today = now_local.date()  # Use station-local date, not UTC
        is_today = target_date_val == station_today if target_date_val else False

        # --- Afternoon plays (today's market only) ---
        if is_afternoon and is_today and metar:
            # Lock-in YES
            lockin = check_lockin_yes(
                station, city, target_date_val, metar, bins_list, prob,
                trade_rules, cfg, local_hour,
            )
            if lockin and lockin.confidence_score >= min_conf:
                all_signals.append(lockin)

            # NO tails
            no_tails = check_no_tail(
                station, city, target_date_val, metar, bins_list, prob,
                trade_rules, cfg,
            )
            all_signals.extend(s for s in no_tails if s.confidence_score >= min_conf)

        # --- Forecast plays (tomorrow's market or early today) ---
        if not is_afternoon or not is_today:
            forecast = check_forecast_yes(
                station, city, target_date_val, models, bins_list,
                trade_rules, cfg, metar,
            )
            if forecast and forecast.confidence_score >= min_conf:
                all_signals.append(forecast)

            ladder = check_ladder(
                station, city, target_date_val, models, bins_list,
                trade_rules, cfg,
            )
            if ladder and ladder.confidence_score >= min_conf:
                all_signals.append(ladder)

    return all_signals


# ---------------------------------------------------------------------------
# Lock-in YES
# ---------------------------------------------------------------------------
def check_lockin_yes(station: str, city: str, target_date_val: date,
                      metar: StationMETAR, bins: list, prob: dict,
                      trade_rules: dict, station_cfg: dict,
                      local_hour: int) -> Optional[Signal]:
    """Check Lock-in YES conditions and score."""
    rules = trade_rules.get("lockin_yes", {})
    high_set_prob = prob.get("high_set_prob", 0)

    # Condition 1: high set probability
    if high_set_prob < rules.get("min_high_set_probability", 0.80):
        return None

    # Condition 2+3: velocity is negative, consecutive falling >= 3
    if not metar.velocity or metar.velocity.trend != "falling":
        return None
    if metar.velocity.consecutive_falling < rules.get("min_falling_readings", 3):
        return None

    # Find the bin containing the day's high
    day_high = metar.velocity.day_high
    unit = station_cfg.get("unit", "C")
    if unit == "F":
        day_high = metar.velocity.day_high_f

    target_bin = None
    for mbin in bins:
        b = mbin.bin if hasattr(mbin, "bin") else mbin
        low = b.low if hasattr(b, "low") else b.get("low", 0)
        high = b.high if hasattr(b, "high") else b.get("high", 0)
        label = b.label if hasattr(b, "label") else b.get("label", "")
        yes_price = mbin.yes_price if hasattr(mbin, "yes_price") else mbin.get("yes_price", 0)

        if low <= day_high < high or (low <= day_high and day_high <= high):
            target_bin = mbin
            break

    if not target_bin:
        return None

    # Get price
    yes_price = target_bin.yes_price if hasattr(target_bin, "yes_price") else target_bin.get("yes_price", 0)
    if yes_price > rules.get("max_entry_price", 0.75):
        return None

    bin_label = target_bin.bin.label if hasattr(target_bin, "bin") else target_bin.get("bin", {}).get("label", "")
    market_id = target_bin.market_id if hasattr(target_bin, "market_id") else target_bin.get("market_id", "")
    poly_url = target_bin.polymarket_url if hasattr(target_bin, "polymarket_url") else target_bin.get("polymarket_url", "")

    # --- Confidence Score ---
    score = 0
    components = {}

    # High set probability (max 40)
    pts_high = min(40, int(high_set_prob * 40))
    score += pts_high
    components["high_set"] = pts_high

    # METAR velocity (max 25)
    vel_str = min(1.0, abs(metar.velocity.velocity) / 2.0)
    pts_vel = int(vel_str * 25)
    score += pts_vel
    components["velocity"] = pts_vel

    # Model agreement (max 15)
    from models import ModelForecast
    models_in_bin = 0
    model_summaries = []
    # (model_data is not passed directly — we check from prob/station data)
    # For now count based on available data
    pts_model = 0
    components["model_agreement"] = pts_model
    score += pts_model

    # Rounding edge (max 10)
    pts_rounding = 0
    if metar.rounding_edge:
        if metar.rounding_edge.exact_bin == bin_label:
            pts_rounding = 10 if not metar.rounding_edge.is_near_boundary else 5
    score += pts_rounding
    components["rounding_edge"] = pts_rounding

    # Coastal penalty (up to -15)
    is_coastal = station_cfg.get("is_coastal", False)
    coastal_pen = 0
    if is_coastal and local_hour < 16:
        coastal_pen = -15
        score += coastal_pen
    components["coastal_penalty"] = coastal_pen

    score = max(0, min(100, score))

    # --- EV Calculation ---
    win_prob = high_set_prob * 0.95
    # Placeholder sizing — allocator will finalize
    est_shares = 10
    profit_if_win = est_shares * (1.0 - yes_price)
    loss_if_lose = est_shares * yes_price
    ev = win_prob * profit_if_win - (1 - win_prob) * loss_if_lose

    # METAR summary
    vel = metar.velocity
    trend_arrow = "→".join(
        str(int(metar.temp_c - i)) if unit == "C" else str(int(metar.temp_f - i))
        for i in range(min(4, vel.consecutive_falling + 1))
    )
    metar_summary = (
        f"Now: {metar.temp_c if unit=='C' else metar.temp_f:.0f}°{unit} | "
        f"Day High: {day_high:.0f}°{unit} (set {vel.hours_since_high:.1f}h ago)\n"
        f"Trend: falling {vel.hours_since_high:.1f}hrs, {vel.velocity:.1f}°{unit}/hr"
    )

    return Signal(
        trade_type="lockin_yes",
        station=station,
        city=city,
        target_date=target_date_val,
        side="YES",
        bin_label=bin_label,
        entry_price=yes_price,
        confidence_score=score,
        confidence_components=components,
        ev=round(ev, 2),
        win_probability=round(win_prob, 3),
        profit_if_win=round(profit_if_win, 2),
        loss_if_lose=round(loss_if_lose, 2),
        market_id=market_id,
        polymarket_url=poly_url,
        metar_summary=metar_summary,
    )


# ---------------------------------------------------------------------------
# NO Tail
# ---------------------------------------------------------------------------
def check_no_tail(station: str, city: str, target_date_val: date,
                   metar: StationMETAR, bins: list, prob: dict,
                   trade_rules: dict, station_cfg: dict) -> List[Signal]:
    """Check NO tail opportunities for bins above the day's high."""
    rules = trade_rules.get("no_tail", {})
    signals: List[Signal] = []

    high_set_prob = prob.get("high_set_prob", 0)
    bin_decay = prob.get("bin_decay", {})
    unit = station_cfg.get("unit", "C")

    if not metar.velocity or metar.velocity.trend != "falling":
        return signals

    day_high = metar.velocity.day_high if unit == "C" else metar.velocity.day_high_f

    for mbin in bins:
        b = mbin.bin if hasattr(mbin, "bin") else mbin
        low = b.low if hasattr(b, "low") else b.get("low", 0)
        high_val = b.high if hasattr(b, "high") else b.get("high", 0)
        label = b.label if hasattr(b, "label") else b.get("label", "")
        yes_price = mbin.yes_price if hasattr(mbin, "yes_price") else mbin.get("yes_price", 0)
        market_id = mbin.market_id if hasattr(mbin, "market_id") else mbin.get("market_id", "")
        poly_url = mbin.polymarket_url if hasattr(mbin, "polymarket_url") else mbin.get("polymarket_url", "")

        # Must be above the day's high
        if low <= day_high:
            continue

        # Ranges above high
        bin_width = max(1, high_val - low)
        ranges_above = (low - day_high) / bin_width
        if ranges_above < rules.get("min_ranges_above_high", 2):
            continue

        # YES price minimum
        if yes_price < rules.get("min_yes_price", 0.20):
            continue

        # NO cost maximum
        no_cost = 1.0 - yes_price
        if no_cost > rules.get("max_no_cost", 0.80):
            continue

        # Decay impossibility
        bin_prob = bin_decay.get(label, 0.5)
        impossibility = 1.0 - bin_prob
        if impossibility < rules.get("min_decay_impossibility", 0.92):
            continue

        # --- Confidence Score ---
        score = 0
        components = {}

        # Decay impossibility (max 40)
        pts_decay = min(40, int(impossibility * 40))
        score += pts_decay
        components["decay"] = pts_decay

        # METAR velocity (max 25)
        vel_str = min(1.0, abs(metar.velocity.velocity) / 2.0)
        pts_vel = int(vel_str * 25)
        score += pts_vel
        components["velocity"] = pts_vel

        # Distance from high (max 20)
        pts_dist = min(20, int(ranges_above * 7))
        score += pts_dist
        components["distance"] = pts_dist

        # Model cap agreement (max 15) — placeholder
        pts_model = 0
        components["model_cap"] = pts_model
        score += pts_model

        score = max(0, min(100, score))

        # EV
        est_shares = 10
        profit_if_win = est_shares * yes_price  # NO win pays YES price per share
        loss_if_lose = est_shares * no_cost
        ev = impossibility * profit_if_win - (1 - impossibility) * loss_if_lose

        signals.append(Signal(
            trade_type="no_tail",
            station=station,
            city=city,
            target_date=target_date_val,
            side="NO",
            bin_label=label,
            entry_price=no_cost,
            confidence_score=score,
            confidence_components=components,
            ev=round(ev, 2),
            win_probability=round(impossibility, 3),
            profit_if_win=round(profit_if_win, 2),
            loss_if_lose=round(loss_if_lose, 2),
            market_id=market_id,
            polymarket_url=poly_url,
            metar_summary=f"Day High: {day_high:.0f}°{unit}, {ranges_above:.1f} ranges above",
        ))

    return signals


# ---------------------------------------------------------------------------
# Forecast YES
# ---------------------------------------------------------------------------
def check_forecast_yes(station: str, city: str, target_date_val: date,
                        models_data: dict, bins: list,
                        trade_rules: dict, station_cfg: dict,
                        metar: Optional[StationMETAR] = None) -> Optional[Signal]:
    """Check for forecast-based YES mispricing."""
    rules = trade_rules.get("forecast_yes", {})
    unit = station_cfg.get("unit", "C")

    if not models_data:
        return None

    # Get bias-corrected forecasts
    forecasts: Dict[str, float] = {}
    for model_name, forecast in models_data.items():
        if hasattr(forecast, "bias_corrected_c"):
            temp = forecast.bias_corrected_c if unit == "C" else forecast.bias_corrected_f
        elif isinstance(forecast, dict):
            temp = forecast.get("bias_corrected_c", 0) if unit == "C" else forecast.get("bias_corrected_f", 0)
        else:
            continue
        if temp:
            forecasts[model_name] = temp

    if len(forecasts) < 2:
        return None

    # Get weights
    weights = {}
    for model_name, forecast in models_data.items():
        if hasattr(forecast, "weight"):
            weights[model_name] = forecast.weight
        elif isinstance(forecast, dict):
            weights[model_name] = forecast.get("weight", 0.25)
        else:
            weights[model_name] = 0.25

    # Weighted consensus
    consensus = models_mod.weighted_consensus(forecasts, weights)

    # Find the bin containing the consensus
    target_bin = None
    agreeing_models = 0

    for mbin in bins:
        b = mbin.bin if hasattr(mbin, "bin") else mbin
        low = b.low if hasattr(b, "low") else b.get("low", 0)
        high_val = b.high if hasattr(b, "high") else b.get("high", 0)
        yes_price = mbin.yes_price if hasattr(mbin, "yes_price") else mbin.get("yes_price", 0)

        if low <= consensus < high_val:
            target_bin = mbin
            # Count models agreeing on this bin
            for model_name, temp in forecasts.items():
                if low <= temp < high_val:
                    agreeing_models += 1
            break

    if not target_bin:
        return None

    yes_price = target_bin.yes_price if hasattr(target_bin, "yes_price") else target_bin.get("yes_price", 0)
    if yes_price > rules.get("max_entry_price", 0.15):
        return None

    if agreeing_models < rules.get("min_model_agreement", 2):
        return None

    # Model probability — use normal distribution per model
    # Each model has a forecast. Assume MAE ~2 degrees as std dev.
    # P(bin) for each model = fraction of normal dist that falls in the bin.
    import math
    model_prob = 0.0
    total_weight = sum(weights.get(m, 0.25) for m in forecasts)
    bin_low_val = target_bin.bin.low if hasattr(target_bin, "bin") else 0
    bin_high_val = target_bin.bin.high if hasattr(target_bin, "bin") else 100
    
    for m, t in forecasts.items():
        w = weights.get(m, 0.25) / max(0.01, total_weight)
        std_dev = 2.0  # assumed MAE as std dev (will improve with dynamic weighting)
        # P(bin_low <= temp < bin_high) under Normal(forecast, std_dev)
        z_low = (bin_low_val - t) / std_dev
        z_high = (bin_high_val - t) / std_dev
        # Use error function approximation for CDF
        def _norm_cdf(z):
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        p_model = _norm_cdf(z_high) - _norm_cdf(z_low)
        model_prob += w * p_model
    
    model_prob = min(0.60, max(0.0, model_prob))

    edge = model_prob - yes_price
    min_edge = rules.get("min_edge_pct", 15) / 100.0
    if edge < min_edge:
        return None

    bin_label = target_bin.bin.label if hasattr(target_bin, "bin") else target_bin.get("bin", {}).get("label", "")
    market_id = target_bin.market_id if hasattr(target_bin, "market_id") else target_bin.get("market_id", "")
    poly_url = target_bin.polymarket_url if hasattr(target_bin, "polymarket_url") else target_bin.get("polymarket_url", "")

    # --- Confidence Score ---
    score = 0
    components = {}

    # Model agreement (max 30)
    pts_agree = min(30, agreeing_models * 10)
    score += pts_agree
    components["agreement"] = pts_agree

    # Edge size (max 25)
    pts_edge = min(25, int(edge * 100))
    score += pts_edge
    components["edge"] = pts_edge

    # Model trust (max 20) — based on momentum divergence
    pts_trust = 15  # default mid-range
    components["trust"] = pts_trust
    score += pts_trust

    # Bias data confidence (max 15)
    pts_bias = 8  # default
    components["bias_conf"] = pts_bias
    score += pts_bias

    # Book depth (max 10) — placeholder
    pts_book = 5
    components["book_depth"] = pts_book
    score += pts_book

    score = max(0, min(100, score))

    # EV
    est_shares = 10
    profit_if_win = est_shares * (1.0 - yes_price)
    loss_if_lose = est_shares * yes_price
    ev = model_prob * profit_if_win - (1 - model_prob) * loss_if_lose

    # Model summary
    model_lines = " | ".join(
        f"{m.upper()} {t:.0f}°{unit}" for m, t in forecasts.items()
    )

    return Signal(
        trade_type="forecast_yes",
        station=station,
        city=city,
        target_date=target_date_val,
        side="YES",
        bin_label=bin_label,
        entry_price=yes_price,
        confidence_score=score,
        confidence_components=components,
        ev=round(ev, 2),
        win_probability=round(model_prob, 3),
        profit_if_win=round(profit_if_win, 2),
        loss_if_lose=round(loss_if_lose, 2),
        market_id=market_id,
        polymarket_url=poly_url,
        model_summary=model_lines,
        metar_summary=f"Consensus: {consensus:.1f}°{unit}",
    )


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------
def check_ladder(station: str, city: str, target_date_val: date,
                  models_data: dict, bins: list,
                  trade_rules: dict, station_cfg: dict) -> Optional[Signal]:
    """Check for ladder opportunity across multiple bins."""
    rules = trade_rules.get("ladder", {})
    unit = station_cfg.get("unit", "C")

    if not models_data:
        return None

    # Get forecasts
    forecasts: Dict[str, float] = {}
    for model_name, forecast in models_data.items():
        if hasattr(forecast, "bias_corrected_c"):
            temp = forecast.bias_corrected_c if unit == "C" else forecast.bias_corrected_f
        elif isinstance(forecast, dict):
            temp = forecast.get("bias_corrected_c", 0) if unit == "C" else forecast.get("bias_corrected_f", 0)
        else:
            continue
        if temp:
            forecasts[model_name] = temp

    if len(forecasts) < 2:
        return None

    # Model spread
    temps = list(forecasts.values())
    spread = max(temps) - min(temps)
    min_spread = rules.get("min_model_spread_f", 3.0) if unit == "F" else rules.get("min_model_spread_c", 2.0)
    if spread < min_spread:
        return None

    # Find ladder range: from min forecast to max forecast
    min_forecast = min(temps)
    max_forecast = max(temps)

    ladder_bins = []
    for mbin in bins:
        b = mbin.bin if hasattr(mbin, "bin") else mbin
        low = b.low if hasattr(b, "low") else b.get("low", 0)
        high_val = b.high if hasattr(b, "high") else b.get("high", 0)
        yes_price = mbin.yes_price if hasattr(mbin, "yes_price") else mbin.get("yes_price", 0)
        label = b.label if hasattr(b, "label") else b.get("label", "")

        # Include bins within forecast range (+ 1 bin buffer on each side)
        if low >= min_forecast - (high_val - low) and high_val <= max_forecast + (high_val - low):
            # SKIP bins priced at 0¢ — no liquidity, impossible to trade
            if yes_price < 0.01:
                continue
            if yes_price <= rules.get("max_individual_bin_price", 0.10):
                ladder_bins.append({
                    "label": label,
                    "low": low,
                    "high": high_val,
                    "yes_price": yes_price,
                    "market_id": mbin.market_id if hasattr(mbin, "market_id") else mbin.get("market_id", ""),
                    "polymarket_url": mbin.polymarket_url if hasattr(mbin, "polymarket_url") else mbin.get("polymarket_url", ""),
                })

    if len(ladder_bins) < rules.get("min_bins", 3):
        return None

    # Total cost vs range probability
    total_cost = sum(b["yes_price"] for b in ladder_bins)
    # Estimate range probability (simplified: sum of bin probabilities)
    weights = {}
    for model_name, forecast in models_data.items():
        if hasattr(forecast, "weight"):
            weights[model_name] = forecast.weight
        else:
            weights[model_name] = 0.25

    # Simple range probability estimate
    range_prob = min(0.80, len(ladder_bins) * 0.12)  # rough estimate
    cost_ratio = total_cost / max(0.01, range_prob)

    if cost_ratio > rules.get("max_total_cost_vs_prob", 0.40):
        return None

    # --- Confidence Score ---
    score = 0
    components = {}

    # Cost vs probability (max 35)
    pts_cost = int((1 - min(1, cost_ratio)) * 35)
    score += pts_cost
    components["cost_ratio"] = pts_cost

    # Model diversity (max 25)
    unique_bins = len(set(
        next(
            (b["label"] for b in ladder_bins
             if b["low"] <= t < b["high"]),
            ""
        )
        for t in forecasts.values()
    ))
    pts_div = min(25, unique_bins * 8)
    score += pts_div
    components["diversity"] = pts_div

    # Historical success (max 20)
    pts_hist = 10  # neutral default
    score += pts_hist
    components["history"] = pts_hist

    # Book depth (max 20) — placeholder
    pts_book = 10
    score += pts_book
    components["book_depth"] = pts_book

    score = max(0, min(100, score))

    # EV — based on actual $5 position, not per-share
    # If any bin wins, you get $1/share payout on that bin's shares
    # Total cost is spread across bins. If one bin wins, payout = (cost_on_that_bin / price) * $1
    # Simplified: avg payout if win = position_size / avg_price, but capped by real sizing
    avg_price = total_cost / max(1, len(ladder_bins))
    avg_price = max(0.01, avg_price)  # floor to prevent division explosion
    est_position = 5.0  # rough $5 position — allocator will finalize
    est_shares_per_bin = est_position / len(ladder_bins) / avg_price
    win_payout = est_shares_per_bin * 1.0  # $1 per share if the bin wins
    ev = range_prob * win_payout - est_position

    model_lines = " | ".join(f"{m.upper()} {t:.0f}°{unit}" for m, t in forecasts.items())

    return Signal(
        trade_type="ladder",
        station=station,
        city=city,
        target_date=target_date_val,
        side="YES",
        bin_label=f"Ladder: {len(ladder_bins)} bins",
        bins=ladder_bins,
        entry_price=max(0.01, total_cost / max(1, len(ladder_bins))),  # avg, floored
        confidence_score=score,
        confidence_components=components,
        ev=round(ev, 2),
        win_probability=round(range_prob, 3),
        profit_if_win=round(win_payout, 2),
        loss_if_lose=round(est_position, 2),
        model_summary=f"{model_lines} | Spread: {spread:.1f}°{unit}",
        metar_summary=f"Range: {ladder_bins[0]['label']} to {ladder_bins[-1]['label']}",
    )
