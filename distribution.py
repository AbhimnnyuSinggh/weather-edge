"""
distribution.py — Bin Probability Distribution Engine

Core engine for calculating probability distributions across market bins.
Uses normal distributions (N(forecast, MAE)) per model, then weight-averages.
With ensemble data: counts members per bin for non-parametric distribution.
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

import tracker
from models import DEFAULT_MAE, ModelForecast

logger = logging.getLogger("distribution")


def build_bin_distribution(
    forecasts: Dict[str, ModelForecast],
    bins: list,
    unit: str = "F",
    ensemble_members: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Build probability distribution across bins.

    For each model:
      1. Get bias-corrected forecast temp
      2. Get model MAE (from tracker or DEFAULT_MAE)
      3. Create N(forecast, MAE)
      4. For each bin [low, high): P(bin) = CDF(high) - CDF(low)
    Then weight-average across models.

    With ensemble data: count how many of 31 members fall in each bin.

    Args:
        forecasts: {model_name: ModelForecast}
        bins: list of MarketBin objects with .bin.low, .bin.high, .bin.label
        unit: "F" or "C"
        ensemble_members: Optional list of 31 GFS member temps

    Returns:
        {bin_label: probability} — sums to ~1.0
    """
    if not bins or not forecasts:
        return {}

    # Extract bin edges
    bin_edges = []
    for b in bins:
        low = b.bin.low if hasattr(b, 'bin') else b.get("bin_low")
        high = b.bin.high if hasattr(b, 'bin') else b.get("bin_high")
        label = b.bin.label if hasattr(b, 'bin') else b.get("bin_label", "")
        bin_edges.append((low, high, label))

    if not bin_edges:
        return {}

    # --- Ensemble path: non-parametric ---
    if ensemble_members and len(ensemble_members) >= 10:
        probs = _ensemble_distribution(ensemble_members, bin_edges)
        # Blend with model-based if we have both
        model_probs = _model_distribution(forecasts, bin_edges, unit)
        if model_probs:
            # 60% ensemble, 40% model-based
            for label in probs:
                if label in model_probs:
                    probs[label] = 0.6 * probs[label] + 0.4 * model_probs[label]
        return probs

    # --- Standard path: normal distributions ---
    return _model_distribution(forecasts, bin_edges, unit)


def _model_distribution(
    forecasts: Dict[str, ModelForecast],
    bin_edges: List[Tuple],
    unit: str,
) -> Dict[str, float]:
    """Normal distribution approach per model, then weight-average."""
    if not forecasts:
        return {}

    # Collect per-model bin probabilities
    all_model_probs: Dict[str, Dict[str, float]] = {}
    model_weights: Dict[str, float] = {}

    for model_name, forecast in forecasts.items():
        # Get forecast temp in the right unit
        if unit.upper() == "F":
            temp = forecast.bias_corrected_f
        else:
            temp = forecast.bias_corrected_c

        # Get MAE for this model (from DB or default)
        mae = DEFAULT_MAE.get(model_name, 2.0)
        # Convert MAE to same unit as bins
        if unit.upper() == "F" and model_name in DEFAULT_MAE:
            mae = mae * 9.0 / 5.0  # Convert °C MAE to °F

        # Build normal distribution
        sigma = max(0.5, mae)
        probs = {}
        for low, high, label in bin_edges:
            if low is None and high is not None:
                # Edge bin: "X or below"
                p = norm.cdf(high, loc=temp, scale=sigma)
            elif high is None and low is not None:
                # Edge bin: "X or above"
                p = 1.0 - norm.cdf(low, loc=temp, scale=sigma)
            elif low is not None and high is not None:
                p = norm.cdf(high, loc=temp, scale=sigma) - norm.cdf(low, loc=temp, scale=sigma)
            else:
                p = 0.0
            probs[label] = max(0.001, p)  # Floor at 0.1%

        all_model_probs[model_name] = probs
        model_weights[model_name] = forecast.weight

    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight == 0:
        total_weight = len(model_weights)
        model_weights = {m: 1.0 for m in model_weights}

    # Weight-average across models
    combined: Dict[str, float] = {}
    for low, high, label in bin_edges:
        weighted_sum = 0.0
        for model_name, probs in all_model_probs.items():
            w = model_weights[model_name] / total_weight
            weighted_sum += w * probs.get(label, 0.001)
        combined[label] = weighted_sum

    # Normalize to sum to ~1.0
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}

    return combined


def _ensemble_distribution(
    members: List[float],
    bin_edges: List[Tuple],
) -> Dict[str, float]:
    """Count ensemble members per bin for non-parametric distribution."""
    n = len(members)
    probs = {}

    for low, high, label in bin_edges:
        count = 0
        for temp in members:
            if low is None and high is not None:
                if temp <= high:
                    count += 1
            elif high is None and low is not None:
                if temp >= low:
                    count += 1
            elif low is not None and high is not None:
                if low <= temp < high:
                    count += 1
        probs[label] = max(0.001, count / n)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def format_distribution_text(
    probs: Dict[str, float],
    bin_prices: Dict[str, float],
    unit: str = "F",
) -> str:
    """
    Format probability distribution for Telegram alert.
    Shows: Distribution, Market prices, and Edge per bin.
    """
    if not probs:
        return ""

    # Find the bin with highest probability
    best_bin = max(probs, key=probs.get)

    dist_parts = []
    price_parts = []
    edge_parts = []

    for label in probs:
        prob = probs[label]
        price = bin_prices.get(label, 0)
        edge = prob - price

        marker = " ←" if label == best_bin else ""
        dist_parts.append(f"{label}({prob*100:.0f}%){marker}")
        price_parts.append(f"{label}({price*100:.0f}¢)")

        if abs(edge) > 0.05:  # Only show significant edges
            sign = "+" if edge > 0 else ""
            best_marker = " ← BEST" if label == best_bin else ""
            edge_parts.append(f"{label}({sign}{edge*100:.0f}%){best_marker}")

    lines = [
        f"Distribution: {' | '.join(dist_parts)}",
        f"Market: {' | '.join(price_parts)}",
    ]
    if edge_parts:
        lines.append(f"Edge: {' | '.join(edge_parts)}")

    return "\n".join(lines)
