"""
distribution.py — Bin Probability Distribution Engine

Core engine for calculating probability distributions across market bins.
Uses normal distributions (N(forecast, MAE)) per model, then weight-averages.
With ensemble data: counts members per bin for non-parametric distribution.
"""

import logging
import logging
import math
from datetime import date
from typing import Dict, List, Optional, Tuple

from models import ModelForecast

logger = logging.getLogger("distribution")


def calculate_bin_probabilities(models_data: Dict[str, ModelForecast], bins: list, unit: str = "C") -> Dict[str, float]:
    """
    For each bin, calculate the probability that the actual temp falls in it.
    Uses normal distribution per model, weighted by inverse-MAE.
    """
    DEFAULT_MAE = {
        "gfs": 1.8, "ecmwf": 1.5, "icon": 2.0, "gem": 2.2, "jma": 2.0,
        "nws": 1.5, "noaa_mos": 1.3, "visual_crossing": 2.0,
    }

    def norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # Collect forecasts and weights
    forecasts = []  # list of (temp, weight, mae)
    for name, forecast in models_data.items():
        temp = forecast.bias_corrected_c if unit == "C" else forecast.bias_corrected_f
        if temp is None or temp == 0:
            continue
        mae = DEFAULT_MAE.get(name, 2.0)
        weight = 1.0 / max(0.5, mae)
        forecasts.append((temp, weight, mae))

    if not forecasts:
        return {}

    total_weight = sum(w for _, w, _ in forecasts)

    # For each bin, sum weighted probabilities from each model
    bin_probs = {}
    for mbin in bins:
        bin_low = mbin.bin.low if hasattr(mbin, 'bin') else mbin.get('low', 0)
        bin_high = mbin.bin.high if hasattr(mbin, 'bin') else mbin.get('high', 0)
        bin_label = mbin.bin.label if hasattr(mbin, 'bin') else mbin.get('label', '')

        prob = 0.0
        for temp, weight, mae in forecasts:
            std_dev = mae  # Use MAE as standard deviation
            
            # Handle open-ended bins
            if bin_low is None and bin_high is not None:
                z_high = (bin_high - temp) / std_dev
                model_prob = norm_cdf(z_high)
            elif bin_high is None and bin_low is not None:
                z_low = (bin_low - temp) / std_dev
                model_prob = 1.0 - norm_cdf(z_low)
            elif bin_low is not None and bin_high is not None:
                z_low = (bin_low - temp) / std_dev
                z_high = (bin_high - temp) / std_dev
                model_prob = norm_cdf(z_high) - norm_cdf(z_low)
            else:
                model_prob = 0.0
                
            prob += (weight / total_weight) * model_prob

        bin_probs[bin_label] = round(prob, 4)

    return bin_probs


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
