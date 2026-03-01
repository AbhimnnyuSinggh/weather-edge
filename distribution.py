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


def calculate_bin_probabilities(models_data, bins, ensemble_members=None, metar_high=None, unit="C"):
    """
    Two methods combined:
    Method A: Normal distribution per model (each centered on ITS OWN forecast)
    Method B: Ensemble counting (if 31 GFS members available)
    Final = 60% ensemble + 40% model curves (if ensemble available)
           = 100% model curves (if no ensemble)
    """
    
    DEFAULT_MAE = {
        "gfs": 1.8, "ecmwf": 1.5, "icon": 2.0, "gem": 2.2, "jma": 2.0,
        "nws": 1.5, "noaa_mos": 1.3, "visual_crossing": 2.0,
    }
    
    def norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    
    # ── METHOD A: Normal distribution per model ──
    forecasts = []
    for name, forecast in models_data.items():
        try:
            temp = forecast.bias_corrected_c if unit == "C" else forecast.bias_corrected_f
            if temp is None:
                continue
            mae = DEFAULT_MAE.get(name, 2.0)
            weight = 1.0 / max(0.5, mae)
            forecasts.append((temp, weight, mae))
        except AttributeError:
            continue
    
    model_probs = {}
    if forecasts:
        total_weight = sum(w for _, w, _ in forecasts)
        
        for mbin in bins:
            bin_low = _get_bin_low(mbin)
            bin_high = _get_bin_high(mbin)
            bin_label = _get_bin_label(mbin)
            
            prob = 0.0
            for temp, weight, mae in forecasts:
                # CRITICAL FIX: Use the FULL MAE as std_dev, not mae * 0.35
                std_dev = max(1.0, mae)
                
                # CRITICAL FIX: Center on THIS MODEL'S forecast, NOT predicted_high
                center = temp
                
                z_low = (bin_low - center) / std_dev if bin_low is not None else -10
                z_high = (bin_high - center) / std_dev if bin_high is not None else 10
                model_prob = norm_cdf(z_high) - norm_cdf(z_low)
                
                prob += model_prob * (weight / total_weight)
            
            if bin_label:
                model_probs[bin_label] = prob
    
    # ── METHOD B: Ensemble counting (31 GFS members) ──
    ensemble_probs = {}
    if ensemble_members and len(ensemble_members) >= 20:
        for mbin in bins:
            bin_low = _get_bin_low(mbin)
            bin_high = _get_bin_high(mbin)
            bin_label = _get_bin_label(mbin)
            
            count = 0
            for member_temp in ensemble_members:
                if member_temp is None:
                    continue
                in_bin = True
                if bin_low is not None and member_temp < bin_low:
                    in_bin = False
                if bin_high is not None and member_temp >= bin_high:
                    in_bin = False
                if in_bin:
                    count += 1
            
            if bin_label:
                ensemble_probs[bin_label] = count / len([m for m in ensemble_members if m is not None])
    
    # ── COMBINE ──
    final_probs = {}
    all_labels = set(list(model_probs.keys()) + list(ensemble_probs.keys()))
    
    for label in all_labels:
        mp = model_probs.get(label, 0)
        ep = ensemble_probs.get(label, 0)
        
        if ensemble_probs:
            # 60% ensemble + 40% model curves
            final_probs[label] = 0.6 * ep + 0.4 * mp
        else:
            final_probs[label] = mp
    
    # METAR floor: zero out bins below observed high
    if metar_high is not None:
        for mbin in bins:
            bin_high = _get_bin_high(mbin)
            bin_label = _get_bin_label(mbin)
            if bin_label and bin_high is not None and bin_high < metar_high:
                final_probs[bin_label] = 0.0
    
    # Normalize to sum to 1.0
    total = sum(final_probs.values())
    if total > 0:
        for label in final_probs:
            final_probs[label] = round(final_probs[label] / total, 4)
    else:
        for label in final_probs:
            final_probs[label] = 0.0
            
    return final_probs

# Helper functions to extract bin properties
def _get_bin_low(mbin):
    if hasattr(mbin, 'bin'):
        return getattr(mbin.bin, 'low', None)
    return mbin.get('low', None)

def _get_bin_high(mbin):
    if hasattr(mbin, 'bin'):
        return getattr(mbin.bin, 'high', None)
    return mbin.get('high', None)

def _get_bin_label(mbin):
    if hasattr(mbin, 'bin'):
        return getattr(mbin.bin, 'label', '')
    return mbin.get('label', '')

def format_distribution_text(probs: Dict[str, float], bin_prices: Dict[str, float], unit: str = "F") -> str:
    """Format simple distribution text (retained for generic output)."""
    if not probs:
        return ""
    
    best_bin = max(probs, key=probs.get) if probs else ""
    dist_parts = []
    
    for label, prob in probs.items():
        marker = " ←" if label == best_bin else ""
        dist_parts.append(f"{label}({prob*100:.0f}%){marker}")
        
    return f"Distribution: {' | '.join(dist_parts)}"
