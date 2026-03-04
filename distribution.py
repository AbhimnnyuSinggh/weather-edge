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


def calculate_bin_probabilities(bins, predicted_high, sigma, metar_high=None):
    """
    Strict PDF: probability mass only for bins correctly enclosing the predicted mean.
    No cumulative stacking — mutually exclusive.
    Uses the dashboard's `predicted_high` and `sigma` to generate a flawless bell curve.
    """
    def norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        
    probs = {}
    
    # We use the finely tuned predicted_high as our mean, and sigma as our std_dev
    mean = predicted_high
    sigma = max(0.5, sigma) # Fallback to prevent divide-by-zero
    
    for mbin in bins:
        bin_low = _get_bin_low(mbin)
        bin_high = _get_bin_high(mbin)
        bin_label = _get_bin_label(mbin)
        
        if not bin_label:
            continue
            
        z_low = (bin_low - mean) / sigma if bin_low is not None else -10
        z_high = (bin_high - mean) / sigma if bin_high is not None else 10
        
        # Integrate PDF over the mutually exclusive bin boundaries
        prob = norm_cdf(z_high) - norm_cdf(z_low)
        
        # METAR floor logic: theoretically impossible to be below current day's reality
        if metar_high is not None and bin_high is not None and bin_high <= metar_high:
            prob = 0.0
            
        probs[bin_label] = max(0.0, prob)
        
    # Normalize to sum to 1.0 (strict mutual exclusivity)
    total = sum(probs.values())
    if total > 0:
        for label in probs:
            probs[label] = round(probs[label] / total, 4)
    else:
        for label in probs:
            probs[label] = 0.0
            
    return probs

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
