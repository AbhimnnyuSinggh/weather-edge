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

@dataclass
class Signal:
    trade_type: str
    station: str
    city: str
    target_date: date
    side: str
    bin_label: str
    bins: List[dict] = field(default_factory=list)
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
    distribution_summary: str = ""

# ---------------------------------------------------------------------------
# City Dashboard - Trade Analysis
# ---------------------------------------------------------------------------
def analyze_market(market_group: MarketGroup, probs: Dict[str, float], total_cap: float, local_hour: int,
                   predicted_high=None, models_data=None, metar_high=None, confidence="HIGH") -> tuple[Dict[str, dict], float]:
    """
    Analyze the market active bins vs calculated probabilities to generate
    specific trade recommendations for the City Dashboard.
    
    Args:
        market_group: Market bins and prices
        probs: Dict of bin probabilities from our distribution
        total_cap: Total capital available
        local_hour: Current local hour at the station
        predicted_high: Bot's predicted daily high (P1 fix)
        models_data: Dict of model forecasts (P1 fix)
        metar_high: Current METAR observed high (P3 fix)
        confidence: "HIGH", "MEDIUM", or "LOW" (P7 fix)
    
    Returns:
       Dict mapped by trade type: { type_key: { ... data ... } }
       Reserved capital used
    """
    trades = {
        "forecast_yes": {"valid": False, "label": "FORECAST YES (Growth Engine)", "skip_reason": "No bins meet edge threshold."},
        "no_tail": {"valid": False, "label": "NO TAIL (Stability Floor)", "skip_reason": "Waiting for afternoon... NO tails activate after 2 PM local when METAR confirms high is set and temp is falling."},
        "ladder": {"valid": False, "label": "LADDER (Insurance)", "skip_reason": "NO LADDER — Models agree too closely. Ladders only trigger when spread > 5°F / 3°C."},
        "lockin": {"valid": False, "label": "LOCK-IN (Bread & Butter)", "skip_reason": "Waiting for afternoon... Lock-in activates after 2 PM local when METAR shows temp falling from day's high."}
    }
    
    # Capital limits
    reserve = total_cap * 0.15
    max_deployable = total_cap - reserve
    
    if max_deployable <= 0:
        for k in trades: trades[k]["skip_reason"] = "Insufficient balance."
        return trades, 0
        
    sorted_bins = sorted(market_group.bins, key=lambda b: b.bin.low if b.bin.low is not None else -999)
    if not sorted_bins:
        return trades, 0

    # ── P7 FIX: Confidence-based thresholds ──
    # LOW confidence → require larger edge, smaller allocation, discounted EV
    if confidence == "LOW":
        yes_edge_threshold = 0.20   # Require 20% edge instead of 10%
        yes_alloc_pct = 5           # 5% allocation instead of 15%
        ev_discount = 0.5           # 50% haircut on EV
    elif confidence == "MEDIUM":
        yes_edge_threshold = 0.15
        yes_alloc_pct = 10
        ev_discount = 0.75
    else:  # HIGH
        yes_edge_threshold = 0.10
        yes_alloc_pct = 15
        ev_discount = 1.0

    # 1. FORECAST YES
    best_fc_bin = None
    best_fc_edge = 0
    for b in sorted_bins:
        if b.yes_price < 0.01: continue
        lbl = b.bin.label
        prob = probs.get(lbl, 0)
        mkt_price = b.yes_price
        edge = prob - mkt_price
        
        if edge > yes_edge_threshold and edge > best_fc_edge:
            if b.yes_price <= 0.30:
                # ── P1 FIX: Check if any model actually supports this bin ──
                if models_data and predicted_high is not None:
                    bin_low = b.bin.low if b.bin.low is not None else -999
                    bin_high = b.bin.high if b.bin.high is not None else 999
                    model_support = 0
                    near_support = 0
                    for name, fc in models_data.items():
                        if not fc: continue
                        t = getattr(fc, 'bias_corrected_f', None)
                        if t is not None:
                            if bin_low <= t < bin_high:
                                model_support += 1
                            elif (bin_low - 1) <= t < (bin_high + 1):
                                near_support += 1
                    if model_support == 0 and near_support < 2:
                        logger.info("P1: Skipping bin %s — no model supports it (support=%d, near=%d)",
                                   lbl, model_support, near_support)
                        continue  # Skip bin — no model supports it
                
                best_fc_edge = edge
                best_fc_bin = b
    
    # ── P3 FIX: Don't recommend YES on a bin the METAR floor has nearly eliminated ──
    if metar_high is not None and best_fc_bin:
        bin_high_val = best_fc_bin.bin.high if best_fc_bin.bin.high is not None else 999
        if bin_high_val <= metar_high + 1:
            logger.info("P3: Skipping bin %s — METAR floor (%.0f°F) has reached this bin",
                       best_fc_bin.bin.label, metar_high)
            trades["forecast_yes"]["skip_reason"] = f"METAR floor ({metar_high:.0f}°F) has reached this bin. High is likely above it."
            best_fc_bin = None
                
    if best_fc_bin:
        lbl = best_fc_bin.bin.label
        prob = probs.get(lbl, 0)
        mkt_price = best_fc_bin.yes_price
        
        alloc_pct = yes_alloc_pct  # P7: confidence-adjusted allocation
        alloc_amount = max_deployable * (alloc_pct / 100.0)
        shares = alloc_amount / mkt_price
        # Floor shares realistically:
        shares = float(int(shares))
        cost = shares * mkt_price
        payout = shares * 1.0
        profit = payout - cost
        ev = (prob * profit) - ((1.0 - prob) * cost)
        
        # ── P5 FIX: Apply confidence discount to EV ──
        adjusted_ev = ev * ev_discount
        
        # Guard against math errors
        if adjusted_ev <= 100:
            trades["forecast_yes"] = {
                "valid": True,
                "label": "FORECAST YES (Growth Engine)",
                "action_emoji": "✅",
                "action": "BUY YES",
                "side": "YES",
                "bin_label": lbl,
                "price": int(mkt_price * 100),
                "alloc_pct": alloc_pct,
                "alloc_amount": alloc_amount,
                "shares": shares,
                "cost": cost,
                "payout": payout,
                "profit": profit,
                "win_prob": int(prob * 100),
                "lose_prob": int((1.0 - prob) * 100),
                "ev": adjusted_ev,
                "edge": int(best_fc_edge * 100),
                "timing_advice": "Place now. Price likely rises as afternoon approaches and models confirm. Early entry = cheaper shares = bigger profit."
            }
            
    # 2. LADDER (P8 FIX: Relaxed thresholds + METAR floor awareness)
    # Look for 2 adjacent bins with combined prob > 40% and some edge
    for i in range(len(sorted_bins) - 1):
        b1 = sorted_bins[i]
        b2 = sorted_bins[i+1]
        
        lbl1, lbl2 = b1.bin.label, b2.bin.label
        p1, p2 = probs.get(lbl1, 0), probs.get(lbl2, 0)
        m1, m2 = b1.yes_price, b2.yes_price
        
        if m1 <= 0.01 or m2 <= 0.01: continue
            
        edge1 = p1 - m1
        edge2 = p2 - m2
        
        # ── P8 FIX: Relaxed conditions — combined > 40%, at least one has 3% edge ──
        if (p1 + p2) > 0.40 and (edge1 > 0.03 or edge2 > 0.03) and (m1 + m2) < 0.50:
            # P3: Skip if METAR floor has passed the lower bin
            if metar_high is not None:
                b1_high = b1.bin.high if b1.bin.high is not None else 999
                if b1_high <= metar_high:
                    continue  # METAR has already passed this bin
            
            alloc_pct = 10
            alloc_amount = max_deployable * (alloc_pct / 100.0)
            alloc_per_rung = alloc_amount / 2.0
            
            shares1 = float(int(alloc_per_rung / m1))
            cost1 = shares1 * m1
            ev1 = (p1 * shares1 * 1.0) - cost1
            
            shares2 = float(int(alloc_per_rung / m2))
            cost2 = shares2 * m2
            ev2 = (p2 * shares2 * 1.0) - cost2
            
            total_ev = (ev1 + ev2) * ev_discount  # P5: confidence discount
            total_cost = cost1 + cost2
            avg_win_prob = (p1 + p2)
            
            if total_ev <= 100:
                trades["ladder"] = {
                    "valid": True,
                    "label": "LADDER (Insurance)",
                    "action_emoji": "🪜",
                    "action": "BUY YES",
                    "side": "YES",
                    "bin_label": f"{lbl1} & {lbl2}",
                    "price": f"{int(m1*100)}¢ & {int(m2*100)}",
                    "alloc_pct": alloc_pct,
                    "alloc_amount": alloc_amount,
                    "shares": shares1 + shares2,
                    "cost": total_cost,
                    "payout": (shares1 + shares2) * 1.0,
                    "profit": ((shares1 + shares2) * 1.0) - total_cost,
                    "win_prob": int(avg_win_prob * 100),
                    "lose_prob": int((1.0 - avg_win_prob) * 100),
                    "ev": total_ev,
                    "edge": int(((edge1+edge2)/2) * 100),
                    "timing_advice": "Spread trades mitigate model disagreement risk. Both bins offer +EV."
                }
            break

    # Common variable used by both LOCK-IN and NO TAIL
    highest_prob_bin_idx = -1
    if sorted_bins:
        highest_prob_bin_idx = max(range(len(sorted_bins)), key=lambda i: probs.get(sorted_bins[i].bin.label, 0))

    # 3. LOCK-IN YES
    # Just check if highest prob is >85%
    if highest_prob_bin_idx >= 0:
        b = sorted_bins[highest_prob_bin_idx]
        lbl = b.bin.label
        prob = probs.get(lbl, 0)
        mkt_price = b.yes_price
        
        if prob > 0.85 and mkt_price < 0.80 and mkt_price >= 0.01:
            edge = prob - mkt_price
            if edge > 0.05:
                alloc_pct = 25
                alloc_amount = max_deployable * (alloc_pct / 100.0)
                shares = float(int(alloc_amount / mkt_price))
                cost = shares * mkt_price
                payout = shares * 1.0
                profit = payout - cost
                ev = (prob * profit) - ((1.0 - prob) * cost)
                adjusted_ev = ev * ev_discount  # P5
                
                if adjusted_ev <= 100:
                    trades["lockin"] = {
                        "valid": True,
                        "label": "LOCK-IN YES (Bread & Butter)",
                        "action_emoji": "🔒",
                        "action": "BUY YES",
                        "side": "YES",
                        "bin_label": lbl,
                        "price": int(mkt_price * 100),
                        "alloc_pct": alloc_pct,
                        "alloc_amount": alloc_amount,
                        "shares": shares,
                        "cost": cost,
                        "payout": payout,
                        "profit": profit,
                        "win_prob": int(prob * 100),
                        "lose_prob": int((1.0 - prob) * 100),
                        "ev": adjusted_ev,
                        "edge": int(edge * 100),
                        "timing_advice": "Wait for afternoon METAR to confirm peak before sizing up."
                    }

    # 4. NO TAIL
    if highest_prob_bin_idx >= 0 and local_hour >= 14:
        for i in range(highest_prob_bin_idx + 2, len(sorted_bins)):
            b = sorted_bins[i]
            lbl = b.bin.label
            prob = probs.get(lbl, 0)
            mkt_price = b.yes_price
            
            if prob < 0.05 and mkt_price > 0.10:
                no_price = 1.0 - mkt_price
                no_prob = 1.0 - prob
                edge = no_prob - no_price
                if edge > 0.08 and no_price > 0.01:
                    alloc_pct = 35
                    alloc_amount = max_deployable * (alloc_pct / 100.0)
                    shares = alloc_amount / no_price
                    shares = float(int(shares))
                    cost = shares * no_price
                    profit = shares * mkt_price
                    ev = (no_prob * profit) - ((1.0 - no_prob) * cost)
                    if ev <= 100:
                        trades["no_tail"] = {
                            "valid": True,
                            "label": "NO TAIL (Stability Floor)",
                            "action_emoji": "⛔",
                            "action": "BUY NO",
                            "side": "NO",
                            "bin_label": lbl,
                            "price": int(mkt_price * 100),
                            "alloc_pct": alloc_pct,
                            "alloc_amount": alloc_amount,
                            "shares": shares,
                            "cost": cost,
                            "payout": shares * 1.0, 
                            "profit": profit,
                            "win_prob": int(no_prob * 100),
                            "lose_prob": int((1.0 - no_prob) * 100),
                            "ev": ev,
                            "edge": int(edge * 100),
                            "timing_advice": "Check METAR explicitly. NO tails trigger best when temp has provably peaked."
                        }
                    break
    
    # Calculate deployed cap
    deployed = sum(t["cost"] for t in trades.values() if t.get("valid"))
    
    return trades, deployed
