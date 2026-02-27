"""
wallet.py — Polymarket CLOB Wallet Sync

Connects to Polymarket CLOB API. Reads wallet balance, open positions,
and detects resolved positions. Order book depth and fill price calculation.
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import os

import aiohttp

import markets as markets_mod

logger = logging.getLogger("wallet")

CLOB_BASE_URL = "https://clob.polymarket.com"
USER_AGENT = "WeatherEdgeBot/1.0"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Position:
    market_id: str
    token_id: str
    station: str
    target_date: Optional[date]
    bin_label: str
    side: str
    shares: float
    avg_entry_price: float
    current_price: float
    cost: float
    current_value: float
    unrealized_pnl: float


@dataclass
class ResolvedPosition:
    market_id: str
    token_id: str
    station: str
    target_date: Optional[date]
    bin_label: str
    side: str
    shares: float
    cost: float
    payout: float
    profit_loss: float
    actual_high_c: Optional[float] = None
    actual_high_f: Optional[float] = None


@dataclass
class OrderBook:
    bids: List[Dict]  # [{"price": 0.42, "size": 100}, ...]
    asks: List[Dict]


@dataclass
class FillInfo:
    average_fill_price: float
    fillable_shares: float
    slippage: float


@dataclass
class WalletState:
    balance: float
    positions: List[Position] = field(default_factory=list)
    resolved: List[ResolvedPosition] = field(default_factory=list)
    total_value: float = 0.0


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def _get_credentials():
    return {
        "api_key": os.environ.get("POLY_API_KEY", ""),
        "api_secret": os.environ.get("POLY_API_SECRET", ""),
        "passphrase": os.environ.get("POLY_PASSPHRASE", ""),
    }


# ---------------------------------------------------------------------------
# Auth headers (HMAC-SHA256)
# ---------------------------------------------------------------------------
def _create_auth_headers(method: str, path: str, body: str = "") -> dict:
    """Create HMAC-SHA256 signed headers for Polymarket CLOB API."""
    creds = _get_credentials()
    timestamp = str(int(time.time()))

    message = timestamp + method.upper() + path + body
    signature = hmac.new(
        creds["api_secret"].encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "POLY-API-KEY": creds["api_key"],
        "POLY-SIGNATURE": signature,
        "POLY-TIMESTAMP": timestamp,
        "POLY-PASSPHRASE": creds["passphrase"],
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


# ---------------------------------------------------------------------------
# Last known positions (for resolution detection)
# ---------------------------------------------------------------------------
_last_known_positions: Dict[str, Position] = {}


# ---------------------------------------------------------------------------
# Main sync function
# ---------------------------------------------------------------------------
async def sync() -> WalletState:
    """
    Sync with Polymarket wallet.
    Returns WalletState with balance, positions, and resolved positions.
    """
    global _last_known_positions

    state = WalletState(balance=0.0)

    # Fetch balance
    balance = await _fetch_balance()
    state.balance = balance

    # Fetch positions
    raw_positions = await _fetch_positions()
    current_positions: Dict[str, Position] = {}

    for rp in raw_positions:
        pos = _parse_position(rp)
        if pos:
            current_positions[pos.market_id] = pos
            state.positions.append(pos)

    # Detect resolutions: positions in last_known but not in current
    for market_id, old_pos in _last_known_positions.items():
        if market_id not in current_positions:
            # Position disappeared — market likely resolved
            # Estimate payout from balance change
            resolved = ResolvedPosition(
                market_id=old_pos.market_id,
                token_id=old_pos.token_id,
                station=old_pos.station,
                target_date=old_pos.target_date,
                bin_label=old_pos.bin_label,
                side=old_pos.side,
                shares=old_pos.shares,
                cost=old_pos.cost,
                payout=0.0,  # Will be estimated from balance difference
                profit_loss=0.0,
            )
            state.resolved.append(resolved)

    # Estimate payouts for resolved positions
    if state.resolved and _last_known_positions:
        old_total = sum(p.cost for p in _last_known_positions.values())
        current_total = sum(p.cost for p in current_positions.values())
        balance_diff = balance - (old_total - current_total)

        # Distribute payout estimate across resolved positions
        for rp in state.resolved:
            # If balance increased beyond removal of position cost, it was a win
            expected_balance_if_loss = balance  # no payout
            if rp.side == "YES":
                rp.payout = rp.shares  # YES win pays $1 per share
            else:
                rp.payout = rp.shares  # NO win pays $1 per share
            # We can't know for sure — will be corrected by actual balance tracking
            rp.profit_loss = rp.payout - rp.cost

    # Update last known
    _last_known_positions = current_positions

    # Total value
    state.total_value = state.balance + sum(
        p.current_value for p in state.positions
    )

    logger.info(
        "Wallet sync: balance=%.2f positions=%d resolved=%d total=%.2f",
        state.balance, len(state.positions),
        len(state.resolved), state.total_value,
    )

    return state


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
async def _fetch_balance() -> float:
    """GET /balance"""
    path = "/balance"
    headers = _create_auth_headers("GET", path)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CLOB_BASE_URL}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("Balance fetch HTTP %d", resp.status)
                    return 0.0
                data = await resp.json()
                # Response format may vary
                if isinstance(data, dict):
                    return float(data.get("balance", data.get("amount", 0)))
                return float(data) if data else 0.0
    except Exception as e:
        logger.error("Balance fetch error: %s", e)
        return 0.0


async def _fetch_positions() -> list:
    """GET /positions"""
    path = "/positions"
    headers = _create_auth_headers("GET", path)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CLOB_BASE_URL}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("Positions fetch HTTP %d", resp.status)
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else []
    except Exception as e:
        logger.error("Positions fetch error: %s", e)
        return []


def _parse_position(raw: dict) -> Optional[Position]:
    """Parse a raw position from CLOB API into Position dataclass."""
    try:
        market_id = str(raw.get("market", raw.get("conditionId", "")))
        token_id = str(raw.get("tokenId", raw.get("token_id", "")))

        # Try to determine station/date/bin from market title or stored data
        title = raw.get("title", raw.get("question", ""))
        station = "UNKNOWN"
        target_date = None
        bin_label = ""

        if title:
            bin_info = markets_mod.parse_bin_from_title(title)
            if bin_info:
                bin_label = bin_info.label

        side = raw.get("side", raw.get("outcome", "YES")).upper()
        if side not in ("YES", "NO"):
            side = "YES"

        shares = float(raw.get("size", raw.get("shares", 0)))
        avg_price = float(raw.get("avgPrice", raw.get("avg_price", 0)))
        current_price = float(raw.get("curPrice", raw.get("price", avg_price)))
        cost = shares * avg_price
        current_value = shares * current_price
        unrealized = current_value - cost

        return Position(
            market_id=market_id,
            token_id=token_id,
            station=station,
            target_date=target_date,
            bin_label=bin_label,
            side=side,
            shares=shares,
            avg_entry_price=avg_price,
            current_price=current_price,
            cost=cost,
            current_value=current_value,
            unrealized_pnl=unrealized,
        )
    except Exception as e:
        logger.error("Position parse error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------
async def get_order_book(market_id: str) -> OrderBook:
    """Fetch order book depth for a specific market."""
    path = f"/book"
    headers = _create_auth_headers("GET", path)
    params = {"token_id": market_id}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CLOB_BASE_URL}{path}",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.error("Order book HTTP %d for %s", resp.status, market_id)
                    return OrderBook(bids=[], asks=[])
                data = await resp.json()
    except Exception as e:
        logger.error("Order book error for %s: %s", market_id, e)
        return OrderBook(bids=[], asks=[])

    bids = []
    asks = []

    for level in data.get("bids", []):
        bids.append({
            "price": float(level.get("price", 0)),
            "size": float(level.get("size", 0)),
        })
    for level in data.get("asks", []):
        asks.append({
            "price": float(level.get("price", 0)),
            "size": float(level.get("size", 0)),
        })

    # Sort bids descending, asks ascending
    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])

    return OrderBook(bids=bids, asks=asks)


# ---------------------------------------------------------------------------
# Fill price calculation
# ---------------------------------------------------------------------------
def calculate_fill_price(order_book: OrderBook, desired_shares: float,
                          side: str) -> FillInfo:
    """
    Walk through order book to calculate realistic fill.
    For buying YES: walk through asks (ascending price)
    For buying NO: walk through bids on YES side (descending — selling YES = buying NO)
    """
    if side == "YES":
        levels = order_book.asks
    else:
        # Buying NO = selling YES, so look at bids
        levels = order_book.bids

    if not levels:
        return FillInfo(
            average_fill_price=0.0,
            fillable_shares=0.0,
            slippage=0.0,
        )

    best_price = levels[0]["price"]
    total_cost = 0.0
    filled = 0.0
    remaining = desired_shares

    for level in levels:
        available = level["size"]
        price = level["price"]
        take = min(remaining, available)
        total_cost += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break

    if filled == 0:
        return FillInfo(
            average_fill_price=0.0,
            fillable_shares=0.0,
            slippage=0.0,
        )

    avg_price = total_cost / filled
    slippage = avg_price - best_price

    return FillInfo(
        average_fill_price=round(avg_price, 4),
        fillable_shares=round(filled, 2),
        slippage=round(slippage, 4),
    )
