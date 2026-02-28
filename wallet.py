"""
wallet.py — Polymarket Wallet via Data API

Fetches REAL wallet state from Polymarket Data API:
- Portfolio value (total holdings)
- Open positions (size, avgPrice, curPrice, cashPnl)
- Closed positions (realizedPnl for auto-resolution)

Falls back to internal DB tracking if API is unreachable.
Order book queries via public CLOB endpoint (no auth needed).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("wallet")

USER_AGENT = "WeatherEdgeBot/1.0"
DATA_API_BASE = "https://data-api.polymarket.com"


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
    event_slug: str = ""
    redeemable: bool = False


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
    bids: List[Dict]
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
    source: str = "api"  # "api" or "db_fallback"


# ---------------------------------------------------------------------------
# Internal capital tracking (fallback)
# ---------------------------------------------------------------------------
_config_capital: float = 100.0
_manual_capital: Optional[float] = None


def set_config_capital(amount: float):
    global _config_capital
    _config_capital = amount
    logger.info("Config capital set to %.2f", amount)


def set_manual_capital(amount: float):
    global _manual_capital
    _manual_capital = amount
    logger.info("Manual capital override to %.2f", amount)


def _wallet_address() -> Optional[str]:
    return os.environ.get("POLY_WALLET_ADDRESS")


# ---------------------------------------------------------------------------
# Polymarket Data API — fetch portfolio value
# ---------------------------------------------------------------------------
async def fetch_portfolio_value() -> Optional[float]:
    """
    GET /value?user={address}
    Returns total portfolio value (positions + cash).
    """
    addr = _wallet_address()
    if not addr:
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DATA_API_BASE}/value",
                params={"user": addr},
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Data API /value HTTP %d", resp.status)
                    return None
                data = await resp.json()

        # Response is a list with one item: [{"user": "0x...", "value": 15.57}]
        if isinstance(data, list) and data:
            return float(data[0].get("value", 0))
        return None

    except Exception as e:
        logger.error("Data API /value error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Polymarket Data API — fetch open positions
# ---------------------------------------------------------------------------
async def fetch_positions() -> List[Position]:
    """
    GET /positions?user={address}
    Returns all open positions with size, avgPrice, curPrice, etc.
    """
    addr = _wallet_address()
    if not addr:
        return []

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DATA_API_BASE}/positions",
                params={"user": addr},
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Data API /positions HTTP %d", resp.status)
                    return []
                data = await resp.json()

        if not isinstance(data, list):
            return []

        positions = []
        for p in data:
            event_slug = p.get("eventSlug", "")
            title = p.get("title", "")

            # Filter: only weather temperature positions
            if "temperature" not in title.lower() and "temperature" not in event_slug:
                continue

            # Extract station from eventSlug
            station = _extract_station_from_slug(event_slug)

            # Extract target date from endDate
            target_date = None
            end_date_str = p.get("endDate")
            if end_date_str:
                try:
                    target_date = date.fromisoformat(end_date_str[:10])
                except ValueError:
                    pass

            # Extract bin label from title
            bin_label = _extract_bin_label(title)

            size = float(p.get("size") or 0)
            avg_price = float(p.get("avgPrice") or 0)
            cur_price = float(p.get("curPrice") or 0)
            initial_value = float(p.get("initialValue") or 0)
            current_value = float(p.get("currentValue") or 0)
            cash_pnl = float(p.get("cashPnl") or 0)
            redeemable = bool(p.get("redeemable", False))
            outcome = p.get("outcome", "Yes")

            positions.append(Position(
                market_id=p.get("conditionId", ""),
                token_id=p.get("asset", ""),
                station=station,
                target_date=target_date,
                bin_label=bin_label,
                side=outcome,
                shares=size,
                avg_entry_price=avg_price,
                current_price=cur_price,
                cost=initial_value,
                current_value=current_value,
                unrealized_pnl=cash_pnl,
                event_slug=event_slug,
                redeemable=redeemable,
            ))

        return positions

    except Exception as e:
        logger.error("Data API /positions error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Polymarket Data API — fetch closed (resolved) positions
# ---------------------------------------------------------------------------
async def fetch_closed_positions() -> List[dict]:
    """
    GET /closed-positions?user={address}
    Returns resolved trades with realizedPnl.
    """
    addr = _wallet_address()
    if not addr:
        return []

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DATA_API_BASE}/closed-positions",
                params={"user": addr},
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Data API /closed-positions HTTP %d", resp.status)
                    return []
                data = await resp.json()

        if not isinstance(data, list):
            return []

        return data

    except Exception as e:
        logger.error("Data API /closed-positions error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Main sync — combines API data with DB fallback
# ---------------------------------------------------------------------------
async def sync() -> WalletState:
    """
    Fetch real wallet state from Polymarket Data API.
    Falls back to internal DB tracking if API unavailable.
    """
    # Try Data API first
    total_value = await fetch_portfolio_value()
    positions = await fetch_positions()

    if total_value is not None:
        # API success — calculate cash balance
        positions_value = sum(p.current_value for p in positions)
        cash_balance = max(0, total_value - positions_value)

        state = WalletState(
            balance=cash_balance,
            positions=positions,
            resolved=[],
            total_value=total_value,
            source="api",
        )

        logger.info(
            "Wallet sync (API): total=$%.2f cash=$%.2f positions=%d value=$%.2f",
            total_value, cash_balance, len(positions), positions_value,
        )
        return state

    # Fallback to internal tracking
    logger.warning("Data API unavailable — falling back to DB tracking")
    return await _db_fallback_sync()


async def _db_fallback_sync() -> WalletState:
    """Internal capital tracking from database."""
    import tracker

    base_capital = _manual_capital if _manual_capital is not None else _config_capital

    try:
        summary = await tracker.get_capital_summary()
        total_profit = summary.get("total_profit", 0.0)
        total_loss = summary.get("total_loss", 0.0)
        deployed = summary.get("deployed", 0.0)

        balance = base_capital + total_profit - total_loss - deployed
        total_value = base_capital + total_profit - total_loss

        positions = []
        for t in summary.get("open_positions", []):
            positions.append(Position(
                market_id=t.get("market_id", ""),
                token_id="",
                station=t.get("station", ""),
                target_date=t.get("target_date"),
                bin_label=t.get("bin_label", ""),
                side=t.get("side", "YES"),
                shares=t.get("shares", 0),
                avg_entry_price=t.get("entry_price", 0),
                current_price=t.get("entry_price", 0),
                cost=t.get("cost", 0),
                current_value=t.get("cost", 0),
                unrealized_pnl=0.0,
            ))
    except Exception as e:
        logger.warning("DB capital query failed: %s — using config", e)
        balance = base_capital
        total_value = base_capital
        positions = []

    return WalletState(
        balance=max(0, balance),
        positions=positions,
        resolved=[],
        total_value=max(0, total_value),
        source="db_fallback",
    )


# ---------------------------------------------------------------------------
# Helpers — parse position data
# ---------------------------------------------------------------------------
_SLUG_TO_STATION = {
    "seoul": "RKSI",
    "nyc": "KLGA",
    "new-york": "KLGA",
    "miami": "KMIA",
    "chicago": "KORD",
    "london": "EGLC",
    "ankara": "LTAC",
    "buenos-aires": "SAEZ",
    "dallas": "KDFW",
    "atlanta": "KATL",
    "seattle": "KSEA",
    "wellington": "NZWN",
    "toronto": "CYYZ",
    "paris": "LFPG",
}


def _extract_station_from_slug(event_slug: str) -> str:
    """Extract ICAO station from event slug like 'highest-temperature-in-nyc-on-...'"""
    slug_lower = event_slug.lower()
    for city_slug, icao in _SLUG_TO_STATION.items():
        if city_slug in slug_lower:
            return icao
    return "UNKNOWN"


def _extract_bin_label(title: str) -> str:
    """
    Extract bin label from position title.
    e.g. "Will the highest temperature in Atlanta be between 40-41°F..." -> "40-41°F"
    """
    import re
    # Match patterns like "40-41°F", "76°F or higher", "39°F or below"
    match = re.search(r"(\d+[-–]\d+°[FCfc])", title)
    if match:
        return match.group(1)
    match = re.search(r"(\d+°[FCfc]\s+or\s+(?:higher|above|below|less))", title, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any temperature-like pattern
    match = re.search(r"(\d+[-–]?\d*\s*°[FCfc])", title)
    if match:
        return match.group(1)
    return title[:40] if title else ""


# ---------------------------------------------------------------------------
# Order book (public CLOB endpoint — no auth needed)
# ---------------------------------------------------------------------------
async def get_order_book(token_id: str) -> OrderBook:
    try:
        url = "https://clob.polymarket.com/book"
        params = {"token_id": token_id}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return OrderBook(bids=[], asks=[])
                data = await resp.json()
    except Exception:
        return OrderBook(bids=[], asks=[])

    bids = [{"price": float(l.get("price", 0)), "size": float(l.get("size", 0))}
            for l in data.get("bids", [])]
    asks = [{"price": float(l.get("price", 0)), "size": float(l.get("size", 0))}
            for l in data.get("asks", [])]
    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])
    return OrderBook(bids=bids, asks=asks)


def calculate_fill_price(order_book: OrderBook, desired_shares: float,
                          side: str) -> FillInfo:
    levels = order_book.asks if side == "YES" else order_book.bids
    if not levels:
        return FillInfo(average_fill_price=0.0, fillable_shares=0.0, slippage=0.0)

    best_price = levels[0]["price"]
    total_cost = 0.0
    filled = 0.0
    remaining = desired_shares

    for level in levels:
        take = min(remaining, level["size"])
        total_cost += take * level["price"]
        filled += take
        remaining -= take
        if remaining <= 0:
            break

    if filled == 0:
        return FillInfo(average_fill_price=0.0, fillable_shares=0.0, slippage=0.0)

    avg_price = total_cost / filled
    return FillInfo(
        average_fill_price=round(avg_price, 4),
        fillable_shares=round(filled, 2),
        slippage=round(abs(avg_price - best_price), 4),
    )
