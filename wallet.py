"""
wallet.py — Capital Tracking & Polymarket Wallet Sync

Tracks capital internally via database. Uses starting_capital from config.
Calculates balance from trade history (profits - losses - deployed).
Order book queries use public CLOB endpoint (no auth needed).
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("wallet")

USER_AGENT = "WeatherEdgeBot/1.0"


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


# ---------------------------------------------------------------------------
# Internal capital tracking
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


async def sync() -> WalletState:
    """
    Calculate wallet state from internal tracking.
    balance = starting_capital + profits - losses - deployed
    """
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
        deployed = 0.0

    state = WalletState(
        balance=max(0, balance),
        positions=positions,
        resolved=[],
        total_value=max(0, total_value),
    )

    logger.info(
        "Wallet sync: balance=%.2f deployed=%.2f total=%.2f positions=%d",
        state.balance, deployed if 'deployed' in dir() else 0,
        state.total_value, len(state.positions),
    )
    return state


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
