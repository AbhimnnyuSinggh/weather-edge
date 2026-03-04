"""
markets.py — Polymarket Price Fetch, Bin Parsing, Order Book

Fetches all active weather markets from Polymarket Gamma API.
Parses market titles to extract city, date, bin labels.
Maps cities to ICAO stations. Fetches order book depth for trade targets.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import math

import aiohttp
from dateutil import parser as dateparser
import yaml

logger = logging.getLogger("markets")

GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
GAMMA_EVENT_SLUG_URL = "https://gamma-api.polymarket.com/events/slug"
USER_AGENT = "WeatherEdgeBot/1.0"

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class BinInfo:
    label: str
    low: float
    high: float
    unit: str
    is_edge: bool = False

@dataclass
class MarketBin:
    market_id: str
    token_id: str
    no_token_id: str = ""
    bin: BinInfo
    yes_price: float
    volume_24h: float = 0.0
    liquidity_usd: float = 0.0
    polymarket_url: str = ""

@dataclass
class MarketGroup:
    station: str
    city: str
    target_date: date
    bins: List[MarketBin] = field(default_factory=list)
    event_id: str = ""
    resolution_source: str = ""

# ---------------------------------------------------------------------------
# City Configuration & Routing
# ---------------------------------------------------------------------------
CITIES = {
    # US Cities (10)
    "nyc": {"icao": "KLGA", "city": "New York", "slug": "nyc", "lat": 40.7772, "lon": -73.8726, "tz": "America/New_York", "unit": "F", "is_coastal": True, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "chi": {"icao": "KORD", "city": "Chicago", "slug": "chicago", "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago", "unit": "F", "is_coastal": False, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "mia": {"icao": "KMIA", "city": "Miami", "slug": "miami", "lat": 25.7959, "lon": -80.2870, "tz": "America/New_York", "unit": "F", "is_coastal": True, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "atl": {"icao": "KATL", "city": "Atlanta", "slug": "atlanta", "lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "unit": "F", "is_coastal": False, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "den": {"icao": "KDEN", "city": "Denver", "slug": "denver", "lat": 39.8561, "lon": -104.6737, "tz": "America/Denver", "unit": "F", "is_coastal": False, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "hou": {"icao": "KIAH", "city": "Houston", "slug": "houston", "lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "unit": "F", "is_coastal": True, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "phx": {"icao": "KPHX", "city": "Phoenix", "slug": "phoenix", "lat": 33.4373, "lon": -112.0078, "tz": "America/Phoenix", "unit": "F", "is_coastal": False, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "dal": {"icao": "KDFW", "city": "Dallas", "slug": "dallas", "lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "unit": "F", "is_coastal": False, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "sea": {"icao": "KSEA", "city": "Seattle", "slug": "seattle", "lat": 47.4502, "lon": -122.3088, "tz": "America/Los_Angeles", "unit": "F", "is_coastal": True, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},
    "bos": {"icao": "KBOS", "city": "Boston", "slug": "boston", "lat": 42.3656, "lon": -71.0096, "tz": "America/New_York", "unit": "F", "is_coastal": True, "country": "US", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "hrrr", "nbm", "arpege", "ukmo", "bom"]},

    # International Cities (5)
    "sel": {"icao": "RKSI", "city": "Seoul", "slug": "seoul", "lat": 37.4692, "lon": 126.4505, "tz": "Asia/Seoul", "unit": "C", "is_coastal": True, "country": "KR", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "arpege", "ukmo", "bom"]},
    "lon": {"icao": "EGLC", "city": "London", "slug": "london", "lat": 51.5053, "lon": 0.0553, "tz": "Europe/London", "unit": "C", "is_coastal": False, "country": "UK", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "arpege", "ukmo", "bom"]},
    "tok": {"icao": "RJTT", "city": "Tokyo", "slug": "tokyo", "lat": 35.5494, "lon": 139.7798, "tz": "Asia/Tokyo", "unit": "C", "is_coastal": True, "country": "JP", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "arpege", "ukmo", "bom"]},
    "par": {"icao": "LFPG", "city": "Paris", "slug": "paris", "lat": 49.0097, "lon": 2.5479, "tz": "Europe/Paris", "unit": "C", "is_coastal": False, "country": "FR", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "arpege", "ukmo", "bom"]},
    "syd": {"icao": "YSSY", "city": "Sydney", "slug": "sydney", "lat": -33.9461, "lon": 151.1772, "tz": "Australia/Sydney", "unit": "C", "is_coastal": True, "country": "AU", "models": ["gfs", "ecmwf", "icon", "gem", "jma", "arpege", "ukmo", "bom"]},
}

def _city_key_to_icao(city_key: str) -> Optional[str]:
    """Return ICAO for a given /city_key (e.g. mia -> KMIA)."""
    return CITIES.get(city_key.lower(), {}).get("icao")

async def fetch_city_market(city_key: str, target_date: date) -> Optional[MarketGroup]:
    """
    Fetch the Polymarket event for a specific city on a specific date using its exact URL slug.
    Returns the loaded MarketGroup or None if no market exists.
    """
    city_config = CITIES.get(city_key.lower())
    if not city_config:
        return None
        
    slug_city = city_config["slug"]
    month_name = MONTH_NAMES.get(target_date.month, "")
    day = target_date.day
    year = target_date.year
    
    slug = f"highest-temperature-in-{slug_city}-on-{month_name}-{day}-{year}"
    
    try:
        url = f"{GAMMA_EVENT_SLUG_URL}/{slug}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": USER_AGENT}, timeout=15) as resp:
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    logger.error("Gamma API error for %s: HTTP %d", slug, resp.status)
                    return None
                    
                event = await resp.json()
                if not event:
                    return None
                    
                return await _parse_event_async(event, station=city_config["icao"], friendly_city=city_config["city"], target_date=target_date)
                
    except Exception as e:
        logger.error("Error fetching market slug %s: %s", slug, e)
        return None


# ---------------------------------------------------------------------------
def is_valid_price(price: float, liquidity: float = 0.0) -> bool:
    """Return True if price >= 0.01 and liquidity >= $5.00."""
    return price >= 0.01 and liquidity >= 5.0

async def _parse_event_async(event: dict, station: str, friendly_city: str, target_date: date) -> Optional[MarketGroup]:
    """Parse a Gamma API event into a MarketGroup."""
    title = event.get("title", "")
    if not title:
        return None

    # Parse resolution source from description
    description = event.get("description", "")
    resolution_source = _parse_resolution_source(description)
    event_slug = event.get("slug", "")

    # Determine unit from title or description
    title_lower = title.lower()
    unit = "F"
    if "°c" in title_lower or "celsius" in title_lower or "ºc" in title_lower:
        unit = "C"

    group = MarketGroup(
        station=station,
        city=friendly_city,
        target_date=target_date,
        event_id=str(event.get("id", "")),
        resolution_source=resolution_source,
    )

    # Parse markets (bins) within the event
    markets = event.get("markets", [])
    for market in markets:
        mbin = _parse_market_bin(market, unit, event_slug)
        if mbin:
            # Enforce 0-cent minimums and liquidity checks
            if not is_valid_price(mbin.yes_price, mbin.liquidity_usd):
                continue
            
            mbin.yes_price = max(mbin.yes_price, 0.01)
            group.bins.append(mbin)

    # Sort bins by low bound
    group.bins.sort(key=lambda b: b.bin.low)

    return group





def _parse_resolution_source(description: str) -> str:
    """Extract resolution source from event description."""
    desc_lower = description.lower()
    if "weather underground" in desc_lower or "wunderground" in desc_lower:
        # Try to find ICAO or station name
        icao_match = re.search(r"\b([A-Z]{4})\b", description)
        if icao_match:
            return f"Weather Underground ({icao_match.group(1)})"
        return "Weather Underground"
    return description[:100] if description else "Unknown"


# ---------------------------------------------------------------------------
# Market bin parsing
# ---------------------------------------------------------------------------
def _parse_market_bin(market: dict, default_unit: str, event_slug: str = "") -> Optional[MarketBin]:
    """Parse a single market (bin) from the event."""
    market_id = str(market.get("conditionId", market.get("id", "")))
    question = market.get("question", "")
    group_title = market.get("groupItemTitle", "")

    # Parse bin from available title fields
    bin_info = parse_bin_from_title(group_title or question, default_unit)
    if not bin_info:
        return None

    # Get YES price from outcomes
    yes_price = 0.0
    outcomes = market.get("outcomes", [])
    outcome_prices = market.get("outcomePrices", "")

    if outcome_prices:
        try:
            # outcomePrices is typically a JSON string like "[0.42, 0.58]"
            import json
            prices = json.loads(outcome_prices)
            if prices and len(prices) > 0:
                yes_price = float(prices[0])
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    if yes_price == 0:
        # Fallback: check market-level price
        yes_price = float(market.get("bestBid", 0) or 0)

    # Token IDs (clobTokenIds is usually [YES_TOKEN_ID, NO_TOKEN_ID])
    tokens = market.get("clobTokenIds", "")
    token_id = ""
    no_token_id = ""
    if tokens:
        try:
            import json
            token_list = json.loads(tokens)
            if token_list and len(token_list) > 0:
                token_id = str(token_list[0])
            if token_list and len(token_list) > 1:
                no_token_id = str(token_list[1])
        except (json.JSONDecodeError, ValueError):
            pass

    # Volume and Liquidity
    volume = float(market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidity", 0) or 0)

    # Polymarket URL
    poly_url = f"https://polymarket.com/event/{event_slug}" if event_slug else ""

    return MarketBin(
        market_id=market_id,
        token_id=token_id,
        no_token_id=no_token_id,
        bin=bin_info,
        yes_price=yes_price,
        volume_24h=volume,
        liquidity_usd=liquidity,
        polymarket_url=poly_url,
    )


# ---------------------------------------------------------------------------
# Bin parsing from title strings
# ---------------------------------------------------------------------------
def parse_bin_from_title(title: str, unit: str = "F") -> Optional[BinInfo]:
    """
    Handle various Polymarket bin title formats:
    - "78 - 79" (group_item_title)
    - "Will the high be between 78°F and 79°F?"
    - "12°C to 13°C"
    - "80-81°F"
    - "50+" or "20-" (edge bins)
    """
    if not title:
        return None

    title = title.strip()

    # Detect unit from title
    if "°F" in title or "°f" in title:
        unit = "F"
    elif "°C" in title or "°c" in title:
        unit = "C"

    # Edge bins: "50+" or "48°F or above" or "above 50"
    edge_above = (
        re.search(r"(\d+)\s*\+", title)
        or re.search(r"(?:above|over|more than)\s+(\d+)", title, re.IGNORECASE)
        or re.search(r"(\d+)\s*°[FfCc]?\s+or\s+above", title, re.IGNORECASE)
    )
    if edge_above:
        val = float(edge_above.group(1))
        return BinInfo(
            label=f"{int(val)}+°{unit}",
            low=val, high=val + 100,
            unit=unit, is_edge=True,
        )

    # Edge bins: "20-" or "39°F or below" or "below 20"
    edge_below = (
        re.search(r"(\d+)\s*-\s*$", title)
        or re.search(r"(?:below|under|less than)\s+(\d+)", title, re.IGNORECASE)
        or re.search(r"(\d+)\s*°[FfCc]?\s+or\s+(?:below|less)", title, re.IGNORECASE)
    )
    if edge_below:
        val = float(edge_below.group(1))
        return BinInfo(
            label=f"{int(val)}-°{unit}",
            low=val - 100, high=val,
            unit=unit, is_edge=True,
        )

    # Range patterns: "78 - 79", "78-79", "between 78 and 79"
    range_match = re.search(r"(-?\d+\.?\d*)\s*[-–—to]+\s*(-?\d+\.?\d*)", title)
    if not range_match:
        range_match = re.search(
            r"between\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)", title, re.IGNORECASE
        )

    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if low > high:
            low, high = high, low
        label = f"{int(low)}-{int(high)}°{unit}"
        return BinInfo(label=label, low=low, high=high, unit=unit)

    # Single number (exact temp bin)
    single_match = re.search(r"(\d+)", title)
    if single_match:
        val = float(single_match.group(1))
        return BinInfo(
            label=f"{int(val)}°{unit}",
            low=val, high=val + 1,
            unit=unit,
        )

    return None



