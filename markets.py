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

import aiohttp
from dateutil import parser as dateparser

logger = logging.getLogger("markets")

GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
USER_AGENT = "WeatherEdgeBot/1.0"

# Known city → ICAO mappings (from resolution rules research)
CITY_TO_STATION = {
    "seoul": "RKSI",
    "incheon": "RKSI",
    "new york": "KLGA",
    "nyc": "KLGA",
    "miami": "KMIA",
    "chicago": "KORD",
    "london": "EGLC",
    "ankara": "LTAC",
    "buenos aires": "SAEZ",
    "dallas": "KDFW",
    "atlanta": "KATL",
    "seattle": "KSEA",
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
    bin: BinInfo
    yes_price: float
    volume_24h: float = 0.0
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
# Fetch active weather markets
# ---------------------------------------------------------------------------
async def fetch_active_weather_markets(known_stations: List[str] = None
                                        ) -> Dict[str, MarketGroup]:
    """
    Fetch all active weather events from Gamma API.
    Returns: {station_date_key: MarketGroup}
    """
    params = {
        "tag": "climate",
        "active": "true",
        "closed": "false",
        "limit": 100,
    }

    results: Dict[str, MarketGroup] = {}
    new_cities: List[str] = []

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                GAMMA_API_URL, params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    logger.error("Gamma API HTTP %d", resp.status)
                    return results
                events = await resp.json()
    except Exception as e:
        logger.error("Gamma API error: %s", e)
        return results

    if not isinstance(events, list):
        events = []

    for event in events:
        try:
            group = _parse_event(event)
            if group is None:
                continue

            # Check if this is a known station
            if known_stations and group.station not in known_stations:
                if group.station != "UNKNOWN":
                    new_cities.append(group.city)
                continue

            key = f"{group.station}_{group.target_date.isoformat()}"
            results[key] = group
        except Exception as e:
            logger.error("Event parse error: %s — %s",
                         event.get("title", "?")[:50], e)

    if new_cities:
        logger.info("New cities detected: %s", new_cities)

    return results


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------
def _parse_event(event: dict) -> Optional[MarketGroup]:
    """Parse a Gamma API event into a MarketGroup."""
    title = event.get("title", "")
    if not title:
        return None

    # Filter: only temperature-related weather events
    title_lower = title.lower()
    if not any(kw in title_lower for kw in [
        "temperature", "temp", "high", "°f", "°c", "degrees"
    ]):
        # Not a temperature market — could be rain, wind, etc.
        return None

    # Extract city and date from title
    city, target_date = _parse_title(title)
    if not city or not target_date:
        return None

    # Map city to station
    station = map_city_to_station(city)

    # Parse resolution source from description
    description = event.get("description", "")
    resolution_source = _parse_resolution_source(description)

    # Determine unit from title or description
    unit = "F"
    if "°c" in title_lower or "celsius" in title_lower:
        unit = "C"

    group = MarketGroup(
        station=station,
        city=city,
        target_date=target_date,
        event_id=str(event.get("id", "")),
        resolution_source=resolution_source,
    )

    # Parse markets (bins) within the event
    markets = event.get("markets", [])
    for market in markets:
        mbin = _parse_market_bin(market, unit)
        if mbin:
            group.bins.append(mbin)

    # Sort bins by low bound
    group.bins.sort(key=lambda b: b.bin.low)

    return group


def _parse_title(title: str) -> Tuple[Optional[str], Optional[date]]:
    """
    Parse city and date from event title.
    Examples:
    - "Highest temperature in Seoul on Feb 28"
    - "What will the high temperature be in New York City on March 1, 2026?"
    - "London daily high temperature on 2026-02-28"
    """
    title_lower = title.lower()

    # Extract city: look for "in <city>" pattern
    city = None
    city_match = re.search(r"in\s+([A-Za-z\s]+?)(?:\s+on\s+|\s+(?:for|–|-))", title, re.IGNORECASE)
    if city_match:
        city = city_match.group(1).strip()
    else:
        # Try known city names directly
        for known_city in CITY_TO_STATION:
            if known_city in title_lower:
                city = known_city.title()
                break

    # Extract date
    target_date = None
    # Try explicit date patterns
    date_match = re.search(
        r"(?:on|for)\s+(.+?)(?:\?|$)",
        title, re.IGNORECASE
    )
    if date_match:
        date_str = date_match.group(1).strip().rstrip("?")
        try:
            parsed = dateparser.parse(date_str, fuzzy=True)
            if parsed:
                target_date = parsed.date()
        except (ValueError, TypeError):
            pass

    # Fallback: look for ISO date in title
    if not target_date:
        iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", title)
        if iso_match:
            try:
                target_date = date.fromisoformat(iso_match.group(1))
            except ValueError:
                pass

    # Fallback: look for Month Day format
    if not target_date:
        month_match = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2})",
            title, re.IGNORECASE,
        )
        if month_match:
            try:
                parsed = dateparser.parse(
                    f"{month_match.group(1)} {month_match.group(2)}, {date.today().year}",
                    fuzzy=True,
                )
                if parsed:
                    target_date = parsed.date()
            except (ValueError, TypeError):
                pass

    return city, target_date


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
def _parse_market_bin(market: dict, default_unit: str) -> Optional[MarketBin]:
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

    # Token IDs
    tokens = market.get("clobTokenIds", "")
    token_id = ""
    if tokens:
        try:
            import json
            token_list = json.loads(tokens)
            if token_list:
                token_id = str(token_list[0])
        except (json.JSONDecodeError, ValueError):
            pass

    # Volume
    volume = float(market.get("volume", 0) or 0)

    # Polymarket URL
    slug = market.get("slug", "")
    poly_url = f"https://polymarket.com/event/{slug}" if slug else ""

    return MarketBin(
        market_id=market_id,
        token_id=token_id,
        bin=bin_info,
        yes_price=yes_price,
        volume_24h=volume,
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

    # Edge bins: "50+" or "20-" or "above 50" or "below 20"
    edge_above = re.search(r"(\d+)\s*\+", title) or re.search(
        r"(?:above|over|more than)\s+(\d+)", title, re.IGNORECASE
    )
    if edge_above:
        val = float(edge_above.group(1))
        return BinInfo(
            label=f"{int(val)}+°{unit}",
            low=val, high=val + 100,
            unit=unit, is_edge=True,
        )

    edge_below = re.search(r"(\d+)\s*-\s*$", title) or re.search(
        r"(?:below|under|less than)\s+(\d+)", title, re.IGNORECASE
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


# ---------------------------------------------------------------------------
# City → Station mapping
# ---------------------------------------------------------------------------
def map_city_to_station(city: str) -> str:
    """Map city name from Polymarket to ICAO code."""
    if not city:
        return "UNKNOWN"
    city_lower = city.lower().strip()
    # Direct match
    if city_lower in CITY_TO_STATION:
        return CITY_TO_STATION[city_lower]
    # Partial match
    for known, icao in CITY_TO_STATION.items():
        if known in city_lower or city_lower in known:
            return icao
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# New city detection
# ---------------------------------------------------------------------------
def detect_new_cities(events: list, known_stations: List[str]) -> List[str]:
    """Find cities in events that aren't in our station registry."""
    new_cities = []
    for event in events:
        title = event.get("title", "")
        city, _ = _parse_title(title)
        if city:
            station = map_city_to_station(city)
            if station == "UNKNOWN" or station not in known_stations:
                new_cities.append(city)
    return list(set(new_cities))
