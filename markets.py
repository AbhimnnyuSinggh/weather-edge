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
GAMMA_EVENT_SLUG_URL = "https://gamma-api.polymarket.com/events/slug"
USER_AGENT = "WeatherEdgeBot/1.0"

# Known city → ICAO mappings (from resolution rules research)
CITY_TO_STATION = {
    "seoul": "RKSI",
    "incheon": "RKSI",
    "new york": "KLGA",
    "new york city": "KLGA",
    "nyc": "KLGA",
    "miami": "KMIA",
    "chicago": "KORD",
    "london": "EGLC",
    "ankara": "LTAC",
    "buenos aires": "SAEZ",
    "dallas": "KDFW",
    "atlanta": "KATL",
    "seattle": "KSEA",
    "wellington": "NZWN",
    "toronto": "CYYZ",
    "paris": "LFPG",
}

# City slug names used in Polymarket URLs
# Format: "highest-temperature-in-{slug_city}-on-{month}-{day}-{year}"
CITY_SLUG_NAMES = {
    "RKSI": "seoul",
    "KLGA": "nyc",
    "KMIA": "miami",
    "KORD": "chicago",
    "EGLC": "london",
    "LTAC": "ankara",
    "SAEZ": "buenos-aires",
    "KDFW": "dallas",
    "KATL": "atlanta",
    "KSEA": "seattle",
    "NZWN": "wellington",
    "CYYZ": "toronto",
    "LFPG": "paris",
}

# Friendly city names for display
STATION_CITY_NAMES = {
    "RKSI": "Seoul",
    "KLGA": "NYC",
    "KMIA": "Miami",
    "KORD": "Chicago",
    "EGLC": "London",
    "LTAC": "Ankara",
    "SAEZ": "Buenos Aires",
    "KDFW": "Dallas",
    "KATL": "Atlanta",
    "KSEA": "Seattle",
    "NZWN": "Wellington",
    "CYYZ": "Toronto",
    "LFPG": "Paris",
}

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
# Build event slugs for known stations and dates
# ---------------------------------------------------------------------------
def _build_event_slugs(station_ids: List[str], target_dates: List[date]
                        ) -> List[Tuple[str, str, date]]:
    """
    Build Gamma API event slugs for each station/date combo.
    Returns: [(slug, station, target_date), ...]
    """
    slugs = []
    for station in station_ids:
        slug_city = CITY_SLUG_NAMES.get(station)
        if not slug_city:
            continue
        for d in target_dates:
            month_name = MONTH_NAMES.get(d.month, "")
            slug = f"highest-temperature-in-{slug_city}-on-{month_name}-{d.day}-{d.year}"
            slugs.append((slug, station, d))
    return slugs


# ---------------------------------------------------------------------------
# Fetch active weather markets — slug-based lookup
# ---------------------------------------------------------------------------
async def fetch_active_weather_markets(known_stations: List[str] = None
                                        ) -> Dict[str, MarketGroup]:
    """
    Fetch daily temperature events by constructing slugs directly.
    Polymarket uses: "highest-temperature-in-{city}-on-{month}-{day}-{year}"
    Returns: {station_date_key: MarketGroup}
    """
    from datetime import timedelta
    today = date.today()
    tomorrow = today + timedelta(days=1)
    target_dates = [today, tomorrow]

    stations = known_stations or list(CITY_SLUG_NAMES.keys())
    slugs = _build_event_slugs(stations, target_dates)

    results: Dict[str, MarketGroup] = {}

    async with aiohttp.ClientSession() as session:
        for slug, station, target_date in slugs:
            try:
                url = f"{GAMMA_EVENT_SLUG_URL}/{slug}"
                async with session.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 404:
                        # Market not created yet for this date
                        continue
                    if resp.status != 200:
                        logger.warning("Gamma slug fetch HTTP %d for %s", resp.status, slug)
                        continue
                    event = await resp.json()
            except Exception as e:
                logger.error("Gamma slug fetch error for %s: %s", slug, e)
                continue

            try:
                group = _parse_event(event, station)
                if group and group.bins:
                    key = f"{station}_{target_date.isoformat()}"
                    results[key] = group
                    logger.info(
                        "Found market: %s %s — %d bins",
                        station, target_date, len(group.bins),
                    )
            except Exception as e:
                logger.error("Event parse error for %s: %s", slug, e)

    logger.info("Markets fetched: %d events across %d stations",
                len(results), len(stations))
    return results


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------
def _parse_event(event: dict, station: str = None) -> Optional[MarketGroup]:
    """Parse a Gamma API event into a MarketGroup."""
    title = event.get("title", "")
    if not title:
        return None

    # Extract city and date from title
    city, target_date = _parse_title(title)
    if not target_date:
        return None

    # Use provided station or map from city
    if not station:
        station = map_city_to_station(city) if city else "UNKNOWN"

    # Use known city name if station provided
    if not city:
        city = STATION_CITY_NAMES.get(station, "Unknown")

    # Parse resolution source from description
    description = event.get("description", "")
    resolution_source = _parse_resolution_source(description)

    # Determine unit from title or description
    title_lower = title.lower()
    unit = "F"
    if "°c" in title_lower or "celsius" in title_lower or "ºc" in title_lower:
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
