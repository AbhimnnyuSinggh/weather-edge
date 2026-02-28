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
# Geocoding and Auto-Discovery
# ---------------------------------------------------------------------------
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

async def _geocode_city(city: str) -> Optional[dict]:
    """Use Open-Meteo to find lat/lon/timezone for a city."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get("results"):
                    res = data["results"][0]
                    return {
                        "lat": res["latitude"],
                        "lon": res["longitude"],
                        "timezone": res.get("timezone", "UTC"),
                        "country": res.get("country_code", "US"),
                    }
    except Exception as e:
        logger.error("Geocoding error for %s: %s", city, e)
    return None

async def _find_closest_icao(lat: float, lon: float) -> Optional[str]:
    """Find closest ICAO airport via AviationWeather."""
    bbox = f"{lon-2},{lat-2},{lon+2},{lat+2}"
    url = f"https://aviationweather.gov/api/data/stationinfo?format=json&bbox={bbox}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                closest, min_dist = None, 999999
                for st in data:
                    slat, slon, sid = st.get("lat"), st.get("lon"), st.get("icaoId")
                    if sid and slat is not None and slon is not None:
                        dist = _haversine(lat, lon, float(slat), float(slon))
                        if dist < min_dist:
                            min_dist = dist
                            closest = sid
                return closest
    except Exception as e:
        logger.error("ICAO lookup error: %s", e)
    return None

async def _auto_add_station_to_config(city: str) -> str:
    """Geocode, find ICAO, and append to config.yaml."""
    geo = await _geocode_city(city)
    if not geo: return "UNKNOWN"
    icao = await _find_closest_icao(geo["lat"], geo["lon"])
    if not icao: return "UNKNOWN"
    
    # Check if already mapped in dicts
    if city.lower() in CITY_TO_STATION: return CITY_TO_STATION[city.lower()]
    
    # Read config.yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml") if "USER_AGENT" in globals() else "config.yaml"
    try:
        if os.path.exists("config.yaml"):
            cfg_path = "config.yaml"
        with open(cfg_path, "r") as f:
            full_cfg = yaml.safe_load(f)
        
        stations = full_cfg.get("stations", {})
        if icao in stations: return icao
        
        # Determine defaults
        unit = "F" if geo["country"] == "US" else "C"
        models = ["gfs", "ecmwf", "icon", "nws", "noaa"] if geo["country"] == "US" else ["ecmwf", "gfs", "icon"]
        biases = {"gfs": 0.0, "ecmwf": 0.0, "icon": 0.0, "nws": 0.0, "noaa": 0.0} if geo["country"] == "US" else {"ecmwf": 0.0, "gfs": 0.0, "icon": 0.0}

        stations[icao] = {
            "city": city,
            "country": geo["country"],
            "lat": geo["lat"],
            "lon": geo["lon"],
            "timezone": geo["timezone"],
            "unit": unit,
            "wunderground_url": "",
            "is_coastal": False,
            "starting_bias": biases,
            "models": models
        }
        
        with open(cfg_path, "w") as f:
            yaml.dump(full_cfg, f, default_flow_style=False, sort_keys=False)
            
        CITY_TO_STATION[city.lower()] = icao
        STATION_CITY_NAMES[icao] = city
        logger.info("Auto-discovered and added %s (%s) to config.", city, icao)
        
        # Send alert
        from alerts import send_new_city_alert
        import asyncio
        asyncio.create_task(send_new_city_alert(city, f"Discovered and mapped to airport: {icao}"))
        return icao
    except Exception as e:
        logger.error("Error auto-adding %s to config: %s", city, e)
        return "UNKNOWN"

# ---------------------------------------------------------------------------
# Fetch active weather markets — Auto-discovery
# ---------------------------------------------------------------------------
async def fetch_active_weather_markets(known_stations: List[str] = None
                                        ) -> Dict[str, MarketGroup]:
    """
    Fetch all events dynamically and discover temperature markets.
    Returns: {station_date_key: MarketGroup}
    """
    import os
    results: Dict[str, MarketGroup] = {}
    
    try:
        async with aiohttp.ClientSession() as session:
            for offset in range(0, 1000, 100):
                url = f"{GAMMA_API_URL}?tag=weather&active=true&closed=false&limit=100&offset={offset}"
                try:
                    async with session.get(
                        url, headers={"User-Agent": USER_AGENT}, timeout=aiohttp.ClientTimeout(total=20)
                    ) as resp:
                        if resp.status != 200:
                            break
                        events = await resp.json()
                        if not events:
                            break
                        
                        found_any_temperature = False
                        for event in events:
                            title = event.get("title", "").lower()
                            if "highest temperature" in title:
                                found_any_temperature = True
                                group = await _parse_event_async(event)
                                if group and group.bins and group.station != "UNKNOWN":
                                    key = f"{group.station}_{group.target_date.isoformat()}"
                                    results[key] = group
                                    logger.info(
                                        "Found market: %s %s — %d bins",
                                        group.station, group.target_date, len(group.bins),
                                    )
                        
                        # Optimization: if we found weather events but none of them are temperature,
                        # and we have enough, we could break. But we will just loop for safety.
                except Exception as loop_e:
                    logger.warning("Gamma chunk fetch timeout for offset %d", offset)
                    continue

    except Exception as e:
        import traceback
        logger.error("Gamma dynamic fetch error: %s\n%s", e, traceback.format_exc())

    return results


# ---------------------------------------------------------------------------
def is_valid_price(price: float, liquidity: float = 0.0) -> bool:
    """Return True if price >= 0.01 and liquidity >= $5.00."""
    return price >= 0.01 and liquidity >= 5.0

async def _parse_event_async(event: dict, station: str = None) -> Optional[MarketGroup]:
    """Parse a Gamma API event into a MarketGroup and natively auto-discover cities."""
    title = event.get("title", "")
    if not title:
        return None

    # Extract city and date from title
    city, target_date = _parse_title(title)
    if not target_date or not city:
        return None

    # Use provided station or map from city
    if not station:
        station = CITY_TO_STATION.get(city.lower())
        if not station:
            # Auto-discover logic
            station = await _auto_add_station_to_config(city)

    # Use known city name if station provided
    friendly_city = STATION_CITY_NAMES.get(station, city.title())

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
        city=friendly_city,
        target_date=target_date,
        event_id=str(event.get("id", "")),
        resolution_source=resolution_source,
    )

    # Parse markets (bins) within the event
    markets = event.get("markets", [])
    for market in markets:
        mbin = _parse_market_bin(market, unit)
        if mbin:
            # Enforce 0-cent minimums and liquidity checks
            if not is_valid_price(mbin.yes_price, mbin.liquidity_usd):
                continue
            
            mbin.yes_price = max(mbin.yes_price, 0.01)
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

    # Volume and Liquidity
    volume = float(market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidity", 0) or 0)

    # Polymarket URL
    slug = market.get("slug", "")
    poly_url = f"https://polymarket.com/event/{slug}" if slug else ""

    return MarketBin(
        market_id=market_id,
        token_id=token_id,
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
