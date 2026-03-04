"""
metar_scanner.py — Phase 3 Physical Edge Detection

Runs an isolated asyncio background loop every 15 minutes.
Fetches public METAR strings from AviationWeather.gov without using Open-Meteo API limits.
Detects physical anomalies (Cloud Clearing, Wind Vector Shifts) to fire tactical alerts.
"""

import asyncio
import logging
from datetime import datetime
import pytz

import metar as metar_mod
import markets
import alerts

logger = logging.getLogger("metar_scanner")

# Coastal definitions: (ocean_min_degree, ocean_max_degree)
COASTAL_ANGLES = {
    "KMIA": (10, 170), # Atlantic to the East
    "KLGA": (90, 270), # Atlantic/Sound to the South/East
    "KORD": (0, 180),  # Lake Michigan to the East
    "KBOS": (0, 180),  # Atlantic to the East/North
    "YSSY": (0, 180),  # Tasman Sea to the East
}

_state = {}

def is_ocean_wind(icao, wind_dir):
    if not wind_dir: return False
    if icao not in COASTAL_ANGLES: return False
    omin, omax = COASTAL_ANGLES[icao]
    if omin <= omax:
        return omin <= wind_dir <= omax
    else:
        return wind_dir >= omin or wind_dir <= omax

async def scan_loop():
    logger.info("METAR Physical Edge Scanner initializing...")
    await asyncio.sleep(10) # Delay boot to allow Main DB initialization
    
    while True:
        try:
            stations = [cfg["icao"] for cfg in markets.CITIES.values()]
            # 1 API call for all 15 stations. 0 Open-Meteo limits used.
            raw_data = await metar_mod.fetch_all_stations(stations)
            
            for city_slug, station_config in markets.CITIES.items():
                station_id = station_config["icao"]
                tz_name = station_config["tz"]
                city_name = station_config["city"]
                
                if station_id not in raw_data:
                    continue
                    
                current = raw_data[station_id]
                prev = _state.get(station_id)
                
                if prev:
                    local_tz = pytz.timezone(tz_name)
                    now_local = datetime.now(local_tz)
                    
                    prev_raw = getattr(prev, "raw", "").upper()
                    curr_raw = getattr(current, "raw", "").upper()
                    
                    # 1. Cloud Clearing (Sunbeam Spike)
                    # OVC (Overcast) / BKN (Broken) -> CLR (Clear) / FEW / SKC
                    prev_cloudy = "OVC" in prev_raw or "BKN" in prev_raw
                    curr_clear = "CLR" in curr_raw or "FEW" in curr_raw or "SKC" in curr_raw
                    
                    if prev_cloudy and curr_clear and 9 <= now_local.hour <= 13:
                        msg = (
                            f"🚨 *SUNBEAM SPIKE DETECTED at {station_id} ({city_name})*\n\n"
                            f"Clouds cleared hours early. Expect peak heating to overshoot models due to direct solar radiation.\n\n"
                            f"Action: Seek higher bins."
                        )
                        await alerts.send_tactical_edge_alert(msg)
                        
                    # 2. Wind-Vector Spike
                    # Sudden shift from Ocean Breeze (Cold) to Land Breeze (Hot)
                    if getattr(current, "wind_speed_kt", 0) > 5 and getattr(prev, "wind_speed_kt", 0) > 0:
                        prev_ocean = is_ocean_wind(station_id, getattr(prev, "wind_dir", None))
                        curr_ocean = is_ocean_wind(station_id, getattr(current, "wind_dir", None))
                        
                        if prev_ocean and not curr_ocean:
                            msg = (
                                f"🚨 *WIND-VECTOR SHIFT DETECTED at {station_id} ({city_name})*\n\n"
                                f"Sea breeze collapsed. Wind shifted inland towards `{current.wind_dir}°` at `{current.wind_speed_kt}kts`.\n"
                                f"Sudden 2-4°F temperature spike imminent.\n\n"
                                f"Action: Lock in upper bins."
                            )
                            await alerts.send_tactical_edge_alert(msg)
                
                # Save state for comparison in 15 mins
                _state[station_id] = current
                
        except Exception as e:
            logger.error(f"Error in METAR scanner loop: {e}")
            
        await asyncio.sleep(900) # Run every 15 minutes seamlessly

async def start_background_loop():
    await scan_loop()
