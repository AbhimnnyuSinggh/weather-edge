import asyncio
from datetime import date
import sys
sys.path.append("/Users/NewUser/Desktop/PolyWeather")
import markets
import models

async def test():
    print("Testing Market slug builder...")
    # target_date is today
    import datetime as dt
    target_date = dt.date(2026, 3, 1)
    
    city_cmd = "nyc"
    city_cfg = markets.CITIES[city_cmd]
    slug = markets._build_slug(city_cfg["city"], target_date)
    print(f"NYC Slug: {slug}")
    
    market_group = await markets.fetch_city_market(city_cmd, target_date)
    if market_group:
        print(f"Found active market with {len(market_group.bins)} bins")
    else:
        print("No active market found")

    print("\nTesting Model fetching (KLGA)...")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        # NWS
        try:
            nws_temp = await models._fetch_nws(session, city_cfg["lat"], city_cfg["lon"])
            print(f"NWS: {nws_temp}")
        except Exception as e:
            print(f"NWS Error: {e}")
            
        # MOS
        try:
            mos_temp = await models._fetch_noaa_mos(session, city_cfg["icao"])
            print(f"MOS: {mos_temp}")
        except Exception as e:
            print(f"MOS Error: {e}")
            
        # Open-Meteo
        try:
            om_data = await models._fetch_open_meteo(session, city_cfg["lat"], city_cfg["lon"], target_date)
            print(f"Open-Meteo: {om_data}")
        except Exception as e:
            print(f"Open-Meteo Error: {e}")

asyncio.run(test())
