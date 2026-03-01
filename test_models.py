import asyncio
from datetime import date
import aiohttp
import logging
logging.basicConfig(level=logging.INFO)

import markets
import models
import metar
import wallet

async def main():
    print("Testing NYC (US) Models...")
    nyc_cfg = {"KLGA": markets.CITIES["nyc"]}
    all_models = await models.fetch_all_stations(nyc_cfg)
    print("NYC Models:", list(all_models.get("KLGA", {}).keys()))
    
    print("\nTesting SEOUL (INTL) Models...")
    sel_cfg = {"RKSI": markets.CITIES["sel"]}
    all_models = await models.fetch_all_stations(sel_cfg)
    print("Seoul Models:", list(all_models.get("RKSI", {}).keys()))

    print("\nTesting Data API...")
    try:
        cap = await wallet.get_capital_summary()
        print("Data API Capital:", cap)
    except Exception as e:
        print("Wallet error:", e)

    print("\nTesting Gamma Fetch for NYC...")
    market = await markets.fetch_city_market("nyc", date.today())
    if market:
        print(f"Market Found! Bins: {len(market.bins)}")
    else:
        print("No NYC market today.")
        
if __name__ == "__main__":
    asyncio.run(main())
