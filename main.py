"""
main.py — Main Loop & Orchestration

Entry point. Runs the async event loop. Orchestrates the 12-step scan cycle.
Manages scan frequency (normal 5min vs fast 2min during model releases).
Initialises all modules, connects to database, starts Telegram bot.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, date
from typing import Dict

from aiohttp import web

import yaml

# Try to use uvloop for better async performance (Linux/macOS)
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import alerts
import allocator
import commands
import markets
import metar as metar_mod
import models as models_mod
import probability
import scheduler
import signals as signals_mod
import tracker
import wallet as wallet_mod

from telegram.ext import Application

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


import telegram.error

# ---------------------------------------------------------------------------
# Telegram bot setup
# ---------------------------------------------------------------------------
async def setup_telegram(config: dict) -> Application:
    """Build and initialise the Telegram bot application."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    commands.register_handlers(app)
    await app.initialize()
    
    # Force delete any lingering webhooks that might conflict with polling
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logger.warning("Could not delete webhook: %s", e)

    await app.start()
    
    async def keepalive_polling(app_instance: Application):
        """Continuously check if polling died from 409 Conflict and restart it."""
        while True:
            try:
                if not app_instance.updater.running:
                    logger.warning("Telegram polling is not running (likely due to 409 Conflict). Starting...")
                    await app_instance.updater.start_polling(
                        drop_pending_updates=True,
                        allowed_updates=["message"],
                    )
            except Exception as e:
                logger.error("Error in keepalive polling restarter: %s", e)
            await asyncio.sleep(20)

    # Start the watcher task in the background
    asyncio.create_task(keepalive_polling(app))
    
    return app


# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------
async def idle_forever():
    """Keeps the process alive without burning CPU or API limits."""
    while True:
        await asyncio.sleep(3600)



# ---------------------------------------------------------------------------
# Health check HTTP server (keeps Render free web service alive)
# ---------------------------------------------------------------------------
async def health_handler(request):
    """Health check endpoint for Render / UptimeRobot."""
    return web.Response(text="Weather-Edge Bot is running", status=200)


async def status_handler(request):
    """Quick status endpoint."""
    uptime = (datetime.utcnow() - _boot_time).total_seconds() / 3600.0
    return web.Response(
        text=f"Weather-Edge Bot | Uptime: {uptime:.1f}h",
        status=200,
    )


_boot_time = datetime.utcnow()


async def run_health_server():
    """Run a lightweight HTTP server on $PORT for Render."""
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)

    port = int(os.environ.get("PORT", 10000))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Health check server started on port %d", port)

    # Keep running forever
    while True:
        await asyncio.sleep(3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def run_all():
    """Run health check server and telegram polling only (on-demand mode)."""
    config = load_config()
    logger.info("Config loaded")
    
    # 1. Database
    await tracker.init_db()
    logger.info("Database connected")
    
    # 2. Telegram bot (registers /start + 15 city commands)
    tg_app = await setup_telegram(config)
    
    # 3. Wallet sync once at startup
    try:
        ws = await wallet_mod.sync()
        logger.info("Wallet: $%.2f", ws.balance)
    except Exception as e:
        logger.error("Initial wallet sync failed: %s", e)
        ws = wallet_mod.WalletState(balance=0.0)
    
    # 4. Send startup message
    import markets
    stations_cfg = {city_cfg["icao"]: city_cfg for city_slug, city_cfg in markets.CITIES.items()}
    await alerts.send_startup_message(ws, list(stations_cfg.keys()))
    logger.info("Startup complete — zero background API polling initialized.")
    
    # 5. Run health server + telegram polling forever
    await asyncio.gather(
        run_health_server(),
        idle_forever(),
    )


def main():
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
