"""
ledger.py — Polymarket Accountability Dashboard

Retrieves actual, on-chain settled trades and active positions from the Polymarket API
(via wallet.py). Formats data into a PnL accountability statement for the Telegram /ledger command.
Tracks the performance of the autonomous snipers.
"""

import logging
from datetime import datetime, timedelta
import pytz

import wallet

logger = logging.getLogger("ledger")

async def generate_ledger_report() -> str:
    """Fetches Live and Closed positions from Polymarket Data API and compiles an ROI statement."""
    try:
        # 1. Fetch current live snapshot
        ws = await wallet.sync()
        
        # 2. Fetch trailing closed positions
        closed = await wallet.fetch_closed_positions()
        
        # We need to filter for Weather markets, because user might trade crypto on PM too.
        # wallet.py already filters open positions. We must filter closed ones here.
        weather_closed = []
        for p in closed:
            slug = p.get("eventSlug", "").lower()
            title = p.get("title", "").lower()
            if "temperature" in slug or "temperature" in title:
                weather_closed.append(p)
                
        # 3. Time bounds
        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz)
        
        # Initialize tracking metrics
        yesterday_count = 0
        yesterday_pnl = 0.0
        
        week_count = 0
        week_pnl = 0.0
        
        month_count = len(weather_closed)
        month_pnl = 0.0
        
        # Calculate trailing metrics
        for p in weather_closed:
            pnl = float(p.get("realizedPnl", 0))
            month_pnl += pnl
            
            # Dates in Gamma are usually ISO strings
            end_date_str = p.get("endDate")
            if end_date_str:
                try:
                    # e.g., '2026-03-04T00:00:00Z'
                    dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).astimezone(tz)
                    days_ago = (now - dt).days
                    
                    if days_ago <= 1:
                        yesterday_count += 1
                        yesterday_pnl += pnl
                    if days_ago <= 7:
                        week_count += 1
                        week_pnl += pnl
                except:
                    pass

        # Format output
        source = "🟢 API Sync Active" if ws.source == "api" else "🟡 DB Fallback Mode"
        
        lines = [
            f"📒 *POLYMARKET ACCOUNTABILITY LEDGER*",
            f"_{now.strftime('%b %d, %Y | %H:%M %Z')}_",
            f"{source}\n",
            f"💰 **Live Wallet**",
            f"• Portfolio Value: `${ws.total_value:.2f}`",
            f"• Deployable Cash: `${ws.balance:.2f}`",
            f"• Open Snipes: `{len(ws.positions)} active`\n",
            f"📊 **Settled Auto-Trades (Weather Only)**",
            f"• **Yesterday:** `{yesterday_count} trades` | {'+' if yesterday_pnl >= 0 else ''}${yesterday_pnl:.2f} PnL",
            f"• **Trailing 7D:** `{week_count} trades` | {'+' if week_pnl >= 0 else ''}${week_pnl:.2f} PnL",
            f"• **Trailing 30D:** `{month_count} trades` | {'+' if month_pnl >= 0 else ''}${month_pnl:.2f} PnL",
        ]
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Ledger generation failed: {e}")
        return "❌ Error generating accountability ledger. Polymarket API may be unreachable."
