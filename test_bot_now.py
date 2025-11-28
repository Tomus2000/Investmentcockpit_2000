#!/usr/bin/env python3
"""
Quick test script to send portfolio summary and recommendations RIGHT NOW
Run this to test if your bot is working!
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_USER_ID", 
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY"
]

missing_vars = []
for var in required_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n💡 Make sure your .env file exists and has all required variables.")
    print("   See .env.example for the format.")
    sys.exit(1)

# Import after checking env vars
from bot import (
    send_portfolio_summary,
    send_investment_recommendations,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_USER_ID
)
from telegram import Bot
from telegram.ext import Application

async def test_messages():
    """Send test messages immediately"""
    print("🚀 Starting bot test...")
    print(f"📱 Bot Token: {TELEGRAM_BOT_TOKEN[:10]}..." if TELEGRAM_BOT_TOKEN else "❌ No token")
    print(f"📱 User ID: {TELEGRAM_USER_ID}")
    
    # Create bot application
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    except Exception as e:
        print(f"❌ Failed to create bot application: {e}")
        return
    
    # Initialize the application
    try:
        await application.initialize()
        await application.start()
        print("✅ Bot initialized and started")
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        return
    
    # Create a proper context using ContextTypes
    from telegram.ext import ContextTypes
    
    # Create context manually
    class SimpleContext:
        def __init__(self, bot):
            self.bot = bot
    
    context = SimpleContext(application.bot)
    
    try:
        # Test 1: Send portfolio summary
        print("\n📊 Sending portfolio summary...")
        print("   Checking portfolio data...")
        await send_portfolio_summary(context)
        print("✅ Portfolio summary sent!")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test 2: Send investment recommendations
        print("\n🤖 Sending investment recommendations...")
        await send_investment_recommendations(context)
        print("✅ Investment recommendations sent!")
        
        print("\n🎉 All messages sent successfully!")
        print("📱 Check your Telegram now!")
        
    except Exception as e:
        print(f"\n❌ Error sending messages: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        
        # Try to send error message to user
        try:
            await application.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=f"❌ Error in test script: {str(e)}"
            )
        except Exception as send_error:
            print(f"Could not send error message: {send_error}")
    finally:
        try:
            await application.stop()
            await application.shutdown()
        except:
            pass

if __name__ == "__main__":
    print("=" * 60)
    print("TELEGRAM BOT TEST - Sending Messages NOW")
    print("=" * 60)
    print()
    asyncio.run(test_messages())

