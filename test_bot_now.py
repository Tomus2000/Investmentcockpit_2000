#!/usr/bin/env python3
"""
Quick test script to send portfolio summary and recommendations RIGHT NOW
Run this to test if your bot is working!
"""

import os
import asyncio
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Helper to get and clean environment variables
def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable and strip quotes/whitespace"""
    value = os.getenv(key, default)
    if value:
        return str(value).strip().strip('"').strip("'").strip()
    return default

# Check required environment variables
required_vars = {
    "TELEGRAM_BOT_TOKEN": get_env_var("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_USER_ID": get_env_var("TELEGRAM_USER_ID", ""),
    "OPENAI_API_KEY": get_env_var("OPENAI_API_KEY", ""),
    "SUPABASE_URL": get_env_var("SUPABASE_URL", ""),
    "SUPABASE_KEY": get_env_var("SUPABASE_KEY", "")
}

missing_vars = [var for var, value in required_vars.items() if not value]

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n💡 Make sure your .env file exists and has all required variables.")
    print("   For GitHub Actions, make sure secrets are set in repository settings.")
    sys.exit(1)

# Set cleaned values back to environment for bot.py to use
for key, value in required_vars.items():
    os.environ[key] = value

# Convert TELEGRAM_USER_ID to int
TELEGRAM_USER_ID = required_vars["TELEGRAM_USER_ID"]
if TELEGRAM_USER_ID:
    try:
        TELEGRAM_USER_ID = int(TELEGRAM_USER_ID)
    except ValueError:
        print(f"❌ TELEGRAM_USER_ID must be a valid integer, got: {required_vars['TELEGRAM_USER_ID']}")
        sys.exit(1)

TELEGRAM_BOT_TOKEN = required_vars["TELEGRAM_BOT_TOKEN"]

# Import after checking env vars
try:
    from bot import (
        send_portfolio_summary,
        send_investment_recommendations,
        load_portfolio_from_supabase,
        get_supabase_client
    )
    from telegram.ext import Application
except ImportError as e:
    print(f"❌ Failed to import from bot.py: {e}")
    print("💡 Make sure bot.py is in the same directory and all dependencies are installed.")
    sys.exit(1)

async def test_messages():
    """Send test messages immediately"""
    print("🚀 Starting bot test...")
    print(f"📱 Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'} ({TELEGRAM_BOT_TOKEN[:10]}...)" if TELEGRAM_BOT_TOKEN else "❌ No token")
    print(f"📱 User ID: {TELEGRAM_USER_ID if TELEGRAM_USER_ID else '❌ Missing'}")
    
    # Test Supabase connection first
    print("\n🔍 Testing Supabase connection...")
    try:
        supabase_client = get_supabase_client()
        if supabase_client:
            print("✅ Supabase client created")
            print("📊 Loading portfolio from Supabase...")
            portfolio = load_portfolio_from_supabase()
            print(f"✅ Portfolio loaded: {len(portfolio)} positions")
            if portfolio:
                print("📋 Portfolio positions:")
                for pos in portfolio:
                    print(f"   - {pos['Ticker']}: {pos['Quantity']} @ ${pos['Buy Price']:.2f}")
            else:
                print("⚠️ Warning: No portfolio positions found in Supabase")
                print("   💡 Add positions in Streamlit app first, then try again")
                print("   The bot will still send a message about this.")
        else:
            print("❌ Supabase client not available (check credentials)")
            print("   💡 Verify SUPABASE_URL and SUPABASE_KEY in GitHub Secrets")
    except Exception as e:
        print(f"❌ Supabase test failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing anyway...")
    
    # Create bot application
    print("\n🤖 Creating bot application...")
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        print("✅ Bot application created")
    except Exception as e:
        print(f"❌ Failed to create bot application: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize the application
    print("🔄 Initializing bot...")
    try:
        await application.initialize()
        await application.start()
        print("✅ Bot initialized and started")
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        import traceback
        traceback.print_exc()
        return
    
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
                text=f"❌ Error in test script: {str(e)[:400]}"
            )
            print("✅ Error message sent to Telegram")
        except Exception as send_error:
            print(f"❌ Could not send error message: {send_error}")
    finally:
        try:
            print("\n🛑 Shutting down bot...")
            await application.stop()
            await application.shutdown()
            print("✅ Bot shut down cleanly")
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("TELEGRAM BOT TEST - Sending Messages NOW")
    print("=" * 60)
    print()
    asyncio.run(test_messages())

