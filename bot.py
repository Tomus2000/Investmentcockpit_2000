# -*- coding: utf-8 -*-
"""
Telegram Bot for Stock News Notifications and Portfolio Tracking
Sends daily news summaries, instant alerts, and scheduled portfolio updates
"""

import os
import asyncio
import logging
import requests
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from typing import List, Dict, Optional
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Helper function to get and clean environment variables
def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable and strip quotes/whitespace"""
    value = os.getenv(key, default)
    if value:
        return str(value).strip().strip('"').strip("'").strip()
    return default

# Configuration from environment variables (with quote stripping)
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = get_env_var("TELEGRAM_BOT_TOKEN", "")
FINNHUB_API_KEY = get_env_var("FINNHUB_API_KEY", "")

# Supabase Configuration from environment variables (with quote stripping)
SUPABASE_URL = get_env_var("SUPABASE_URL", "")
SUPABASE_KEY = get_env_var("SUPABASE_KEY", "")

# Your Telegram user ID (get this by messaging @userinfobot)
TELEGRAM_USER_ID = get_env_var("TELEGRAM_USER_ID", "")
if TELEGRAM_USER_ID:
    try:
        TELEGRAM_USER_ID = int(TELEGRAM_USER_ID)
    except ValueError:
        TELEGRAM_USER_ID = None
        logging.warning("TELEGRAM_USER_ID must be a valid integer")

# Validate required environment variables
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not found in environment variables")
if not TELEGRAM_BOT_TOKEN:
    logging.error("TELEGRAM_BOT_TOKEN not found in environment variables")
if not FINNHUB_API_KEY:
    logging.warning("FINNHUB_API_KEY not found in environment variables")
if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("Supabase credentials not found in environment variables")

# Lisbon timezone
LISBON_TZ = pytz.timezone('Europe/Lisbon')

# Portfolio tracking - Your default portfolio (fallback)
portfolio_stocks = {"PLUG", "QQQ", "VTI", "VEA", "BTC-USD", "RHM.DE"}

# Initialize Supabase client
def get_supabase_client() -> Optional[Client]:
    """Get Supabase client with proper error handling"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase credentials not configured")
        logger.error(f"URL present: {bool(SUPABASE_URL)}, KEY present: {bool(SUPABASE_KEY)}")
        return None
    try:
        # Additional cleaning (get_env_var already strips, but double-check)
        url = SUPABASE_URL.strip().strip('"').strip("'").strip()
        key = SUPABASE_KEY.strip().strip('"').strip("'").strip()
        
        # Validate URL format
        if not url.startswith("https://") or ".supabase.co" not in url:
            logger.error(f"Invalid Supabase URL format: {url[:50]}...")
            return None
        
        # Validate key format (JWT tokens start with eyJ or sb_publishable_)
        if not (key.startswith("eyJ") or key.startswith("sb_publishable_")):
            logger.warning(f"Supabase key format unexpected. Starts with: {key[:20]}...")
            logger.warning("Expected 'eyJ...' (legacy anon) or 'sb_publishable_...' (new key)")
        
        logger.info(f"Connecting to Supabase: {url[:30]}...")
        client = create_client(url, key)
        
        # Test the connection
        try:
            test_response = client.table('portfolio_positions').select('id').limit(1).execute()
            logger.info("✅ Supabase connection test successful")
        except Exception as test_error:
            error_msg = str(test_error)
            if "401" in error_msg or "Invalid API key" in error_msg or "Unauthorized" in error_msg:
                logger.error("❌ Supabase authentication failed - Invalid API key")
                logger.error("Please check your SUPABASE_KEY in environment variables")
                logger.error("💡 Try using the Legacy anon public key from Supabase Dashboard")
                return None
            else:
                # Table might not exist yet, that's OK
                logger.warning(f"Connection test warning (might be OK): {error_msg[:100]}")
        
        return client
    except Exception as e:
        logger.error(f"Could not initialize Supabase client: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

supabase: Optional[Client] = get_supabase_client()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# OpenAI API configuration (using requests instead of openai module)

def load_portfolio_from_supabase() -> List[Dict]:
    """Load portfolio positions from Supabase"""
    client = get_supabase_client()
    if not client:
        logger.warning("Supabase client not available")
        logger.warning(f"URL configured: {bool(SUPABASE_URL)}, KEY configured: {bool(SUPABASE_KEY)}")
        return []
    
    try:
        logger.info("Fetching portfolio from Supabase...")
        response = client.table('portfolio_positions').select('*').execute()
        
        logger.info(f"Supabase response received. Has data: {bool(response.data)}, Count: {len(response.data) if response.data else 0}")
        
        if response.data and len(response.data) > 0:
            portfolio = []
            for row in response.data:
                try:
                    portfolio.append({
                        "Ticker": str(row.get('ticker', '')).strip(),
                        "Buy Price": float(row.get('buy_price', 0)),
                        "Quantity": float(row.get('quantity', 0))
                    })
                except Exception as row_error:
                    logger.error(f"Error processing row: {row_error}, Row data: {row}")
            
            logger.info(f"Successfully loaded {len(portfolio)} positions from Supabase")
            return portfolio
        else:
            logger.warning("No portfolio data found in Supabase (table exists but is empty)")
            return []
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error loading portfolio from Supabase: {error_msg}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Check if it's an authentication error
        if "401" in error_msg or "Invalid API key" in error_msg or "Unauthorized" in error_msg:
            logger.error("Authentication failed - check your Supabase API key")
        elif "relation" in error_msg.lower() or "does not exist" in error_msg.lower():
            logger.warning("Table might not exist yet - this is OK for first run")
        else:
            import traceback
            logger.error(traceback.format_exc())
    
    return []

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for tickers using yfinance"""
    prices = {}
    for ticker in tickers:
        try:
            # Try multiple methods to get current price
            ticker_obj = yf.Ticker(ticker)
            
            # Method 1: Try intraday 1-minute last price
            try:
                h1m = ticker_obj.history(period="1d", interval="1m")
                if not h1m.empty:
                    prices[ticker] = float(h1m["Close"].dropna().iloc[-1])
                    continue
            except:
                pass
            
            # Method 2: Try fast_info
            try:
                price = ticker_obj.fast_info.get("last_price", None)
                if price:
                    prices[ticker] = float(price)
                    continue
            except:
                pass
            
            # Method 3: Fallback to last daily close
            try:
                h = ticker_obj.history(period="1d")
                if not h.empty:
                    prices[ticker] = float(h["Close"].iloc[-1])
                    continue
            except:
                pass
            
            # If all methods fail
            prices[ticker] = None
            logger.warning(f"Could not fetch price for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            prices[ticker] = None
    return prices

def calculate_portfolio_stats(portfolio: List[Dict], prices: Dict[str, float]) -> Dict:
    """Calculate portfolio statistics"""
    if not portfolio:
        return {}
    
    # Create DataFrame
    df = pd.DataFrame(portfolio)
    df['Current Price'] = df['Ticker'].map(prices)
    df = df[df['Current Price'].notna()]  # Remove positions with no price data
    
    if df.empty:
        return {}
    
    df['Cost Basis'] = df['Buy Price'] * df['Quantity']
    df['Market Value'] = df['Current Price'] * df['Quantity']
    df['P/L'] = df['Market Value'] - df['Cost Basis']
    df['P/L %'] = np.where(df['Cost Basis'] > 0, (df['P/L'] / df['Cost Basis']) * 100, 0)
    
    total_cost_basis = float(df['Cost Basis'].sum())
    total_market_value = float(df['Market Value'].sum())
    total_pl = float(df['P/L'].sum())
    total_pl_pct = float((total_market_value - total_cost_basis) / total_cost_basis * 100) if total_cost_basis > 0 else 0
    
    mv_sum = df['Market Value'].sum()
    df['Weight %'] = np.where(mv_sum > 0, (df['Market Value'] / mv_sum) * 100, 0.0)
    
    # Additional statistics
    best_performer = df.loc[df['P/L %'].idxmax()] if not df.empty else None
    worst_performer = df.loc[df['P/L %'].idxmin()] if not df.empty else None
    largest_position = df.loc[df['Market Value'].idxmax()] if not df.empty else None
    concentration_risk = float(df['Weight %'].max()) if not df.empty else 0
    
    return {
        'total_cost_basis': total_cost_basis,
        'total_market_value': total_market_value,
        'total_pl': total_pl,
        'total_pl_pct': total_pl_pct,
        'num_positions': len(df),
        'best_performer': best_performer,
        'worst_performer': worst_performer,
        'largest_position': largest_position,
        'concentration_risk': concentration_risk,
        'positions_df': df
    }

def format_portfolio_summary(stats: Dict) -> str:
    """Format portfolio statistics as a Telegram message"""
    if not stats:
        return "❌ Could not calculate portfolio statistics. Please check your portfolio data."
    
    message = "📊 **Portfolio Summary**\n\n"
    message += f"💰 **Total Market Value:** ${stats['total_market_value']:,.2f}\n"
    message += f"💵 **Total Cost Basis:** ${stats['total_cost_basis']:,.2f}\n"
    message += f"📈 **Total P/L:** ${stats['total_pl']:,.2f} ({stats['total_pl_pct']:+.2f}%)\n"
    message += f"📦 **Positions:** {stats['num_positions']}\n\n"
    
    message += "**Top Positions:**\n"
    df = stats['positions_df'].sort_values('Market Value', ascending=False).head(5)
    for _, row in df.iterrows():
        pl_emoji = "🟢" if row['P/L'] >= 0 else "🔴"
        message += f"{pl_emoji} **{row['Ticker']}** - ${row['Market Value']:,.2f} ({row['Weight %']:.1f}%) - {row['P/L %']:+.2f}%\n"
    
    if stats['best_performer'] is not None:
        bp = stats['best_performer']
        message += f"\n🏆 **Best Performer:** {bp['Ticker']} ({bp['P/L %']:+.2f}%)\n"
    
    if stats['worst_performer'] is not None:
        wp = stats['worst_performer']
        message += f"📉 **Worst Performer:** {wp['Ticker']} ({wp['P/L %']:+.2f}%)\n"
    
    message += f"\n⚠️ **Concentration Risk:** {stats['concentration_risk']:.1f}% (largest position)\n"
    
    return message

def get_investment_recommendations(portfolio: List[Dict], stats: Dict) -> str:
    """Get investment recommendations using OpenAI Chat API"""
    if not portfolio or not stats:
        return "❌ Could not generate recommendations. Portfolio data unavailable."
    
    # Prepare portfolio summary for AI
    portfolio_text = "Current Portfolio:\n"
    df = stats['positions_df']
    for _, row in df.iterrows():
        portfolio_text += f"- {row['Ticker']}: {row['Quantity']:.2f} shares @ ${row['Buy Price']:.2f} (Current: ${row['Current Price']:.2f}, P/L: {row['P/L %']:+.2f}%, Weight: {row['Weight %']:.1f}%)\n"
    
    portfolio_text += f"\nTotal Portfolio Value: ${stats['total_market_value']:,.2f}\n"
    portfolio_text += f"Total P/L: {stats['total_pl_pct']:+.2f}%\n"
    portfolio_text += f"Number of Positions: {stats['num_positions']}\n"
    portfolio_text += f"Concentration Risk: {stats['concentration_risk']:.1f}%\n"
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert financial advisor and portfolio analyst. Provide actionable investment recommendations based on the user's portfolio. Focus on diversification, risk management, and potential opportunities. Keep recommendations concise, specific, and actionable. Format for Telegram with emojis and clear sections."
                },
                {
                    "role": "user",
                    "content": f"Analyze this portfolio and provide investment recommendations:\n\n{portfolio_text}\n\nProvide:\n1. Overall portfolio assessment\n2. Diversification recommendations\n3. Specific buy/sell/hold recommendations for existing positions\n4. Potential new investment opportunities\n5. Risk management suggestions\n\nKeep it concise and actionable (max 800 words)."
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return f"❌ Error generating recommendations: {response.status_code}"
    except Exception as e:
        logger.error(f"Error getting investment recommendations: {e}")
        return f"❌ Error generating recommendations: {str(e)}"

class StockNewsBot:
    def __init__(self):
        self.finnhub_url = "https://finnhub.io/api/v1"
        
    def get_news(self, symbol: str, days_back: int = 1) -> List[Dict]:
        """Fetch news for a specific stock from Finnhub"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': FINNHUB_API_KEY
            }
            
            response = requests.get(f"{self.finnhub_url}/company-news", params=params)
            if response.status_code == 200:
                return response.json()[:5]  # Limit to 5 most recent articles
            return []
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def summarize_news_with_ai(self, news_articles: List[Dict], symbol: str) -> str:
        """Use OpenAI to summarize news articles"""
        if not news_articles:
            return f"📰 No recent news found for {symbol}"
        
        # Prepare news content for AI
        news_text = f"Recent news for {symbol}:\n\n"
        for i, article in enumerate(news_articles[:3], 1):  # Limit to top 3 articles
            news_text += f"{i}. {article.get('headline', 'No headline')}\n"
            news_text += f"   Summary: {article.get('summary', 'No summary available')}\n"
            news_text += f"   Source: {article.get('source', 'Unknown')}\n"
            news_text += f"   Date: {article.get('datetime', 'Unknown')}\n\n"
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a financial news analyst. Summarize stock news in a concise, informative way for a Telegram message. Focus on key developments, market impact, and investor sentiment. Keep it under 500 characters."},
                    {"role": "user", "content": news_text}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error summarizing news: {response.status_code}"
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"📰 {symbol} News Summary:\n\n" + "\n".join([f"• {article.get('headline', 'No headline')}" for article in news_articles[:3]])
    
    async def send_news_alert(self, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Send instant news alert for a specific stock"""
        if not TELEGRAM_USER_ID:
            logger.warning("TELEGRAM_USER_ID not set. Cannot send alerts.")
            return
            
        news_articles = self.get_news(symbol, days_back=1)
        if news_articles:
            summary = self.summarize_news_with_ai(news_articles, symbol)
            message = f"🚨 **Instant News Alert for {symbol}**\n\n{summary}"
            
            try:
                await context.bot.send_message(
                    chat_id=TELEGRAM_USER_ID,
                    text=message,
                    parse_mode='Markdown'
                )
                logger.info(f"Sent news alert for {symbol}")
            except Exception as e:
                logger.error(f"Failed to send news alert: {e}")
    
    async def send_daily_summary(self, context: ContextTypes.DEFAULT_TYPE):
        """Send daily news summary for all portfolio stocks"""
        if not TELEGRAM_USER_ID or not portfolio_stocks:
            return
            
        message = "📊 **Daily Portfolio News Summary**\n\n"
        
        for symbol in portfolio_stocks:
            news_articles = self.get_news(symbol, days_back=1)
            if news_articles:
                summary = self.summarize_news_with_ai(news_articles, symbol)
                message += f"**{symbol}:**\n{summary}\n\n"
            else:
                message += f"**{symbol}:** No recent news\n\n"
        
        # Add market overview
        try:
            market_news = self.get_news("AAPL", days_back=1)  # Use AAPL as market proxy
            if market_news:
                market_summary = self.summarize_news_with_ai(market_news, "Market Overview")
                message += f"**Market Overview:**\n{market_summary}"
        except:
            pass
        
        try:
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("Sent daily news summary")
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

# Bot instance
news_bot = StockNewsBot()

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 **Stock News Bot Started!**\n\n"
        "Commands:\n"
        "/add <SYMBOL> - Add stock to portfolio\n"
        "/remove <SYMBOL> - Remove stock from portfolio\n"
        "/list - Show portfolio\n"
        "/news <SYMBOL> - Get news for specific stock\n"
        "/daily - Get daily summary\n"
        "/portfolio - Get current portfolio value and statistics\n"
        "/recommendations - Get AI investment recommendations\n"
        "/test - Send portfolio summary and recommendations NOW (for testing)\n"
        "/help - Show this help\n\n"
        "⏰ **Scheduled Updates:**\n"
        "• Portfolio summary: 9:00 AM (Lisbon time)\n"
        "• Investment recommendations: 9:30 AM (Lisbon time)"
    )

async def add_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /add command"""
    if not context.args:
        await update.message.reply_text("Usage: /add <SYMBOL>")
        return
    
    symbol = context.args[0].upper()
    portfolio_stocks.add(symbol)
    
    await update.message.reply_text(f"✅ Added {symbol} to portfolio")
    
    # Send instant news alert
    await news_bot.send_news_alert(context, symbol)

async def remove_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /remove command"""
    if not context.args:
        await update.message.reply_text("Usage: /remove <SYMBOL>")
        return
    
    symbol = context.args[0].upper()
    if symbol in portfolio_stocks:
        portfolio_stocks.remove(symbol)
        await update.message.reply_text(f"❌ Removed {symbol} from portfolio")
    else:
        await update.message.reply_text(f"❌ {symbol} not in portfolio")

async def list_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /list command"""
    if portfolio_stocks:
        stocks_list = "\n".join(sorted(portfolio_stocks))
        await update.message.reply_text(f"📋 **Portfolio:**\n{stocks_list}")
    else:
        await update.message.reply_text("📋 Portfolio is empty")

async def get_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /news command"""
    if not context.args:
        await update.message.reply_text("Usage: /news <SYMBOL>")
        return
    
    symbol = context.args[0].upper()
    await news_bot.send_news_alert(context, symbol)

async def daily_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /daily command"""
    await news_bot.send_daily_summary(context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await start(update, context)

async def test_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /test command - Send portfolio summary and recommendations immediately"""
    await update.message.reply_text("🚀 Sending portfolio summary and recommendations now...")
    
    try:
        # Send portfolio summary
        await send_portfolio_summary(context)
        await asyncio.sleep(1)
        
        # Send recommendations
        await send_investment_recommendations(context)
        
        await update.message.reply_text("✅ Messages sent! Check your Telegram.")
    except Exception as e:
        logger.error(f"Error in test_now: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def send_portfolio_summary(context: ContextTypes.DEFAULT_TYPE):
    """Send portfolio summary at 9am Lisbon time"""
    if not TELEGRAM_USER_ID:
        logger.warning("TELEGRAM_USER_ID not set. Cannot send portfolio summary.")
        return
    
    try:
        logger.info("Fetching portfolio and calculating statistics...")
        portfolio = load_portfolio_from_supabase()
        
        if not portfolio:
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text="⚠️ No portfolio data found. Please add positions to your portfolio."
            )
            return
        
        # Get current prices
        tickers = [pos['Ticker'] for pos in portfolio]
        prices = fetch_current_prices(tickers)
        
        # Calculate statistics
        stats = calculate_portfolio_stats(portfolio, prices)
        
        if not stats:
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text="❌ Could not calculate portfolio statistics. Please check your portfolio data."
            )
            return
        
        # Format and send message
        message = format_portfolio_summary(stats)
        await context.bot.send_message(
            chat_id=TELEGRAM_USER_ID,
            text=message,
            parse_mode='Markdown'
        )
        logger.info("Sent portfolio summary")
    except Exception as e:
        logger.error(f"Failed to send portfolio summary: {e}")
        try:
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=f"❌ Error generating portfolio summary: {str(e)}"
            )
        except:
            pass

async def send_investment_recommendations(context: ContextTypes.DEFAULT_TYPE):
    """Send investment recommendations at 9:30am Lisbon time"""
    if not TELEGRAM_USER_ID:
        logger.warning("TELEGRAM_USER_ID not set. Cannot send recommendations.")
        return
    
    try:
        logger.info("Generating investment recommendations...")
        portfolio = load_portfolio_from_supabase()
        
        if not portfolio:
            logger.warning("No portfolio found for recommendations")
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text="⚠️ **No portfolio data found in database.**\n\n"
                     "Please add your portfolio positions in the Streamlit app first.\n\n"
                     "The portfolio will be saved to Supabase and then available for Telegram updates."
            )
            return
        
        # Get current prices
        tickers = [pos['Ticker'] for pos in portfolio]
        prices = fetch_current_prices(tickers)
        
        # Calculate statistics
        stats = calculate_portfolio_stats(portfolio, prices)
        
        if not stats:
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text="❌ Could not calculate portfolio statistics. Please check your portfolio data."
            )
            return
        
        # Get recommendations
        recommendations = get_investment_recommendations(portfolio, stats)
        
        message = "🤖 **Investment Recommendations**\n\n" + recommendations
        
        await context.bot.send_message(
            chat_id=TELEGRAM_USER_ID,
            text=message,
            parse_mode='Markdown'
        )
        logger.info("Sent investment recommendations")
    except Exception as e:
        logger.error(f"Failed to send investment recommendations: {e}")
        try:
            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=f"❌ Error generating recommendations: {str(e)}"
            )
        except:
            pass

async def schedule_daily_tasks(application: Application):
    """Schedule daily tasks at 9am and 9:30am Lisbon time"""
    last_sent_9am = None
    last_sent_930am = None
    
    # Wait for application to be ready
    await asyncio.sleep(10)
    logger.info("Scheduler started. Waiting for 9:00 AM and 9:30 AM (Lisbon time)...")
    
    while True:
        try:
            # Get current time in Lisbon
            now_lisbon = datetime.now(LISBON_TZ)
            current_date = now_lisbon.date()
            current_time = now_lisbon.time()
            
            # Target times
            target_time_9am = datetime.strptime("09:00", "%H:%M").time()
            target_time_930am = datetime.strptime("09:30", "%H:%M").time()
            
            # Log current time every hour for debugging
            if current_time.minute == 0:
                logger.info(f"Current Lisbon time: {current_time.strftime('%H:%M')} - Waiting for scheduled times...")
            
            # Check if it's 9:00 (within 1 minute window) and we haven't sent today
            if (target_time_9am.hour == current_time.hour and 
                target_time_9am.minute == current_time.minute and
                last_sent_9am != current_date):
                logger.info(f"⏰ It's 9:00 AM! Sending portfolio summary...")
                try:
                    # Create a simple context object with bot
                    class SimpleContext:
                        def __init__(self, bot):
                            self.bot = bot
                    
                    context = SimpleContext(application.bot)
                    await send_portfolio_summary(context)
                    last_sent_9am = current_date
                    logger.info(f"✅ Sent 9am portfolio summary on {current_date}")
                except Exception as e:
                    logger.error(f"❌ Failed to send 9am summary: {e}")
                # Wait 2 minutes to avoid sending multiple times
                await asyncio.sleep(120)
            
            # Check if it's 9:30 (within 1 minute window) and we haven't sent today
            if (target_time_930am.hour == current_time.hour and 
                target_time_930am.minute == current_time.minute and
                last_sent_930am != current_date):
                logger.info(f"⏰ It's 9:30 AM! Sending investment recommendations...")
                try:
                    # Create a simple context object with bot
                    class SimpleContext:
                        def __init__(self, bot):
                            self.bot = bot
                    
                    context = SimpleContext(application.bot)
                    await send_investment_recommendations(context)
                    last_sent_930am = current_date
                    logger.info(f"✅ Sent 9:30am recommendations on {current_date}")
                except Exception as e:
                    logger.error(f"❌ Failed to send 9:30am recommendations: {e}")
                # Wait 2 minutes to avoid sending multiple times
                await asyncio.sleep(120)
            
            # Check every minute
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in scheduled task loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await asyncio.sleep(60)

def main():
    """Start the bot"""
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("add", add_stock))
    application.add_handler(CommandHandler("remove", remove_stock))
    application.add_handler(CommandHandler("list", list_portfolio))
    application.add_handler(CommandHandler("news", get_news))
    application.add_handler(CommandHandler("daily", daily_summary))
    application.add_handler(CommandHandler("help", help_command))
    
    # Add command for manual portfolio summary
    async def portfolio_summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await send_portfolio_summary(context)
    
    # Add command for manual recommendations
    async def recommendations_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await send_investment_recommendations(context)
    
    application.add_handler(CommandHandler("portfolio", portfolio_summary_cmd))
    application.add_handler(CommandHandler("recommendations", recommendations_cmd))
    application.add_handler(CommandHandler("test", test_now))  # Test command to send messages now
    
    # Run the bot
    logger.info("Starting Stock News Bot with scheduled portfolio updates...")
    logger.info("Portfolio summary scheduled for 9:00 AM Lisbon time")
    logger.info("Investment recommendations scheduled for 9:30 AM Lisbon time")
    
    # Start scheduler in background thread (runs in parallel with bot)
    import threading
    def start_scheduler_thread():
        """Start scheduler in separate thread with its own event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(schedule_daily_tasks(application))
        except Exception as e:
            logger.error(f"Scheduler thread error: {e}")
        finally:
            loop.close()
    
    scheduler_thread = threading.Thread(target=start_scheduler_thread, daemon=True)
    scheduler_thread.start()
    logger.info("✅ Scheduler thread started")
    
    # Run the bot (blocking)
    logger.info("🚀 Starting bot...")
    application.run_polling()

if __name__ == '__main__':
    main()
