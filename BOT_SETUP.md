# Telegram Stock News Bot Setup

## Quick Start

1. **Get your Telegram User ID:**
   - Message @userinfobot on Telegram
   - Copy your user ID (numbers only)

2. **Update bot.py:**
   - Open `bot.py`
   - Replace `TELEGRAM_USER_ID = None` with your actual user ID:
   ```python
   TELEGRAM_USER_ID = 123456789  # Your actual user ID
   ```

3. **Install dependencies:**
   ```powershell
   py -m pip install -r requirements_bot.txt
   ```

4. **Run the bot:**
   ```powershell
   py bot.py
   ```

## How to Use

1. **Start the bot:** Message your bot on Telegram and send `/start`

2. **Add stocks to portfolio:**
   ```
   /add AAPL
   /add TSLA
   /add MSFT
   ```

3. **Get instant news:** When you add a stock, you'll get immediate news alerts

4. **Daily summary:** Send `/daily` for news summary of all portfolio stocks

5. **Other commands:**
   - `/list` - Show your portfolio
   - `/news AAPL` - Get news for specific stock
   - `/remove AAPL` - Remove stock from portfolio

## Features

- ✅ Instant news alerts when adding stocks
- ✅ AI-powered news summaries using OpenAI
- ✅ Daily portfolio news summaries
- ✅ Real-time stock news from Finnhub
- ✅ Easy portfolio management

## Troubleshooting

- **Bot not responding:** Check your user ID is correct
- **No news:** Some stocks may have limited news coverage
- **API errors:** Check your OpenAI and Telegram bot tokens

## Auto-start (Optional)

To run the bot automatically on Windows startup:
1. Create a batch file `start_bot.bat`:
   ```batch
   cd "C:\Users\tomel\OneDrive\Dokumente\Investment Cockpit"
   py bot.py
   ```
2. Add to Windows startup folder

