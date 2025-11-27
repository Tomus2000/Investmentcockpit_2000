# Investment Cockpit

A Streamlit dashboard for screening growth stocks and managing a simple portfolio input with live quotes and factor analysis.

## Setup

### 1. Environment Variables

**IMPORTANT:** Before running the app, you must set up your `.env` file with API keys.

See [ENV_SETUP.md](ENV_SETUP.md) for detailed instructions on getting all required API keys.

Create a `.env` file in the root directory with:
```env
OPENAI_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_USER_ID=your_user_id_here
FINNHUB_API_KEY=your_key_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
```

### 2. Install Dependencies

Create/activate a virtual environment (optional but recommended), then:

```powershell
pip install -r requirements.txt
```

### 3. Run the App

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Telegram Bot

The bot sends daily portfolio updates and investment recommendations. See [BOT_SETUP.md](BOT_SETUP.md) for setup instructions.

To run the bot:
```powershell
pip install -r requirements_bot.txt
python bot.py
```

## Features

- 📊 Portfolio tracking with real-time prices
- 🤖 AI-powered investment recommendations
- 📈 Stock screening and analysis
- 💾 Portfolio saved to Supabase (cloud storage)
- 📱 Telegram bot for daily updates

## Notes

- By default the app uses Yahoo Finance for prices and Finnhub for fundamentals
- Portfolio data is automatically saved to Supabase
- All API keys are stored securely in `.env` file (never commit this file!)


