# API Keys & Secrets Summary

This document lists all API keys and configuration secrets required for the Investment Cockpit project.

## 📋 Complete List of Required API Keys

### 1. **OPENAI_API_KEY**
- **Used in:** `app.py`, `bot.py`
- **Purpose:** AI-powered investment recommendations and news summarization
- **Where to get it:** https://platform.openai.com/api-keys
- **Required for:** 
  - Portfolio analysis and recommendations
  - News summarization in Telegram bot
  - Investment proposal generation

### 2. **FINNHUB_API_KEY**
- **Used in:** `app.py`, `bot.py`
- **Purpose:** Stock fundamentals, company news, and financial data
- **Where to get it:** https://finnhub.io/ (free tier available)
- **Required for:**
  - Stock screening and analysis
  - Company fundamentals
  - News fetching in Telegram bot

### 3. **SUPABASE_URL**
- **Used in:** `app.py`, `bot.py`
- **Purpose:** Database connection URL for Supabase
- **Where to get it:** Supabase Dashboard → Settings → API → Project URL
- **Format:** `https://your-project-id.supabase.co`
- **Required for:**
  - Portfolio data storage
  - AI recommendation caching

### 4. **SUPABASE_KEY**
- **Used in:** `app.py`, `bot.py`
- **Purpose:** Supabase anonymous/public API key for database access
- **Where to get it:** Supabase Dashboard → Settings → API → `anon` public key
- **Required for:**
  - Portfolio data storage
  - AI recommendation caching

### 5. **TELEGRAM_BOT_TOKEN** (Bot only)
- **Used in:** `bot.py` only
- **Purpose:** Authentication for Telegram Bot API
- **Where to get it:** 
  1. Message @BotFather on Telegram
  2. Use `/newbot` command
  3. Follow instructions to create bot
  4. Copy the token provided
- **Required for:**
  - Sending portfolio updates
  - Sending investment recommendations
  - Daily scheduled messages

### 6. **TELEGRAM_USER_ID** (Bot only)
- **Used in:** `bot.py` only
- **Purpose:** Your Telegram user ID to receive bot messages
- **Where to get it:**
  1. Message @userinfobot on Telegram
  2. It will reply with your user ID (numbers only)
- **Required for:**
  - Receiving bot notifications
  - Portfolio summary messages
  - Investment recommendations

## 📝 Environment Variables Template

Create a `.env` file in the root directory with:

```env
# OpenAI API Key (Required for app.py and bot.py)
OPENAI_API_KEY=sk-proj-...

# Finnhub API Key (Required for app.py and bot.py)
FINNHUB_API_KEY=d0hiea9r01qup0c6eeugd0hiea9r01qup0c6eev0

# Supabase Configuration (Required for app.py and bot.py)
SUPABASE_URL=https://uniuqxphvoxhmowkrret.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Telegram Bot Configuration (Required for bot.py only)
TELEGRAM_BOT_TOKEN=8419734592:AAHkiVOYrTSDuvW0POpEmE0s7bmhefxz6xE
TELEGRAM_USER_ID=8052172643
```

## 🔍 Usage by File

### `app.py` (Streamlit Dashboard)
Requires:
- ✅ `OPENAI_API_KEY` - For AI recommendations
- ✅ `FINNHUB_API_KEY` - For stock data
- ✅ `SUPABASE_URL` - For portfolio storage
- ✅ `SUPABASE_KEY` - For portfolio storage
- ❌ `TELEGRAM_BOT_TOKEN` - Not needed
- ❌ `TELEGRAM_USER_ID` - Not needed

### `bot.py` (Telegram Bot)
Requires:
- ✅ `OPENAI_API_KEY` - For investment recommendations
- ✅ `FINNHUB_API_KEY` - For news fetching
- ✅ `SUPABASE_URL` - For portfolio loading
- ✅ `SUPABASE_KEY` - For portfolio loading
- ✅ `TELEGRAM_BOT_TOKEN` - For bot authentication
- ✅ `TELEGRAM_USER_ID` - For sending messages

## 🔐 Security Notes

1. **Never commit these keys to GitHub**
   - `.env` is in `.gitignore`
   - `secrets.toml` is in `.gitignore`

2. **For Streamlit Cloud deployment:**
   - Add all keys in Streamlit Cloud dashboard → Settings → Secrets
   - Use the same variable names

3. **For local development:**
   - Create `.env` file in root directory
   - Add all required keys

4. **Key Rotation:**
   - If keys are accidentally exposed, rotate them immediately
   - Update both `.env` and Streamlit Cloud secrets

## 📊 Priority Levels

### Critical (App won't work without):
- `OPENAI_API_KEY` - Required for AI features
- `SUPABASE_URL` - Required for portfolio storage
- `SUPABASE_KEY` - Required for portfolio storage

### Important (Features will be limited):
- `FINNHUB_API_KEY` - Some stock data features won't work
- `TELEGRAM_BOT_TOKEN` - Bot won't function
- `TELEGRAM_USER_ID` - Bot can't send messages

## ✅ Verification Checklist

After setting up your `.env` file:

- [ ] `OPENAI_API_KEY` is set
- [ ] `FINNHUB_API_KEY` is set
- [ ] `SUPABASE_URL` is set (starts with `https://`)
- [ ] `SUPABASE_KEY` is set (JWT token format)
- [ ] `TELEGRAM_BOT_TOKEN` is set (if using bot)
- [ ] `TELEGRAM_USER_ID` is set (if using bot)
- [ ] `.env` file is NOT committed to git
- [ ] All keys are valid and active

## 🆘 Troubleshooting

**"API key not found" warning:**
- Check `.env` file exists in root directory
- Verify variable names match exactly (case-sensitive)
- Restart the app after adding keys

**"Supabase connection failed":**
- Verify `SUPABASE_URL` is correct (no trailing slash)
- Check `SUPABASE_KEY` is the `anon` public key (not service role key)
- Ensure Supabase project is active

**"Telegram bot not working":**
- Verify bot token is correct
- Check user ID is a valid integer
- Ensure bot is started with `/start` command first

