# Streamlit Secrets Format Guide

## ✅ Correct Format in Streamlit Cloud

When adding secrets in Streamlit Cloud (Settings → Secrets), you can use **with or without quotes**:

### Option 1: With Quotes (Recommended)
```toml
FINNHUB_API_KEY = "d0hiea9r01qup0c6eeugd0hiea9r01qup0c6eev0"
SUPABASE_URL = "https://mrxfkoldrtkeotkuirlp.supabase.co"
TELEGRAM_BOT_TOKEN = "8419734592:AAHkiVOYrTSDuvW0POpEmE0s7bmhefxz6xE"
TELEGRAM_USER_ID = "8052172643"
OPENAI_API_KEY = "sk-proj-..."
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
APP_PASSWORD = "GetRich$OrDieTryin1999!%&"
```

### Option 2: Without Quotes (Also Works)
```toml
FINNHUB_API_KEY = d0hiea9r01qup0c6eeugd0hiea9r01qup0c6eev0
SUPABASE_URL = https://mrxfkoldrtkeotkuirlp.supabase.co
TELEGRAM_BOT_TOKEN = 8419734592:AAHkiVOYrTSDuvW0POpEmE0s7bmhefxz6xE
TELEGRAM_USER_ID = 8052172643
OPENAI_API_KEY = sk-proj-...
SUPABASE_KEY = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
APP_PASSWORD = GetRich$OrDieTryin1999!%&
```

## 🔧 What the Code Does

The code **automatically strips quotes and whitespace** from all secrets, so both formats work!

## ✅ Variable Names (Case-Sensitive!)

Make sure these exact names are used:

- `FINNHUB_API_KEY` (not `FINNHUB_KEY` or `finnhub_api_key`)
- `SUPABASE_URL` (not `SUPABASE_URLS` or `supabase_url`)
- `SUPABASE_KEY` (not `SUPABASE_API_KEY` or `supabase_key`)
- `TELEGRAM_BOT_TOKEN` (not `TELEGRAM_TOKEN` or `telegram_bot_token`)
- `TELEGRAM_USER_ID` (not `TELEGRAM_USER` or `telegram_user_id`)
- `OPENAI_API_KEY` (not `OPENAI_KEY` or `openai_api_key`)
- `APP_PASSWORD` (not `PASSWORD` or `app_password`)

## 🚨 Common Mistakes

1. **Wrong variable name**: `SUPABASE_API_KEY` instead of `SUPABASE_KEY`
2. **Extra spaces**: `SUPABASE_URL = "https://..."` (space before = is OK, but not recommended)
3. **Missing key**: Forgot to add one of the required keys
4. **Wrong key type**: Using `service_role` key instead of `anon` key for Supabase

## ✅ Verification

After adding secrets, the app will:
- Automatically strip quotes
- Show clear error messages if keys are missing
- Test Supabase connection on startup

If you see errors, check:
1. Variable names match exactly (case-sensitive!)
2. All 7 keys are present
3. Supabase key is the `anon` public key (starts with `eyJ`)

