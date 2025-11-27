# Environment Variables Setup

This project uses environment variables to securely store API keys and configuration. Never commit your `.env` file to version control!

## Setup Instructions

1. **Create a `.env` file** in the root directory of the project

2. **Add the following variables** to your `.env` file:

```env
# App Password Protection (REQUIRED for production deployment)
APP_PASSWORD=your_secure_password_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_USER_ID=your_telegram_user_id_here

# Finnhub API Key
FINNHUB_API_KEY=your_finnhub_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

## Getting Your API Keys

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Create a new API key
4. Copy the key to your `.env` file

### Telegram Bot Token
1. Message @BotFather on Telegram
2. Use `/newbot` command to create a bot
3. Follow the instructions and get your bot token
4. Copy the token to your `.env` file

### Telegram User ID
1. Message @userinfobot on Telegram
2. It will reply with your user ID (numbers only)
3. Copy the ID to your `.env` file

### Finnhub API Key
1. Go to https://finnhub.io/
2. Sign up for a free account
3. Get your API key from the dashboard
4. Copy the key to your `.env` file

### Supabase Credentials
1. Go to https://supabase.com/
2. Create a new project or use existing one
3. Go to Settings > API
4. Copy the Project URL to `SUPABASE_URL`
5. Copy the `anon` public key to `SUPABASE_KEY`

## Security Notes

- ✅ The `.env` file is already in `.gitignore` - it will NOT be committed
- ✅ Never share your `.env` file or API keys
- ✅ If you accidentally commit API keys, rotate them immediately
- ✅ Use different API keys for development and production

## Verification

After setting up your `.env` file, you can verify it's working by:

1. Running the app: `streamlit run app.py`
2. Check the console for any warnings about missing environment variables
3. If you see warnings, double-check your `.env` file spelling and values

