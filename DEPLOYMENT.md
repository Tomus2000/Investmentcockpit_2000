# Deployment Guide

This guide covers deploying the Investment Cockpit to GitHub and Streamlit Cloud.

## Prerequisites

- GitHub account
- Streamlit Cloud account (free tier available)
- All API keys ready (see [ENV_SETUP.md](ENV_SETUP.md))

## Step 1: Prepare Repository for GitHub

### 1.1 Verify .gitignore

Make sure your `.gitignore` includes:
- `.env` (local environment variables)
- `.streamlit/secrets.toml` (local Streamlit secrets)
- All sensitive files

### 1.2 Remove Any Hardcoded Secrets

✅ **Already done!** The code now uses environment variables and Streamlit secrets.

### 1.3 Commit and Push to GitHub

```bash
# Initialize git if not already done
git init

# Add all files (except those in .gitignore)
git add .

# Commit
git commit -m "Initial commit: Investment Cockpit ready for deployment"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/Investmentcockpit_2000.git

# Push to GitHub
git push -u origin main
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### 2.2 Deploy Your App

1. Click **"New app"**
2. Select your repository: `Investmentcockpit_2000`
3. Select branch: `main` (or your default branch)
4. Main file path: `app.py`
5. Click **"Deploy"**

### 2.3 Configure Secrets in Streamlit Cloud

1. Go to your app's settings (click the three dots menu)
2. Click **"Secrets"**
3. Add the following secrets:

```toml
# App Password Protection (REQUIRED!)
APP_PASSWORD = "your_secure_password_here"

# API Keys
OPENAI_API_KEY = "your_openai_api_key_here"
FINNHUB_API_KEY = "your_finnhub_api_key_here"
SUPABASE_URL = "your_supabase_url_here"
SUPABASE_KEY = "your_supabase_anon_key_here"
```

**🔒 Security Note:** Set a strong `APP_PASSWORD` to protect your app from unauthorized access!

4. Click **"Save"**

### 2.4 Verify Deployment

1. Your app will automatically redeploy after saving secrets
2. Check the app URL (e.g., `https://your-app-name.streamlit.app`)
3. You should see a password prompt
4. Enter your password to access the app
5. Verify all features work correctly

## Step 3: Local Development Setup

For local development, use the `.env` file:

1. Create a `.env` file in the root directory
2. Add your API keys and password:

```env
APP_PASSWORD=your_secure_password_here
OPENAI_API_KEY=your_openai_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
```

3. Run locally:
```bash
streamlit run app.py
```

**Note:** If `APP_PASSWORD` is not set locally, the app will show a warning but allow access (for development convenience).

## How It Works

The app supports **both** deployment methods:

- **Streamlit Cloud**: Uses `st.secrets` (configured in Streamlit Cloud dashboard)
- **Local Development**: Uses `.env` file with `python-dotenv`

The code automatically detects which method is available and uses it.

## Password Protection

The app now includes password protection:

- **Production (Streamlit Cloud)**: Set `APP_PASSWORD` in Streamlit secrets - **REQUIRED**
- **Local Development**: Set `APP_PASSWORD` in `.env` file (optional, shows warning if not set)
- Users must enter the password to access the app
- Password is stored securely in environment variables/secrets

## Security Checklist

Before deploying, ensure:

- ✅ No API keys in code
- ✅ `.env` file is in `.gitignore`
- ✅ `.streamlit/secrets.toml` is in `.gitignore`
- ✅ All secrets are configured in Streamlit Cloud
- ✅ `APP_PASSWORD` is set (strong password recommended)
- ✅ Repository is set to private (optional but recommended)

## Troubleshooting

### App fails to start on Streamlit Cloud

- Check that all required secrets are set in Streamlit Cloud
- Verify the main file path is correct (`app.py`)
- Check the logs in Streamlit Cloud dashboard

### "Secrets not found" error

- Verify secrets are set in Streamlit Cloud (Settings → Secrets)
- For local: Check `.env` file exists and has correct variable names
- Restart the app after adding secrets

### "Password not working"

- Verify `APP_PASSWORD` is set correctly in Streamlit Cloud secrets
- Check for typos in the password
- Password is case-sensitive

### Supabase connection errors

- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
- Check Supabase project is active
- Verify RLS policies allow access (see `supabase_setup.sql`)

## Updating Your App

1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Your update message"
   git push
   ```
3. Streamlit Cloud will automatically redeploy

## Telegram Bot Deployment

The Telegram bot (`bot.py`) is designed for local/server deployment, not Streamlit Cloud.

For bot deployment:
- Use a VPS (e.g., DigitalOcean, AWS EC2)
- Or use a cloud service like Railway, Render, or Heroku
- Set environment variables in your hosting platform
- Keep the bot running 24/7 for scheduled messages

See [BOT_SETUP.md](BOT_SETUP.md) for bot-specific setup.

