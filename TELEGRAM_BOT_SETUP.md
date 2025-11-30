# 🚀 Telegram Bot Setup Guide - GET IT WORKING NOW

## ✅ What You Need to Do

### Step 1: Set GitHub Secrets (REQUIRED)

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets (EXACT names, case-sensitive):

1. `TELEGRAM_BOT_TOKEN` - Your bot token from @BotFather
2. `TELEGRAM_USER_ID` - Your user ID (number only, from @userinfobot)
3. `OPENAI_API_KEY` - Your OpenAI API key
4. `FINNHUB_API_KEY` - Your Finnhub API key
5. `SUPABASE_URL` - Your Supabase project URL
6. `SUPABASE_KEY` - Your Supabase anon public key

**IMPORTANT:** 
- No quotes around values
- No extra spaces
- Copy-paste exactly as shown in Supabase/Telegram

### Step 2: Make Sure Supabase Tables Exist

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Click **SQL Editor** → **New Query**
4. Copy and paste the entire contents of `supabase_setup.sql`
5. Click **Run**
6. You should see "Success" message

### Step 3: Add Portfolio Data

1. Open your Streamlit app
2. Go to the sidebar
3. Add your portfolio positions manually or upload CSV
4. The portfolio will be saved to Supabase automatically

### Step 4: Test the Bot

**Option A: Test via GitHub Actions (Recommended)**

1. Go to your GitHub repository
2. Click **Actions** tab
3. Click **Send Telegram Messages Now** workflow
4. Click **Run workflow** button (top right)
5. Click the green **Run workflow** button
6. Wait for it to complete
7. Check your Telegram - you should receive messages!

**Option B: Test Locally**

1. Make sure you have a `.env` file with all the keys
2. Run: `python test_bot_now.py`
3. Check your Telegram

## 🔍 Troubleshooting

### "Missing environment variables" error
- Check GitHub Secrets are set correctly
- Make sure secret names match EXACTLY (case-sensitive)
- No quotes in the secret values

### "No portfolio data found" message
- Add positions in Streamlit app first
- Make sure Supabase tables exist (run SQL setup)
- Check Supabase connection in Streamlit (enable debug mode)

### "Invalid API key" error
- For Supabase: Use the **Legacy anon public key** (starts with `eyJ...`)
- Get it from: Supabase Dashboard → Settings → API → Legacy anon, service_role API keys
- For Telegram: Make sure token is correct (from @BotFather)

### Bot not sending messages
- Check TELEGRAM_USER_ID is correct (must be a number)
- Message @userinfobot on Telegram to get your user ID
- Make sure you've started the bot with `/start` command first

## 📋 Quick Checklist

- [ ] All 6 GitHub Secrets are set
- [ ] Supabase tables created (ran SQL setup)
- [ ] Portfolio positions added in Streamlit app
- [ ] Tested via GitHub Actions workflow
- [ ] Received Telegram messages ✅

## 🆘 Still Not Working?

1. Check GitHub Actions logs (click on the failed workflow run)
2. Look for error messages in red
3. Copy the error and check what it says
4. Common issues:
   - Missing secrets → Add them in GitHub Settings
   - Wrong Supabase key → Use Legacy anon key
   - No portfolio data → Add positions in Streamlit first
   - Invalid bot token → Get new token from @BotFather

