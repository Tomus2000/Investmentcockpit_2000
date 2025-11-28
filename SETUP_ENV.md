# Setup .env File - Quick Guide

## ✅ Required Variables

Your `.env` file MUST have these 6 variables:

```env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_USER_ID=your_user_id_here
OPENAI_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
```

## 📝 Step-by-Step Setup

### 1. Create .env file
```bash
# Windows PowerShell
New-Item -Path .env -ItemType File

# Or just create it manually in your editor
```

### 2. Copy from example
```bash
# Windows PowerShell
Copy-Item .env.example .env
```

### 3. Edit .env and add your keys
Open `.env` in a text editor and replace `your_*_here` with your actual keys.

### 4. Verify it works
```bash
python test_bot_now.py
```

## 🔍 Verify Your .env File

Run this to check:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_USER_ID', 'OPENAI_API_KEY', 'FINNHUB_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']; missing = [v for v in vars if not os.getenv(v)]; print('✅ All set!' if not missing else f'❌ Missing: {missing}')"
```

## 🚨 Common Mistakes

- ❌ File named `.env.txt` (should be just `.env`)
- ❌ File in wrong location (should be in root, same folder as `bot.py`)
- ❌ Extra spaces: `KEY = value` (should be `KEY=value`)
- ❌ Missing quotes around values with spaces (usually not needed)
- ❌ Wrong variable names (case-sensitive!)

## ✅ Quick Test

After setting up `.env`, run:
```bash
python test_bot_now.py
```

If you see errors about missing variables, check your `.env` file again.

