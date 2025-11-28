# Environment Variables Checklist

## ✅ Required Variables for Bot

Make sure your `.env` file has ALL of these:

```env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_USER_ID=your_user_id_here
OPENAI_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
```

## 🔍 How to Check

### Option 1: Run the test script
```bash
python test_bot_now.py
```

It will tell you which variables are missing.

### Option 2: Check manually
```bash
# Windows PowerShell
Get-Content .env

# Or check if file exists
Test-Path .env
```

## 📝 Quick Setup

1. **Copy the example file:**
   ```bash
   copy .env.example .env
   ```

2. **Edit .env and add your actual keys**

3. **Verify all keys are set:**
   ```bash
   python test_bot_now.py
   ```

## 🚨 Common Issues

- **File not found**: Make sure `.env` is in the root directory (same folder as `bot.py`)
- **Missing variables**: Check `.env.example` for the exact variable names
- **Wrong format**: No spaces around `=`, no quotes needed (but quotes are OK)
- **Case sensitive**: Variable names must match exactly

## ✅ Verification

After setting up `.env`, run:
```bash
python test_bot_now.py
```

You should see:
- ✅ All variables found
- ✅ Bot initialized
- ✅ Messages sent

