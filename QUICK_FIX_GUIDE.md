# 🚨 QUICK FIX - Get Telegram Bot Working NOW

## ⚡ 3 Steps to Fix Everything

### Step 1: Set GitHub Secrets (5 minutes)

1. Go to: `https://github.com/YOUR_USERNAME/Investmentcockpit_2000/settings/secrets/actions`
2. Click **"New repository secret"** for each:

   **Secret Name:** `TELEGRAM_BOT_TOKEN`  
   **Value:** Your bot token (from @BotFather, no quotes)

   **Secret Name:** `TELEGRAM_USER_ID`  
   **Value:** Your user ID (from @userinfobot, just the number, no quotes)

   **Secret Name:** `OPENAI_API_KEY`  
   **Value:** Your OpenAI key (starts with `sk-proj-...`, no quotes)

   **Secret Name:** `FINNHUB_API_KEY`  
   **Value:** Your Finnhub key (no quotes)

   **Secret Name:** `SUPABASE_URL`  
   **Value:** `https://mrxfkoldrtkeotkuirlp.supabase.co` (no quotes)

   **Secret Name:** `SUPABASE_KEY`  
   **Value:** Your Supabase anon key (starts with `eyJ...`, no quotes)

### Step 2: Make Sure Supabase Tables Exist (2 minutes)

1. Go to: https://supabase.com/dashboard
2. Select your project
3. Click **SQL Editor** → **New Query**
4. Copy ALL of this SQL and paste it:

```sql
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    buy_price NUMERIC NOT NULL,
    quantity NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_recommendations (
    id BIGSERIAL PRIMARY KEY,
    rec_type TEXT NOT NULL,
    portfolio_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_recommendations_type_hash 
ON ai_recommendations(rec_type, portfolio_hash);

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker 
ON portfolio_positions(ticker);

ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all operations on portfolio_positions" ON portfolio_positions
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on ai_recommendations" ON ai_recommendations
    FOR ALL USING (true) WITH CHECK (true);
```

5. Click **Run** (or press Ctrl+Enter)
6. Should see "Success" message

### Step 3: Add Portfolio & Test (3 minutes)

1. **Add Portfolio in Streamlit:**
   - Open your Streamlit app
   - In sidebar, add positions manually or upload CSV
   - Portfolio saves to Supabase automatically

2. **Test Bot via GitHub Actions:**
   - Go to: `https://github.com/YOUR_USERNAME/Investmentcockpit_2000/actions`
   - Click **"Send Telegram Messages Now"** workflow
   - Click **"Run workflow"** button (top right)
   - Click the green **"Run workflow"** button
   - Wait 30-60 seconds
   - **Check your Telegram** - you should get messages!

## ✅ Success Checklist

- [ ] All 6 secrets added in GitHub
- [ ] Supabase tables created (ran SQL)
- [ ] Portfolio positions added in Streamlit
- [ ] GitHub Actions workflow ran successfully
- [ ] Received Telegram messages ✅

## 🆘 Still Not Working?

**Check the GitHub Actions logs:**
1. Click on the workflow run
2. Click on the job that failed
3. Expand the "Send Portfolio Summary and Recommendations" step
4. Look for error messages in red
5. Common errors:
   - "Missing required environment variables" → Secrets not set correctly
   - "Invalid API key" → Wrong Supabase key (use Legacy anon key)
   - "No portfolio data found" → Add positions in Streamlit first
   - "Failed to create bot application" → Wrong Telegram bot token

## 📱 What Messages You Should Receive

1. **Portfolio Summary** - Shows your portfolio value, P/L, top positions
2. **Investment Recommendations** - AI-generated recommendations based on your portfolio

If you get "No portfolio data found" message, that means:
- Bot is working ✅
- But no portfolio in Supabase yet
- **Solution:** Add positions in Streamlit app first!

