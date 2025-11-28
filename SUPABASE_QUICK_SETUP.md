# 🚀 Quick Supabase Setup Guide

## The Problem
You're seeing this error:
```
Could not find the table 'public.portfolio_positions' in the schema cache
```

This means the database tables haven't been created yet in your Supabase project.

## The Solution (2 minutes)

### Step 1: Open Supabase Dashboard
1. Go to [https://supabase.com/dashboard](https://supabase.com/dashboard)
2. Sign in and select your project

### Step 2: Open SQL Editor
1. Click **SQL Editor** in the left sidebar
2. Click **New Query** button (top right)

### Step 3: Run the Setup SQL
1. Open the file `supabase_setup.sql` from this project
2. Copy **ALL** the SQL code
3. Paste it into the SQL Editor
4. Click **Run** (or press `Ctrl+Enter` / `Cmd+Enter`)

### Step 4: Verify
You should see:
- ✅ "Success. No rows returned" or similar success message
- The tables are now created!

### Step 5: Refresh Your App
- Go back to your Streamlit app
- Refresh the page
- The error should be gone! 🎉

## What Gets Created?

The SQL script creates:
1. **`portfolio_positions`** table - Stores your portfolio holdings
2. **`ai_recommendations`** table - Caches AI-generated recommendations
3. **Indexes** - For faster queries
4. **Row Level Security (RLS) policies** - Allows your app to read/write data

## Need Help?

If you still see errors after running the SQL:
1. Make sure you're in the correct Supabase project
2. Check that the SQL ran without errors
3. Verify your `SUPABASE_URL` and `SUPABASE_KEY` are correct in Streamlit secrets
4. Try refreshing the Streamlit app

## Alternative: Copy-Paste This SQL

If you can't find the `supabase_setup.sql` file, use this:

```sql
-- Table for storing portfolio positions
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    buy_price NUMERIC NOT NULL,
    quantity NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for caching AI recommendations
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id BIGSERIAL PRIMARY KEY,
    rec_type TEXT NOT NULL,
    portfolio_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_type_hash 
ON ai_recommendations(rec_type, portfolio_hash);

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker 
ON portfolio_positions(ticker);

-- Enable Row Level Security
ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations
CREATE POLICY "Allow all operations on portfolio_positions" ON portfolio_positions
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on ai_recommendations" ON ai_recommendations
    FOR ALL USING (true) WITH CHECK (true);
```

