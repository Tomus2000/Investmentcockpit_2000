-- Supabase Setup for Investment Cockpit
-- Run this SQL in your Supabase SQL Editor (https://supabase.com/dashboard)

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
    rec_type TEXT NOT NULL,  -- 'portfolio_analysis', 'stock_recommendations', 'selected_stocks'
    portfolio_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS) - Optional but recommended
-- For now, we'll allow all operations with the anon key

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_type_hash 
ON ai_recommendations(rec_type, portfolio_hash);

CREATE INDEX IF NOT EXISTS idx_portfolio_positions_ticker 
ON portfolio_positions(ticker);

-- Allow public access for the anon key (required for the app to work)
ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations
CREATE POLICY "Allow all operations on portfolio_positions" ON portfolio_positions
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on ai_recommendations" ON ai_recommendations
    FOR ALL USING (true) WITH CHECK (true);


