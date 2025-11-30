# -*- coding: utf-8 -*-
"""INVESTMENT COCKPIT"""

import os
import requests, time
import json
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys from environment variables (using get_secret after streamlit import)
# Will be set after streamlit import
FINNHUB_API_KEY = ""
OPENAI_API_KEY = ""
SUPABASE_URL = ""
SUPABASE_KEY = ""

# Validate required environment variables (after secrets are loaded)
if not FINNHUB_API_KEY:
    print("Warning: FINNHUB_API_KEY not found in environment variables")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("Warning: Supabase credentials not found in environment variables")

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from supabase import create_client, Client

# Now that streamlit is imported, we can use get_secret
# Helper function to get secrets from either Streamlit secrets or environment variables
def get_secret(key: str, default: str = "") -> str:
    """Get secret from Streamlit secrets (cloud) or environment variables (local)"""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets'):
            secrets = st.secrets
            if secrets:
                # Method 1: Dictionary-like access using .get() (most reliable)
                try:
                    if hasattr(secrets, 'get'):
                        value = secrets.get(key, None)
                        if value:
                            # Strip quotes and whitespace
                            cleaned = str(value).strip().strip('"').strip("'")
                            if cleaned:
                                return cleaned
                except Exception:
                    pass
                
                # Method 2: Direct dictionary access
                try:
                    if hasattr(secrets, '__contains__') and key in secrets:
                        value = secrets[key]
                        if value:
                            cleaned = str(value).strip().strip('"').strip("'")
                            if cleaned:
                                return cleaned
                except Exception:
                    pass
                
                # Method 3: Direct attribute access
                try:
                    if hasattr(secrets, key):
                        value = getattr(secrets, key)
                        if value:
                            cleaned = str(value).strip().strip('"').strip("'")
                            if cleaned:
                                return cleaned
                except:
                    pass
    except Exception:
        # Silently fail and try environment variables
        pass
    
    # Fallback to environment variables (for local development)
    value = os.getenv(key, default)
    if value:
        # Strip quotes and whitespace from env vars too
        cleaned = str(value).strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return default

# API Keys - supports both Streamlit secrets and .env file
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY", "")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")

# Supabase Configuration - supports both Streamlit secrets and .env file
SUPABASE_URL = get_secret("SUPABASE_URL", "")
SUPABASE_KEY = get_secret("SUPABASE_KEY", "")

# DEBUG: Show what we're reading (only in development, remove in production)
# Uncomment these lines to debug secret reading:
# if hasattr(st, 'secrets'):
#     st.sidebar.write("🔍 DEBUG - Secrets check:")
#     st.sidebar.write("URL found:", "YES" if SUPABASE_URL else "NO")
#     st.sidebar.write("KEY found:", "YES" if SUPABASE_KEY else "NO")
#     if SUPABASE_URL:
#         st.sidebar.write("URL preview:", SUPABASE_URL[:40] + "...")
#     if SUPABASE_KEY:
#         st.sidebar.write("KEY preview:", SUPABASE_KEY[:15] + "...")

# -------------------------------------------------------
# Supabase Client Setup
# -------------------------------------------------------
@st.cache_resource
def get_supabase_client() -> Client:
    """Create and cache Supabase client"""
    # Read secrets inside the function to ensure they're available
    supabase_url = get_secret("SUPABASE_URL", "")
    supabase_key = get_secret("SUPABASE_KEY", "")
    
    # Additional cleaning (get_secret already strips, but double-check)
    if supabase_url:
        supabase_url = supabase_url.strip().strip('"').strip("'").strip()
    if supabase_key:
        supabase_key = supabase_key.strip().strip('"').strip("'").strip()
    
    if not supabase_url or not supabase_key:
        # Check if we're on Streamlit Cloud or local
        is_cloud = hasattr(st, 'secrets') and st.secrets
        if is_cloud:
            st.error("⚠️ Supabase credentials not found in Streamlit secrets. Please add SUPABASE_URL and SUPABASE_KEY in your app's Settings → Secrets.")
        else:
            st.error("⚠️ Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
        return None
    
    # Validate URL format
    if not supabase_url.startswith("https://") or ".supabase.co" not in supabase_url:
        st.error(f"⚠️ Invalid Supabase URL format. Should be: https://your-project.supabase.co")
        st.info(f"Current URL: {supabase_url[:50]}..." if len(supabase_url) > 50 else f"Current URL: {supabase_url}")
        return None
    
    # Validate key format - accept both new publishable keys (sb_publishable_) and legacy anon keys (eyJ)
    key_valid = supabase_key.startswith("eyJ") or supabase_key.startswith("sb_publishable_")
    if not key_valid:
        st.warning("⚠️ Supabase key format unexpected. Expected 'eyJ...' (legacy anon) or 'sb_publishable_...' (new key)")
        st.info("💡 Try using the **Legacy anon public key** from: Supabase Dashboard → Settings → API → Legacy anon, service_role API keys")
    
    try:
        client = create_client(supabase_url, supabase_key)
        
        # Test the connection with a simple query
        try:
            # Try a simple query to verify the key works
            test_response = client.table('portfolio_positions').select('id').limit(1).execute()
            # If we get here, the key is valid
            return client
        except Exception as auth_error:
            error_msg = str(auth_error)
            
            if "401" in error_msg or "Invalid API key" in error_msg or "Unauthorized" in error_msg or "invalid API key" in error_msg.lower():
                st.error("🔒 **Supabase Authentication Failed**")
                st.error("The Supabase API key is invalid or incorrect.")
                st.markdown("""
                **Troubleshooting Steps:**
                
                1. **Check you're using the RIGHT key type:**
                   - Go to Supabase Dashboard → Settings → API
                   - Click **"Legacy anon, service_role API keys"** tab
                   - Copy the **anon public** key (starts with `eyJ...`)
                   - NOT the service_role key
                   - NOT the new publishable key (unless your Supabase version requires it)
                
                2. **Verify in Streamlit secrets:**
                   - Make sure variable name is exactly: `SUPABASE_KEY`
                   - Make sure it's the anon public key
                   - No extra spaces or quotes
                
                3. **Test the key:**
                   - The key should start with `eyJ` (legacy) or `sb_publishable_` (new)
                   - It should be very long (100+ characters)
                """)
                return None
            else:
                # Other error, might be table doesn't exist yet (that's okay)
                return client
    except Exception as e:
        error_msg = str(e)
        
        if "401" in error_msg or "Invalid API key" in error_msg or "invalid API key" in error_msg.lower():
            st.error("🔒 **Supabase Authentication Failed**")
            st.error("Invalid API key. Please verify your SUPABASE_KEY in Streamlit secrets.")
            st.info("💡 Try using the **Legacy anon public key** from Supabase Dashboard → Settings → API")
        else:
            st.error(f"⚠️ Failed to initialize Supabase client: {e}")
        return None

# Initialize Supabase client (will be created on first use)
supabase = None

# -------------------------------------------------------
# Supabase Helper Functions
# -------------------------------------------------------
def get_portfolio_hash(positions: list) -> str:
    """Create a hash of the portfolio to detect changes"""
    sorted_positions = sorted(positions, key=lambda x: x.get('Ticker', ''))
    portfolio_str = json.dumps(sorted_positions, sort_keys=True)
    return hashlib.md5(portfolio_str.encode()).hexdigest()

def load_portfolio_from_supabase() -> list:
    """Load portfolio positions from Supabase"""
    # Get Supabase client (will read secrets if needed)
    supabase_client = get_supabase_client()
    if not supabase_client:
        return []
    try:
        response = supabase_client.table('portfolio_positions').select('*').execute()
        if response.data:
            return [{"Ticker": row['ticker'], "Buy Price": float(row['buy_price']), "Quantity": float(row['quantity'])} 
                    for row in response.data]
    except Exception as e:
        error_msg = str(e)
        # Check if table doesn't exist
        if "PGRST205" in error_msg or "Could not find the table" in error_msg or "does not exist" in error_msg.lower():
            st.error("📋 **Database Table Missing**")
            st.error("The `portfolio_positions` table doesn't exist in your Supabase database yet.")
            st.markdown("""
            **To fix this, run the SQL setup script:**
            
            1. Go to your [Supabase Dashboard](https://supabase.com/dashboard)
            2. Select your project
            3. Click **SQL Editor** in the left sidebar
            4. Click **New Query**
            5. Copy and paste the contents of `supabase_setup.sql` from this project
            6. Click **Run** (or press Ctrl+Enter)
            
            The script will create the required tables automatically.
            
            **Or** you can copy this SQL directly:
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
            
            ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
            ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;
            
            CREATE POLICY "Allow all operations on portfolio_positions" ON portfolio_positions
                FOR ALL USING (true) WITH CHECK (true);
            
            CREATE POLICY "Allow all operations on ai_recommendations" ON ai_recommendations
                FOR ALL USING (true) WITH CHECK (true);
            ```
            """)
            return []
        # Don't show error if it's just authentication (already shown in get_supabase_client)
        elif "401" not in error_msg and "Invalid API key" not in error_msg and "Unauthorized" not in error_msg:
            st.sidebar.warning(f"Could not load portfolio from Supabase: {e}")
    return []

def save_portfolio_to_supabase(positions: list, show_success: bool = False):
    """Save portfolio positions to Supabase"""
    supabase_client = get_supabase_client()
    if not supabase_client:
        # Error already shown in get_supabase_client
        return False
    try:
        # Delete all existing positions (using WHERE clause as required by Supabase)
        # Using .neq("id", 0) which matches all rows since id is bigserial starting from 1
        supabase_client.table('portfolio_positions').delete().neq("id", 0).execute()
        
        # Insert new positions
        if positions:
            rows = [{"ticker": p['Ticker'], "buy_price": float(p['Buy Price']), "quantity": float(p['Quantity'])} 
                    for p in positions]
            supabase_client.table('portfolio_positions').insert(rows).execute()
            if show_success:
                st.sidebar.success(f"✅ Saved {len(positions)} positions to Supabase!")
            return True
        else:
            # Empty portfolio - still saved (deleted all)
            if show_success:
                st.sidebar.info("✅ Cleared portfolio in Supabase")
            return True
    except Exception as e:
        error_msg = str(e)
        # Check if table doesn't exist
        if "PGRST205" in error_msg or "Could not find the table" in error_msg or "does not exist" in error_msg.lower():
            st.error("📋 **Database Table Missing**")
            st.error("The `portfolio_positions` table doesn't exist. Please run the SQL setup script in Supabase.")
            st.info("💡 See the error message above for instructions on how to create the table.")
            return False
        # Don't show error if it's just authentication (already shown in get_supabase_client)
        elif "401" not in error_msg and "Invalid API key" not in error_msg and "Unauthorized" not in error_msg:
            st.sidebar.error(f"❌ Could not save portfolio to Supabase: {e}")
            return False
        return False

def get_cached_recommendation(rec_type: str, portfolio_hash: str) -> str | None:
    """Get cached AI recommendation from Supabase"""
    supabase_client = get_supabase_client()
    if not supabase_client:
        return None
    try:
        response = supabase_client.table('ai_recommendations').select('*').eq('rec_type', rec_type).eq('portfolio_hash', portfolio_hash).order('created_at', desc=True).limit(1).execute()
        if response.data:
            return response.data[0]['content']
    except Exception as e:
        # Silently fail and regenerate
        pass
    return None

def save_recommendation_to_supabase(rec_type: str, portfolio_hash: str, content: str):
    """Save AI recommendation to Supabase"""
    supabase_client = get_supabase_client()
    if not supabase_client:
        return
    try:
        # Delete old recommendations of this type
        supabase_client.table('ai_recommendations').delete().eq('rec_type', rec_type).execute()
        
        # Insert new recommendation
        supabase_client.table('ai_recommendations').insert({
            "rec_type": rec_type,
            "portfolio_hash": portfolio_hash,
            "content": content,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        # Silently fail - caching is optional
        pass

st.set_page_config(layout="wide")

# -------------------------------------------------------
# DEBUG: Show what secrets are being read (enable to troubleshoot)
# -------------------------------------------------------
# Uncomment the block below to see what secrets are being read:
# with st.sidebar.expander("🔍 DEBUG - Secret Reading", expanded=False):
#     st.write("**Direct from st.secrets:**")
#     try:
#         if hasattr(st, 'secrets'):
#             direct_url = st.secrets.get("SUPABASE_URL", "MISSING")
#             direct_key = st.secrets.get("SUPABASE_KEY", "MISSING")
#             st.write(f"URL: `{str(direct_url)[:50]}...`" if direct_url != "MISSING" else "URL: ❌ MISSING")
#             st.write(f"KEY: `{str(direct_key)[:20]}...`" if direct_key != "MISSING" else "KEY: ❌ MISSING")
#     except Exception as e:
#         st.write(f"Error reading secrets: {e}")
#     
#     st.write("**Via get_secret() function:**")
#     st.write(f"URL: `{SUPABASE_URL[:50]}...`" if SUPABASE_URL else "URL: ❌ EMPTY")
#     st.write(f"KEY: `{SUPABASE_KEY[:20]}...`" if SUPABASE_KEY else "KEY: ❌ EMPTY")

# -------------------------------------------------------
# Password Protection
# -------------------------------------------------------
# Get password from environment/secrets
APP_PASSWORD = get_secret("APP_PASSWORD", "")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If no password is set, allow access (for development)
if not APP_PASSWORD:
    st.session_state.authenticated = True
    st.warning("⚠️ No password set. Set APP_PASSWORD in .env or Streamlit secrets for production.")

# Password protection
if not st.session_state.authenticated:
    st.title("🔒 Investment Cockpit - Access Required")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Please enter the password to continue")
        password_input = st.text_input("Password", type="password", key="password_input")
        
        if st.button("Login", type="primary", use_container_width=True):
            if password_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
        
        st.markdown("---")
        st.caption("💡 Contact the administrator if you need access")
    
    st.stop()  # Stop execution if not authenticated

# Main app content (only shown if authenticated)
st.title("📊 Investment Cockpit")
st.markdown("Portfolio Analysis, Strategy Builder & Stock Screener. Designed by Tom")

# Logo positioned in top right corner like the green X in the image
st.markdown("""
<div style='position: absolute; top: 10px; right: 10px; font-size: 24px; z-index: 1000;'>
🧙‍♂️
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Tab Navigation
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Analysis", "📋 Investment Proposal", "💡 Stock Recommendations"])

# -------------------------------------------------------
# API Key Configuration
# -------------------------------------------------------
st.sidebar.header("🔑 API Configuration")
user_openai_key = st.sidebar.text_input(
    "Enter your OpenAI API Key (optional)",
    type="password",
    help="Leave empty to use default API key. Get your key at: https://platform.openai.com/api-keys"
)

# Use user's key if provided, otherwise use default
active_openai_key = user_openai_key if user_openai_key else OPENAI_API_KEY

if not user_openai_key:
    st.sidebar.info("ℹ️ Using default API key")

# -------------------------------------------------------
# Filters for Investment Proposal tab only
# -------------------------------------------------------
# Note: min_score is used in tab3 (Stock Recommendations)
# It will be defined inside tab3 context

# -------------------------------------------------------
# === Portfolio input & overview (NEW UI) ===
# -------------------------------------------------------
st.sidebar.header("📁 Portfolio Input")

# Initialize session state for manual positions - Load from Supabase first
if "manual_positions" not in st.session_state:
    # Try to load from Supabase first
    loaded_positions = load_portfolio_from_supabase()
    if loaded_positions:
        st.session_state.manual_positions = loaded_positions
        st.session_state.portfolio_loaded_from_supabase = True
    else:
        # Default portfolio - PlugPower, Nasdaq ETF, MSCI World ETF, Asia Pacific ETF, Bitcoin, and RHM.de
        st.session_state.manual_positions = [
            {"Ticker": "PLUG", "Buy Price": 2.10, "Quantity": 2493},
            {"Ticker": "QQQ", "Buy Price": 566.67, "Quantity": 15},
            {"Ticker": "VTI", "Buy Price": 299.94, "Quantity": 63},
            {"Ticker": "VEA", "Buy Price": 60.00, "Quantity": 75},
            {"Ticker": "BTC-USD", "Buy Price": 98000, "Quantity": 0.051},
            {"Ticker": "RHM.DE", "Buy Price": 1587.50, "Quantity": 11}
        ]
        st.session_state.portfolio_loaded_from_supabase = False
        # Save default portfolio to Supabase (silently, no success message on initial load)
        save_portfolio_to_supabase(st.session_state.manual_positions, show_success=False)

# Auto-save current portfolio to Supabase if it exists but wasn't loaded from Supabase
if "portfolio_loaded_from_supabase" not in st.session_state:
    st.session_state.portfolio_loaded_from_supabase = False

# If we have positions but they weren't loaded from Supabase, offer to save them
if (st.session_state.manual_positions and 
    not st.session_state.portfolio_loaded_from_supabase and 
    "portfolio_auto_saved" not in st.session_state):
    # Try to auto-save once
    if save_portfolio_to_supabase(st.session_state.manual_positions, show_success=False):
        st.session_state.portfolio_auto_saved = True
        st.session_state.portfolio_loaded_from_supabase = True

# Track portfolio hash for detecting changes
if "portfolio_hash" not in st.session_state:
    st.session_state.portfolio_hash = get_portfolio_hash(st.session_state.manual_positions)

# Track if user requested refresh
if "force_refresh_recommendations" not in st.session_state:
    st.session_state.force_refresh_recommendations = False

def _add_manual_position(ticker: str, buy_price: float, qty: float):
    if not ticker:
        st.sidebar.warning("Please enter a ticker.")
        return
    if buy_price is None or buy_price <= 0:
        st.sidebar.warning("Buy Price must be positive.")
        return
    if qty is None or qty == 0:
        st.sidebar.warning("Quantity must be non-zero.")
        return
    # normalize ticker
    t = ticker.strip().upper()
    st.session_state.manual_positions.append({"Ticker": t, "Buy Price": float(buy_price), "Quantity": float(qty)})
    # Save to Supabase and update hash
    save_portfolio_to_supabase(st.session_state.manual_positions)
    st.session_state.portfolio_hash = get_portfolio_hash(st.session_state.manual_positions)
    st.sidebar.success(f"Added {t} x {qty} @ {buy_price:.2f}")

def _remove_last():
    if st.session_state.manual_positions:
        removed = st.session_state.manual_positions.pop()
        # Save to Supabase and update hash
        save_portfolio_to_supabase(st.session_state.manual_positions)
        st.session_state.portfolio_hash = get_portfolio_hash(st.session_state.manual_positions)
        st.sidebar.info(f"Removed last: {removed['Ticker']}")

def _clear_all():
    st.session_state.manual_positions = []
    # Save to Supabase and update hash
    save_portfolio_to_supabase(st.session_state.manual_positions)
    st.session_state.portfolio_hash = get_portfolio_hash(st.session_state.manual_positions)
    st.sidebar.info("Cleared manual positions.")

# CSV upload (kept)
uploaded_csv = st.sidebar.file_uploader(
    "Upload portfolio CSV (columns: Ticker, Buy Price, Quantity)",
    type=["csv"]
)

# Beautiful manual entry form (no table)
with st.sidebar.expander("➕ Add Position Manually", expanded=True):
    st.caption("Add a single position at a time. Use the buttons below to manage the list.")
    with st.form("manual_add_form", clear_on_submit=True):
        c1, c2 = st.columns([1,1])
        ticker = c1.text_input("Ticker", placeholder="e.g., AAPL")
        buy_price = c2.number_input("Buy Price", min_value=0.0, step=0.01, format="%.2f", value=0.00)
        qty = st.number_input("Quantity", step=1.0, format="%.2f", value=0.00)
        add_clicked = st.form_submit_button("Add to Portfolio")
        if add_clicked:
            _add_manual_position(ticker, buy_price, qty)

    # Quick actions row
    b1, b2 = st.columns(2)
    with b1:
        if st.button("↩️ Remove Last", use_container_width=True):
            _remove_last()
    with b2:
        if st.button("🗑️ Clear All", use_container_width=True):
            _clear_all()

# Compact preview of manual positions
if st.session_state.manual_positions:
    st.sidebar.markdown("**Current manual positions:**")
    for i, p in enumerate(st.session_state.manual_positions, 1):
        st.sidebar.markdown(
            f"- **{p['Ticker']}** — Qty: {p['Quantity']:.2f} @ {p['Buy Price']:.2f}"
        )
    
    # Save to Supabase button
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Save Portfolio to Supabase", use_container_width=True, type="primary"):
        success = save_portfolio_to_supabase(st.session_state.manual_positions, show_success=True)
        if success:
            st.session_state.portfolio_hash = get_portfolio_hash(st.session_state.manual_positions)

# Normalization helpers
def _clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Buy Price", "Quantity"])
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for want in ["ticker", "buy price", "quantity"]:
        if want in cols:
            mapping[cols[want]] = want.title()
        else:
            for k, v in cols.items():
                if want.replace(" ","") == k.replace(" ",""):
                    mapping[v] = want.title()
                    break
    df = df.rename(columns=mapping)
    needed = ["Ticker", "Buy Price", "Quantity"]
    for n in needed:
        if n not in df.columns: df[n] = np.nan
    df = df[needed]
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Buy Price"] = pd.to_numeric(df["Buy Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Ticker","Buy Price","Quantity"])
    df = df[df["Ticker"]!=""]
    df = df[df["Quantity"]!=0]
    return df

csv_df = None
if uploaded_csv is not None:
    try:
        csv_df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

manual_df = pd.DataFrame(st.session_state.manual_positions)
manual_clean = _clean_portfolio(manual_df)
csv_clean = _clean_portfolio(csv_df)
portfolio_input = pd.concat([csv_clean, manual_clean], ignore_index=True)

if not portfolio_input.empty:
    # aggregate duplicates (CSV + manual)
    portfolio_input = (
        portfolio_input
        .groupby("Ticker", as_index=False)
        .agg({"Buy Price":"mean","Quantity":"sum"})
    )

@st.cache_data(ttl=30, show_spinner=False)
def fetch_current_prices(tickers: list) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    prices = {}
    for t in tickers:
        p = None
        try:
            # Try intraday 1-minute last price for near real-time P&L
            h1m = yf.Ticker(t).history(period="1d", interval="1m")
            if not h1m.empty:
                p = float(h1m["Close"].dropna().iloc[-1])
        except Exception:
            p = None
        if p is None:
            try:
                # Fallback to fast_info last_price
                p = yf.Ticker(t).fast_info.get("last_price", None)
            except Exception:
                p = None
        if p is None:
            try:
                # Final fallback to last daily close
                h = yf.Ticker(t).history(period="1d")
                if not h.empty:
                    p = float(h["Close"].iloc[-1])
            except Exception:
                p = None
        prices[t] = p if p is not None else np.nan
    return pd.Series(prices, name="Current Price")

# -------------------------------------------------------
# Tab 1: Portfolio Analysis
# -------------------------------------------------------
with tab1:
    # === Portfolio Overview (main area top)  
    if not portfolio_input.empty:
        st.header("📦 Portfolio Overview")
        current_px = fetch_current_prices(portfolio_input["Ticker"].unique().tolist())
        port = portfolio_input.merge(current_px.rename_axis("Ticker").reset_index(), on="Ticker", how="left")
        port["Cost Basis"] = port["Buy Price"] * port["Quantity"]
        port["Market Value"] = port["Current Price"] * port["Quantity"]
        port["P/L"] = port["Market Value"] - port["Cost Basis"]
        port["P/L %"] = np.where(port["Cost Basis"]>0, port["P/L"]/port["Cost Basis"]*100, np.nan)

        totals = {
            "Total Cost Basis": float(port["Cost Basis"].sum()),
            "Total Market Value": float(port["Market Value"].sum()),
            "Total P/L": float(port["P/L"].sum()),
            "Total P/L %": float(
                (port["Market Value"].sum() - port["Cost Basis"].sum())/port["Cost Basis"].sum()*100
            ) if port["Cost Basis"].sum()>0 else np.nan
        }

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cost Basis", f"${totals['Total Cost Basis']:,.0f}")
        c2.metric("Total Market Value", f"${totals['Total Market Value']:,.0f}")
        c3.metric("Total P/L", f"${totals['Total P/L']:,.0f}",
                  delta=f"{totals['Total P/L %']:.2f}%" if pd.notna(totals["Total P/L %"]) else None)
        c4.metric("Positions", f"{len(port)}")

        mv_sum = port["Market Value"].sum()
        port["Weight %"] = np.where(mv_sum>0, port["Market Value"]/mv_sum*100, 0.0)

        st.subheader("🧾 Positions")
        show_cols = ["Ticker","Quantity","Buy Price","Current Price","Cost Basis","Market Value","P/L","P/L %","Weight %"]
        st.dataframe(port[show_cols].set_index("Ticker").round(2), width='stretch')

        @st.cache_data
        def _portfolio_csv(df):
            return df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Portfolio Overview (CSV)",
            data=_portfolio_csv(port.round(4)),
            file_name="portfolio_overview.csv",
            mime="text/csv"
        )

        # --- Portfolio visuals ---
        st.subheader("📊 P&L by Position")
        try:
            fig_pl = px.bar(
                port.sort_values("P/L", ascending=True),
                x="P/L", y="Ticker", orientation="h",
                color="P/L", color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                labels={"P/L":"Profit / Loss (USD)"},
            )
            fig_pl.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=420)
            st.plotly_chart(fig_pl, width='stretch', key="portfolio_pl_chart")
        except Exception:
            pass

        st.subheader("🥧 Allocation by Market Value")
        try:
            fig_alloc = px.pie(
                port, names="Ticker", values="Market Value",
                hole=0.45,
            )
            fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
            fig_alloc.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=420)
            st.plotly_chart(fig_alloc, width='stretch', key="portfolio_allocation_chart")
        except Exception:
            pass

        st.subheader("📉 Portfolio vs SPY Performance Comparison")
        
        # Time period selector as tabs
        time_periods = {
            "1D": "1d",
            "5D": "5d", 
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
            "3Y": "3y",
            "5Y": "5y"
        }
        
        # Create tabs for time period selection
        selected_period = st.radio(
            "Time Period:",
            options=list(time_periods.keys()),
            index=5,  # Default to 1Y
            horizontal=True,
            key="portfolio_period"
        )
        
        period_code = time_periods[selected_period]
        
        try:
            # Build quantity map
            qty_map = dict(zip(port["Ticker"], port["Quantity"]))
            tickers_period = list(qty_map.keys())
            price_cols = []
            price_df = pd.DataFrame()
            
            # Fetch price data for portfolio stocks
            for t in tickers_period:
                try:
                    h = yf.Ticker(t).history(period=period_code, interval="1d")["Close"].rename(t)
                    if not h.empty and not h.isna().all():
                        price_df = pd.concat([price_df, h], axis=1)
                        price_cols.append(t)
                except Exception:
                    continue
            
            if price_df.empty:
                st.info("Not enough price history to plot portfolio performance.")
            else:
                price_df = price_df.dropna(how="all").ffill()
                
                # Compute portfolio value series
                portfolio_value = pd.Series(0.0, index=price_df.index)
                for t in price_cols:
                    if t in qty_map:
                        portfolio_value = portfolio_value.add(price_df[t] * float(qty_map[t]), fill_value=0.0)

                # Benchmark SPY
                spy_data = yf.Ticker("SPY").history(period=period_code, interval="1d")["Close"]
                if spy_data.empty:
                    st.info("Could not fetch SPY data for comparison.")
                else:
                    # Align dates and forward fill missing values
                    common_dates = portfolio_value.index.intersection(spy_data.index)
                    if len(common_dates) == 0:
                        st.info("No overlapping dates between portfolio and SPY data.")
                    else:
                        portfolio_aligned = portfolio_value.loc[common_dates].dropna()
                        spy_aligned = spy_data.loc[common_dates].dropna()
                        
                        # Normalize to 100 starting value
                        if len(portfolio_aligned) > 0 and len(spy_aligned) > 0:
                            pv_norm = (portfolio_aligned / portfolio_aligned.iloc[0] * 100.0)
                            spy_norm = (spy_aligned / spy_aligned.iloc[0] * 100.0)
                            
                            # Calculate performance metrics
                            portfolio_return = ((pv_norm.iloc[-1] / pv_norm.iloc[0]) - 1) * 100 if len(pv_norm) > 1 else 0
                            spy_return = ((spy_norm.iloc[-1] / spy_norm.iloc[0]) - 1) * 100 if len(spy_norm) > 1 else 0
                            outperformance = portfolio_return - spy_return
                            
                            # Create the chart
                            fig_port = go.Figure()
                            fig_port.add_trace(go.Scatter(
                                x=pv_norm.index, 
                                y=pv_norm.values, 
                                mode='lines', 
                                name='Portfolio', 
                                line=dict(color='#1f77b4', width=2)
                            ))
                            fig_port.add_trace(go.Scatter(
                                x=spy_norm.index, 
                                y=spy_norm.values, 
                                mode='lines', 
                                name='SPY', 
                                line=dict(color='#ff7f0e', width=2)
                            ))
                            
                            title_text = f"Indexed Performance ({selected_period}) - Portfolio: {portfolio_return:.1f}% | SPY: {spy_return:.1f}% | Outperformance: {outperformance:+.1f}%"
                            
                            fig_port.update_layout(
                                title=title_text,
                                xaxis_title="Date", 
                                yaxis_title="Index Level (100 = Start)",
                                hovermode="x unified", 
                                margin=dict(l=20, r=20, t=60, b=20),
                                legend=dict(x=0.02, y=0.98),
                                height=500
                            )
                            st.plotly_chart(fig_port, width='stretch', key="portfolio_vs_spy_chart")
                        else:
                            st.info("Insufficient data for performance comparison.")
                
        except Exception as e:
            st.error(f"Error generating portfolio comparison: {str(e)}")
            st.info("Please try a different time period or check your portfolio data.")

        # === AI Portfolio Analysis ===
        st.markdown("---")
        st.header("🤖 AI Portfolio Analysis")
        
        # Refresh button for AI recommendations
        col_refresh1, col_refresh2 = st.columns([1, 4])
        with col_refresh1:
            if st.button("🔄 Refresh AI Analysis", help="Generate new AI recommendations (uses OpenAI API credits)", key="refresh_ai_analysis_tab1"):
                st.session_state.force_refresh_recommendations = True
                st.rerun()
        with col_refresh2:
            st.caption("💡 AI recommendations are cached to save costs. Click refresh to get new analysis.")
        
        def get_ai_portfolio_analysis_internal(portfolio_data, current_prices, totals):
            """Generate AI-powered portfolio analysis using OpenAI API (internal, uncached)"""
            try:
                # Prepare portfolio summary for AI
                portfolio_summary = f"""
                Portfolio Analysis Request:
                
                Current Holdings:
                """
                for _, row in portfolio_data.iterrows():
                    ticker = row['Ticker']
                    quantity = row['Quantity']
                    buy_price = row['Buy Price']
                    current_price = current_prices.get(ticker, 'N/A')
                    market_value = row['Market Value']
                    pnl = row['P/L']
                    pnl_pct = row['P/L %']
                    
                    current_price_str = f"{current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A"
                    portfolio_summary += f"""
                - {ticker}: {quantity} shares @ ${buy_price:.2f} (Current: ${current_price_str})
                  Market Value: ${market_value:,.0f}, P/L: ${pnl:,.0f} ({pnl_pct:.1f}%)
                """
                
                portfolio_summary += f"""
                
                Portfolio Totals:
                - Total Cost Basis: ${totals['Total Cost Basis']:,.0f}
                - Total Market Value: ${totals['Total Market Value']:,.0f}
                - Total P/L: ${totals['Total P/L']:,.0f} ({totals['Total P/L %']:.1f}%)
                - Number of Positions: {len(portfolio_data)}
                
                Please provide a DEEP, SOPHISTICATED portfolio analysis with the following structure:
                
                ## 🎯 EXECUTIVE SUMMARY
                Brief 2-sentence overview of portfolio health and key concerns/opportunities.
                
                ## 📊 RISK ASSESSMENT
                - **Concentration Risk**: Analyze position sizing and sector concentration
                - **Volatility Analysis**: Assess portfolio volatility vs market
                - **Correlation Risk**: Identify potential correlation risks between holdings
                - **Geographic Risk**: Analyze international exposure and currency risks
                - **Liquidity Risk**: Assess position liquidity and market cap concerns
                
                ## 🏗️ PORTFOLIO CONSTRUCTION
                - **Asset Allocation**: Breakdown by asset classes (stocks, ETFs, crypto, etc.)
                - **Sector Analysis**: Identify sector over/under-weighting vs market
                - **Style Analysis**: Growth vs value, large vs small cap exposure
                - **Geographic Exposure**: US vs international allocation
                
                ## 📈 PERFORMANCE ANALYSIS
                - **Risk-Adjusted Returns**: Sharpe ratio estimation and risk efficiency
                - **Performance Attribution**: Which positions contributed most to returns
                - **Drawdown Analysis**: Maximum potential losses and recovery periods
                - **Volatility Comparison**: Portfolio volatility vs benchmarks (SPY, QQQ)
                
                ## ⚠️ RISK FACTORS & CONCERNS
                - **Top 3 Portfolio Risks**: Specific, actionable risk concerns
                - **Position-Specific Risks**: Individual stock/asset risks
                - **Market Environment Sensitivity**: How portfolio performs in different market conditions
                
                ## 🎯 STRATEGIC RECOMMENDATIONS
                - **Immediate Actions**: 3 specific moves to make now
                - **Position Adjustments**: Specific buy/sell/rebalance recommendations
                - **Hedging Strategies**: Options, bonds, or other hedges to consider
                - **Rebalancing Schedule**: When and how to rebalance
                
                ## 🔮 MARKET OUTLOOK & POSITIONING
                - **Current Market Environment**: How portfolio fits current conditions
                - **Interest Rate Sensitivity**: Impact of rate changes on holdings
                - **Economic Cycle Positioning**: How portfolio aligns with economic cycles
                - **Alternative Scenarios**: Performance in bull/bear/sideways markets
                
                ## 💡 OPPORTUNITY ANALYSIS
                - **Underweighted Sectors**: Sectors to consider adding
                - **Geographic Opportunities**: International markets to explore
                - **Alternative Assets**: REITs, commodities, or other alternatives
                - **Tax Optimization**: Tax-loss harvesting or optimization opportunities
                
                Provide specific, actionable insights with concrete recommendations. Use financial terminology appropriately. Maximum 1200 words.
                """
                
                # Call OpenAI API using requests
                headers = {
                    "Authorization": f"Bearer {active_openai_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a senior portfolio manager and chief investment officer with 20+ years of experience at top-tier investment firms. You have deep expertise in quantitative portfolio analysis, risk management, derivatives, alternative investments, and global markets. Provide institutional-grade analysis with specific, actionable recommendations. Use sophisticated financial terminology and provide detailed reasoning for all recommendations."},
                        {"role": "user", "content": portfolio_summary}
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    return f"API Error: {response.status_code} - {response.text}"
                
            except Exception as e:
                return f"Error generating AI analysis: {str(e)}"
        
        # Generate AI analysis - use cache from Supabase unless forced refresh
        st.markdown("### 🤖 AI Portfolio Analysis")
        
        current_hash = get_portfolio_hash(st.session_state.manual_positions)
        
        # Check for cached recommendation first
        cached_analysis = None
        if not st.session_state.force_refresh_recommendations:
            cached_analysis = get_cached_recommendation('portfolio_analysis', current_hash)
        
        if cached_analysis:
            st.info("📦 Using cached analysis (portfolio unchanged). Click 'Refresh AI Analysis' for new insights.")
            ai_analysis = cached_analysis
        else:
            with st.spinner("🤖 AI is analyzing your portfolio..."):
                ai_analysis = get_ai_portfolio_analysis_internal(port, current_px.to_dict(), totals)
                # Save to Supabase cache
                save_recommendation_to_supabase('portfolio_analysis', current_hash, ai_analysis)
                st.success("✅ New AI analysis generated and cached!")
            
        st.markdown("### 📊 AI Portfolio Insights")
        st.markdown(ai_analysis)
        
        # Enhanced portfolio metrics
        st.markdown("---")
        st.markdown("### 📊 Advanced Portfolio Metrics")
        
        # Calculate additional metrics
        total_weighted_return = (port['Weight %'] * port['P/L %'] / 100).sum()
        concentration_risk = port['Weight %'].max()  # Largest position weight
        num_positions = len(port)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Best Performer", 
                     port.loc[port['P/L %'].idxmax(), 'Ticker'], 
                     f"{port['P/L %'].max():.1f}%")
        
        with col2:
            st.metric("📉 Worst Performer", 
                     port.loc[port['P/L %'].idxmin(), 'Ticker'], 
                     f"{port['P/L %'].min():.1f}%")
        
        with col3:
            st.metric("🎯 Weighted Return", 
                     f"{total_weighted_return:.2f}%")
        
        with col4:
            st.metric("⚠️ Concentration Risk", 
                     f"{concentration_risk:.1f}%", 
                     delta=f"{num_positions} positions")
        
        # Additional risk metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            largest_position = port.loc[port['Market Value'].idxmax(), 'Ticker']
            st.metric("🏆 Largest Position", largest_position)
        
        with col6:
            total_exposure = port['Market Value'].sum()
            st.metric("💰 Total Exposure", f"${total_exposure:,.0f}")
        
        with col7:
            # Calculate portfolio beta approximation (simplified)
            tech_exposure = port[port['Ticker'].isin(['QQQ', 'PLUG'])]['Weight %'].sum()
            st.metric("⚡ Tech Exposure", f"{tech_exposure:.1f}%")
        
        with col8:
            # International exposure
            intl_exposure = port[port['Ticker'].isin(['VEA', 'RHM.DE'])]['Weight %'].sum()
            st.metric("🌍 Intl Exposure", f"{intl_exposure:.1f}%")
    else:
        st.info("📊 Add portfolio positions in the sidebar to see your portfolio analysis here.")

# -------------------------------------------------------
# AI-Powered Portfolio Enhancement Recommendations
# -------------------------------------------------------
def get_ai_stock_recommendations_internal(portfolio_data, totals):
    """Generate AI-powered stock recommendations to enhance portfolio (internal, uncached)"""
    try:
        # Prepare portfolio analysis for AI
        portfolio_analysis = f"""
        Current Portfolio Analysis:
        
        Holdings:
        """
        for _, row in portfolio_data.iterrows():
            ticker = row['Ticker']
            weight = row['Weight %']
            pnl_pct = row['P/L %']
            market_value = row['Market Value']
            
            portfolio_analysis += f"""
        - {ticker}: {weight:.1f}% allocation, {pnl_pct:.1f}% P/L, ${market_value:,.0f} value
        """
        
        portfolio_analysis += f"""
        
        Portfolio Summary:
        - Total Value: ${totals['Total Market Value']:,.0f}
        - Total P/L: {totals['Total P/L %']:.1f}%
        - Number of Positions: {len(portfolio_data)}
        
        Based on this portfolio, please recommend exactly 3 specific stocks that would enhance diversification, 
        reduce risk, and improve returns. 
        
        IMPORTANT: Format each recommendation EXACTLY like this (all on ONE line):
        
        Ticker Symbol: [TICKER] Company Name: [Company Name] Sector/Industry: [Sector/Industry] Investment Thesis: [Brief investment rationale] Suggested Allocation: [X]% Risk Level: [Low/Medium/High] Time Horizon: [Short-term/Medium-term/Long-term] Key Catalysts: [Key growth drivers or catalysts]
        
        Each recommendation should be on a separate line, but all fields for that recommendation should be on the same line.
        
        For each recommendation, provide:
        
        1. **Ticker Symbol** (use widely traded US stocks)
        2. **Company Name**
        3. **Sector/Industry**
        4. **Investment Thesis** (why this stock improves the portfolio)
        5. **Suggested Allocation** (recommended portfolio weight %)
        6. **Risk Level** (Low/Medium/High)
        7. **Time Horizon** (Short-term/Medium-term/Long-term)
        8. **Key Catalysts** (specific reasons to buy now)
        
        Focus on:
        - Filling portfolio gaps (missing sectors/themes)
        - Reducing concentration risk
        - Adding defensive positions if needed
        - Including growth opportunities
        - Balancing US vs international exposure
        - Consider current market environment (2024)
        
        Provide actionable, specific recommendations with clear reasoning. Use well-known, liquid stocks.
        """
        
        # Call OpenAI API
        headers = {
            "Authorization": f"Bearer {active_openai_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a senior equity research analyst and portfolio manager with 15+ years of experience. You specialize in portfolio construction, risk management, and stock selection. Provide specific, actionable stock recommendations based on portfolio analysis. Focus on diversification, risk reduction, and return enhancement."},
                {"role": "user", "content": portfolio_analysis}
            ],
            "max_tokens": 2500,
            "temperature": 0.4
        }
        
        # Try with longer timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=90  # Increased timeout to 90 seconds
                )
                break
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return "API Error: Request timed out after multiple attempts. Please try again."
                st.warning(f"⏰ API timeout (attempt {attempt + 1}/{max_retries}). Retrying...")
                continue
            except Exception as e:
                return f"API Error: {str(e)}"
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# AI Stock Recommendations moved to Investment Proposal tab (tab2) - see tab2 section below

# -------------------------------------------------------
# AI-Selected Stock Analysis Table moved to Investment Proposal tab (tab2) - see tab2 section below

def get_ai_selected_stocks_internal(portfolio_data, totals):
    """Get AI-selected stocks for analysis table (internal, uncached)"""
    try:
        # Build detailed portfolio breakdown
        portfolio_breakdown = "Current Portfolio Holdings:\n"
        for _, row in portfolio_data.iterrows():
            ticker = row['Ticker']
            weight = row.get('Weight %', 0)
            portfolio_breakdown += f"- {ticker}: {weight:.1f}% allocation\n"
        
        # Get sectors/industries from portfolio
        sectors_in_portfolio = []
        for _, row in portfolio_data.iterrows():
            ticker = row['Ticker']
            # Common sector mappings
            if ticker in ['QQQ', 'PLUG']:
                sectors_in_portfolio.append('Technology')
            elif ticker in ['VTI', 'VEA']:
                sectors_in_portfolio.append('Broad Market ETF')
            elif ticker == 'BTC-USD':
                sectors_in_portfolio.append('Cryptocurrency')
            elif 'DE' in ticker:
                sectors_in_portfolio.append('International/European')
        
        sectors_str = ', '.join(set(sectors_in_portfolio)) if sectors_in_portfolio else 'Mixed'
        
        portfolio_summary = f"""
        Current Portfolio Analysis:
        - Total Positions: {len(portfolio_data)}
        - Total Value: ${totals['Total Market Value']:,.0f}
        - Current Holdings: {portfolio_breakdown}
        - Sectors Covered: {sectors_str}
        
        Based on this EXACT portfolio, select 10-15 specific stock tickers that would:
        1. Fill sector gaps (avoid recommending stocks in sectors already well-represented)
        2. Provide diversification beyond current holdings
        3. Complement existing positions (e.g., if heavy in tech, suggest healthcare, consumer staples, or financials)
        4. Offer quality growth or value opportunities
        5. Include some defensive positions if portfolio is too aggressive
        
        IMPORTANT: 
        - Do NOT recommend stocks that are already in the portfolio
        - Focus on sectors NOT well represented: {', '.join(set(sectors_in_portfolio)) if sectors_in_portfolio else 'any underrepresented sectors'}
        - Vary recommendations based on portfolio composition
        - Return ONLY a comma-separated list of ticker symbols (e.g., JNJ,PG,UNH,V,MA,XOM,CVX)
        - Use widely traded US stocks with good liquidity
        """
        
        headers = {
            "Authorization": f"Bearer {active_openai_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a portfolio manager. Return only ticker symbols separated by commas."},
                {"role": "user", "content": portfolio_summary}
            ],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json()
            tickers_text = result['choices'][0]['message']['content'].strip()
            # Clean up the response and extract tickers
            tickers = [t.strip().upper() for t in tickers_text.replace('\n', ',').split(',') if t.strip()]
            return tickers[:15]  # Limit to 15 stocks
        else:
            # Fallback to default stocks if AI fails
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"]
        
    except Exception:
        # Fallback to default stocks
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"]

# AI-Selected Stock Analysis section moved to tab2 below

# Portfolio Stocks – Factor Analysis section removed

# -------------------------------------------------------
# Tab 2: Strategy Builder
# -------------------------------------------------------
with tab2:
    st.header("📋 Investment Proposal")
    st.markdown("Create a personalized investment strategy tailored to your goals and risk profile")
    
    # === Investment Strategy Generator (AT THE TOP) ===
    
    # Investment amount input
    col_amount, col_risk = st.columns(2)
    
    with col_amount:
        investment_amount = st.number_input(
            "💰 Investment Amount (USD)", 
            min_value=1000, 
            value=50000, 
            step=1000,
            help="Enter how much you want to invest"
        )
    
    with col_risk:
        risk_tolerance = st.selectbox(
            "⚠️ Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive", "Very Aggressive"],
            index=1,
            help="Select your comfort level with investment risk"
        )
    
    # Investment categories
    st.subheader("📊 Investment Categories")
    category_cols = st.columns(3)
    
    with category_cols[0]:
        us_stocks = st.checkbox("🇺🇸 US Stocks", value=True)
        large_cap = st.checkbox("💼 Large Cap", value=True)
        growth = st.checkbox("📈 Growth", value=True)
    
    with category_cols[1]:
        international = st.checkbox("🌍 International", value=False)
        bonds = st.checkbox("📊 Bonds/Fixed Income", value=False)
        etfs = st.checkbox("📦 ETFs", value=True)
    
    with category_cols[2]:
        tech = st.checkbox("💻 Technology", value=True)
        real_estate = st.checkbox("🏠 Real Estate (REITs)", value=False)
        commodities = st.checkbox("🏭 Commodities", value=False)
    
    # Time horizon and goals
    st.subheader("🎯 Investment Goals")
    time_horizon = st.selectbox(
        "⏱️ Time Horizon",
        options=["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"],
        index=2
    )
    
    primary_goal = st.radio(
        "Primary Investment Goal",
        options=["Capital Preservation", "Income Generation", "Capital Growth", "Balanced"],
        index=2,
        horizontal=True
    )
    
    # Generate strategy button
    if st.button("🚀 Generate AI Investment Strategy", type="primary", use_container_width=True):
        with st.spinner("🤖 AI is crafting your personalized investment strategy..."):
            
            # Build category string
            selected_categories = []
            if us_stocks: selected_categories.append("US Stocks")
            if large_cap: selected_categories.append("Large Cap")
            if growth: selected_categories.append("Growth")
            if international: selected_categories.append("International")
            if bonds: selected_categories.append("Bonds/Fixed Income")
            if etfs: selected_categories.append("ETFs")
            if tech: selected_categories.append("Technology")
            if real_estate: selected_categories.append("Real Estate/REITs")
            if commodities: selected_categories.append("Commodities")
            
            # Prepare AI prompt
            strategy_prompt = f"""
            Investment Strategy Request:
            
            Investment Amount: ${investment_amount:,.0f}
            Risk Tolerance: {risk_tolerance}
            Time Horizon: {time_horizon}
            Primary Goal: {primary_goal}
            Selected Categories: {', '.join(selected_categories)}
            
            Create a comprehensive, personalized investment strategy with the following structure:
            
            ## 💡 INVESTMENT OVERVIEW
            Brief 2-3 sentence summary of the recommended approach.
            
            ## 🎯 RECOMMENDED INVESTMENTS (Top 10-15 Holdings)
            For each holding, provide:
            1. Ticker Symbol
            2. Name
            3. Allocation %
            4. Rationale (why this investment fits the strategy)
            
            Format each recommendation: **Ticker:** [TICKER] | **Name:** [Name] | **Allocation:** [X]% | **Rationale:** [Brief reason]
            
            Focus on quality investments that match the risk tolerance and categories selected.
            Provide actionable, specific recommendations. Use current market conditions (2024). Maximum 800 words.
            """
            
            # Call OpenAI API
            try:
                headers = {
                    "Authorization": f"Bearer {active_openai_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a certified financial planner and investment advisor with 25+ years of experience. You specialize in portfolio construction, asset allocation, and personalized investment strategies. Provide detailed, actionable investment recommendations with specific percentages and ticker symbols."},
                        {"role": "user", "content": strategy_prompt}
                    ],
                    "max_tokens": 2500,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_strategy = result['choices'][0]['message']['content']
                    
                    # Display the strategy with enhanced formatting
                    st.markdown("### 🎉 Your Personalized Investment Strategy")
                    
                    # Parse and display the strategy in a sophisticated way
                    lines = ai_strategy.split('\n')
                    current_section = None
                    recommendations = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Detect section headers
                        if line.startswith('##'):
                            current_section = line.replace('##', '').strip()
                            if current_section:
                                st.markdown(f"#### {current_section}")
                                st.markdown("---")
                        
                        # Parse investment recommendations
                        elif '**Ticker:**' in line or 'Ticker:' in line:
                            # Extract recommendation data
                            ticker_match = None
                            name_match = None
                            allocation_match = None
                            rationale_match = None
                            
                            # Try different formats
                            if '**Ticker:**' in line:
                                parts = line.split('|')
                                for part in parts:
                                    part = part.strip()
                                    if '**Ticker:**' in part:
                                        ticker_match = part.replace('**Ticker:**', '').replace('**', '').strip()
                                    elif '**Name:**' in part:
                                        name_match = part.replace('**Name:**', '').replace('**', '').strip()
                                    elif '**Allocation:**' in part:
                                        allocation_match = part.replace('**Allocation:**', '').replace('**', '').replace('%', '').strip()
                                    elif '**Rationale:**' in part:
                                        rationale_match = part.replace('**Rationale:**', '').replace('**', '').strip()
                            else:
                                # Fallback parsing
                                if 'Ticker:' in line:
                                    try:
                                        ticker_match = line.split('Ticker:')[1].split('|')[0].strip() if '|' in line else line.split('Ticker:')[1].split('Name:')[0].strip()
                                        if 'Name:' in line:
                                            name_match = line.split('Name:')[1].split('|')[0].strip() if '|' in line.split('Name:')[1] else line.split('Name:')[1].split('Allocation:')[0].strip()
                                        if 'Allocation:' in line:
                                            allocation_match = line.split('Allocation:')[1].split('%')[0].strip() if '%' in line else line.split('Allocation:')[1].split('|')[0].strip()
                                        if 'Rationale:' in line:
                                            rationale_match = line.split('Rationale:')[1].strip()
                                    except:
                                        pass
                            
                            if ticker_match:
                                recommendations.append({
                                    'ticker': ticker_match,
                                    'name': name_match or 'N/A',
                                    'allocation': allocation_match or '0',
                                    'rationale': rationale_match or 'N/A'
                                })
                        
                        # Display other content as regular markdown
                        elif not line.startswith('##') and current_section:
                            if 'INVESTMENT OVERVIEW' in current_section.upper() or 'OVERVIEW' in current_section.upper():
                                st.info(f"💡 {line}")
                            elif 'RECOMMENDED INVESTMENTS' not in current_section.upper():
                                st.markdown(f"• {line}")
                    
                    # Display recommendations in beautiful cards
                    if recommendations:
                        st.markdown("---")
                        st.markdown("### 📊 Recommended Portfolio Allocation")
                        
                        # Calculate total allocation
                        total_alloc = sum([float(r.get('allocation', 0) or 0) for r in recommendations])
                        
                        # Create allocation pie chart
                        if total_alloc > 0:
                            alloc_data = pd.DataFrame(recommendations)
                            alloc_data['allocation'] = pd.to_numeric(alloc_data['allocation'], errors='coerce').fillna(0)
                            
                            fig_alloc = px.pie(
                                alloc_data,
                                values='allocation',
                                names='ticker',
                                title="Portfolio Allocation Breakdown",
                                hole=0.4
                            )
                            fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
                            fig_alloc.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
                            st.plotly_chart(fig_alloc, use_container_width=True)
                        
                        # Display each recommendation in a beautiful card
                        st.markdown("---")
                        st.markdown("### 💼 Detailed Investment Recommendations")
                        
                        for i, rec in enumerate(recommendations, 1):
                            alloc_pct = float(rec.get('allocation', 0) or 0)
                            
                            # Create a sophisticated card layout
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        padding: 25px;
                                        border-radius: 15px;
                                        text-align: center;
                                        color: white;
                                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                                        margin-bottom: 20px;
                                    ">
                                        <h2 style="margin: 0; font-size: 2.5em; font-weight: bold;">{i}</h2>
                                        <h3 style="margin: 10px 0 5px 0; font-size: 1.3em;">{rec.get('ticker', 'N/A')}</h3>
                                        <div style="
                                            background: rgba(255,255,255,0.2);
                                            padding: 10px;
                                            border-radius: 10px;
                                            margin-top: 15px;
                                        ">
                                            <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Allocation</p>
                                            <p style="margin: 5px 0 0 0; font-size: 1.8em; font-weight: bold;">{alloc_pct:.0f}%</p>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div style="
                                        background: white;
                                        padding: 25px;
                                        border-radius: 15px;
                                        border-left: 5px solid #667eea;
                                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                        margin-bottom: 20px;
                                    ">
                                        <h3 style="margin: 0 0 10px 0; color: #333; font-size: 1.4em;">
                                            {rec.get('name', 'N/A')}
                                        </h3>
                                        <p style="margin: 15px 0; color: #666; line-height: 1.6; font-size: 1.05em;">
                                            <strong style="color: #667eea;">Investment Rationale:</strong><br>
                                            {rec.get('rationale', 'N/A')}
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Summary metrics
                        st.markdown("---")
                        st.markdown("### 📈 Strategy Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📊 Total Holdings", len(recommendations))
                        
                        with col2:
                            st.metric("💰 Total Allocation", f"{total_alloc:.0f}%")
                        
                        with col3:
                            avg_alloc = total_alloc / len(recommendations) if recommendations else 0
                            st.metric("📊 Avg. Position Size", f"{avg_alloc:.1f}%")
                        
                        with col4:
                            investment_total = investment_amount
                            st.metric("💵 Investment Amount", f"${investment_total:,.0f}")
                    
                    # Also show the raw strategy text in an expander for reference
                    with st.expander("📄 View Full Strategy Text", expanded=False):
                        st.markdown(ai_strategy)
                    
                    # Add download button
                    @st.cache_data
                    def _strategy_csv(text):
                        return text.encode("utf-8")
                    
                    st.download_button(
                        "⬇️ Download Strategy as Text",
                        data=_strategy_csv(ai_strategy),
                        file_name=f"investment_strategy_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error generating strategy: {str(e)}")

# -------------------------------------------------------
# Tab 3: Stock Recommendations
with tab3:
    st.header("💡 Stock Recommendations")
    st.markdown("AI-powered stock analysis and recommendations based on your portfolio")
    
    # === AI-Selected Stock Analysis Table (Enhanced Screener) ===
    if not portfolio_input.empty:
        st.markdown("---")
        st.header("📊 AI-Selected Stock Analysis")
        
        # Get portfolio data for analysis
        current_px = fetch_current_prices(portfolio_input["Ticker"].unique().tolist())
        port = portfolio_input.merge(current_px.rename_axis("Ticker").reset_index(), on="Ticker", how="left")
        port["Cost Basis"] = port["Buy Price"] * port["Quantity"]
        port["Market Value"] = port["Current Price"] * port["Quantity"]
        port["P/L"] = port["Market Value"] - port["Cost Basis"]
        port["P/L %"] = np.where(port["Cost Basis"]>0, port["P/L"]/port["Cost Basis"]*100, np.nan)
        
        totals = {
            "Total Cost Basis": float(port["Cost Basis"].sum()),
            "Total Market Value": float(port["Market Value"].sum()),
            "Total P/L": float(port["P/L"].sum()),
            "Total P/L %": float(
                (port["Market Value"].sum() - port["Cost Basis"].sum())/port["Cost Basis"].sum()*100
            ) if port["Cost Basis"].sum()>0 else np.nan
        }
        
        # Get AI-selected stocks - use cache from Supabase unless forced refresh
        current_hash_for_stocks = get_portfolio_hash(st.session_state.manual_positions)
        
        # Filter for Stock Recommendations tab
        min_score = st.slider("Minimum Investment Score", 1, 100, 1, key="min_score_stock_recs")
        
        # Add refresh button for AI-selected stocks
        col_refresh_stocks1, col_refresh_stocks2 = st.columns([1, 4])
        with col_refresh_stocks1:
            if st.button("🔄 Refresh Stock Selection", help="Generate new AI-selected stocks based on your current portfolio", key="refresh_stocks_tab3"):
                st.session_state.force_refresh_recommendations = True
                st.rerun()
        with col_refresh_stocks2:
            st.caption("💡 AI selects stocks based on your portfolio. Click refresh to get new selections.")
        
        cached_selected_stocks = None
        if not st.session_state.force_refresh_recommendations:
            cached_selected_stocks = get_cached_recommendation('selected_stocks', current_hash_for_stocks)
        
        if cached_selected_stocks:
            ai_selected_stocks = cached_selected_stocks.split(',')
            st.info(f"📦 Using cached stock selection ({len(ai_selected_stocks)} stocks). Click 'Refresh Stock Selection' above to get new AI recommendations based on your current portfolio.")
        else:
            with st.spinner("🤖 AI is analyzing your portfolio and selecting relevant stocks..."):
                ai_selected_stocks = get_ai_selected_stocks_internal(port, totals)
                # Save to Supabase cache
                save_recommendation_to_supabase('selected_stocks', current_hash_for_stocks, ','.join(ai_selected_stocks))
                st.success(f"✅ Generated {len(ai_selected_stocks)} AI-selected stocks based on your portfolio!")
        
        # Analysis logic (same as before but with AI-selected stocks)
        price_data = {}
        results = []
        
        finnhub_url = "https://finnhub.io/api/v1"
        def get_finnhub_json(endpoint, params):
            params['token'] = FINNHUB_API_KEY
            r = requests.get(f"{finnhub_url}/{endpoint}", params=params)
            return r.json() if r.status_code == 200 else {}
        
        with st.spinner(f"Analyzing AI-selected stocks: {', '.join(ai_selected_stocks[:5])}..."):
            for ticker in ai_selected_stocks:
                try:
                    stock = yf.Ticker(ticker)
                    hist_5y = stock.history(period="5y", interval="1d")
                    if not hist_5y.empty:
                        price_data[ticker] = hist_5y['Close']
        
                    info = stock.info
                    pe = info.get("trailingPE")
                    eps_growth = info.get("earningsQuarterlyGrowth")
                    rev_growth = info.get("revenueGrowth")
                    roe = info.get("returnOnEquity")
                    dividend_yield = info.get("dividendYield")
                    perf_12m = info.get("52WeekChange")
        
                    profile = get_finnhub_json("stock/profile2", {"symbol": ticker})
                    fundamentals = get_finnhub_json("stock/metric", {"symbol": ticker, "metric": "all"})
                    earnings = get_finnhub_json("stock/earnings", {"symbol": ticker})
        
                    pe = pe if pe is not None else fundamentals.get("metric", {}).get("peNormalizedAnnual")
                    eps_growth = eps_growth if eps_growth is not None else fundamentals.get("metric", {}).get("epsGrowth")
                    rev_growth = rev_growth if rev_growth is not None else fundamentals.get("metric", {}).get("revenueGrowthYearOverYear")
                    roe = roe if roe is not None else fundamentals.get("metric", {}).get("roe")
                    dividend_yield = dividend_yield if dividend_yield is not None else fundamentals.get("metric", {}).get("dividendYieldIndicatedAnnual")
                    perf_12m = perf_12m if perf_12m is not None else fundamentals.get("metric", {}).get("52WeekPriceReturnDaily")
                    profit_margin = fundamentals.get("metric", {}).get("netProfitMarginAnnual")
                    beta = info.get("beta") or fundamentals.get("metric", {}).get("beta")
        
                    peg = (pe / (rev_growth * 100)) if pe and rev_growth else None
        
                    history = stock.history(period="6mo", interval="1d")
                    delta = history['Close'].diff()
                    gain = delta.clip(lower=0).rolling(window=14).mean()
                    loss = -delta.clip(upper=0).rolling(window=14).mean()
                    RS = gain / loss
                    RSI = 100 - (100 / (1 + RS))
                    latest_rsi = RSI.iloc[-1] if not RSI.empty else None
        
                    try:
                        latest_earn = earnings[0]
                        actual_eps = latest_earn.get("actual")
                        estimate_eps = latest_earn.get("estimate")
                        earnings_surprise = round((actual_eps - estimate_eps) / estimate_eps * 100, 2) if actual_eps and estimate_eps else 0
                    except:
                        earnings_surprise = 0
        
                    eps_growth = max(min(eps_growth if eps_growth is not None else 0, 2), -1)
                    rev_growth = max(min(rev_growth if rev_growth is not None else 0, 2), -1)
                    roe = max(min(roe if roe is not None else 0, 2), -1)
                    perf_12m = max(min(perf_12m if perf_12m is not None else 0, 2), -1)
                    profit_margin = max(min(profit_margin if profit_margin is not None else 0, 2), -1)
        
                    earnings_surprise_score = max(min((earnings_surprise or 0) / 50, 1), -1)
                    growth_score = np.mean([rev_growth, eps_growth])
                    quality_score = np.mean([roe, profit_margin])
                    momentum_score = perf_12m
                    valuation_score = max(min((50 - pe) / 50, 1), -1) if pe else 0
        
                    raw_score = (
                        0.35 * growth_score +
                        0.2 * momentum_score +
                        0.2 * quality_score +
                        0.15 * earnings_surprise_score +
                        0.1 * valuation_score
                    )
                    investment_score = max(1, min(100, ((raw_score + 1) * 50)))
        
                    results.append({
                        "Ticker": ticker,
                        "Company": profile.get("name") or info.get("shortName", ""),
                        "Industry": profile.get("finnhubIndustry") or info.get("industry", ""),
                        "PE": pe,
                        "PEG": round(peg, 2) if peg else None,
                        "Rev Growth": rev_growth,
                        "EPS Growth": eps_growth,
                        "Earnings Surprise (%)": earnings_surprise,
                        "ROE": roe,
                        "Profit Margin (%)": round(profit_margin * 100, 2) if profit_margin not in [None, 0] else 0,
                        "Beta": round(beta, 2) if beta else None,
                        "RSI": round(latest_rsi, 2) if latest_rsi else None,
                        "12M Perf": perf_12m,
                        "Investment Score (1–100)": round(investment_score, 2),
                    })
                except Exception as e:
                    st.warning(f"Error with {ticker}: {e}")
        
        df = pd.DataFrame(results).fillna(0)
        df = df[df["Investment Score (1–100)"] >= min_score]
        
        st.subheader("📋 AI-Selected Stock Analysis Table")
        st.dataframe(df.set_index("Ticker"), width='stretch')
        
        st.markdown("""
        **📘 Investment Score Explained:**
        - **Growth (35%)**: Revenue and EPS growth (YoY).
        - **Momentum (20%)**: 12-month price performance.
        - **Quality (20%)**: Return on equity and profit margin.
        - **Earnings Momentum (15%)**: Latest earnings surprise (% vs. estimate).
        - **Valuation (10%)**: Moderate P/E rewarded (below 50).
        Scores are normalized and scaled from 1 to 100.
        """)
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="⬇️ Download AI-Selected Stock Analysis as CSV",
            data=convert_df(df),
            file_name="ai_selected_stocks_analysis.csv",
            mime="text/csv"
        )
        
        st.subheader("🔥 Interactive Heatmap of Key Metrics")
        heatmap_df = df.set_index("Ticker")[[
            "Rev Growth","EPS Growth","Earnings Surprise (%)","ROE","Profit Margin (%)","Beta","RSI","12M Perf","Investment Score (1–100)"
        ]]
        z = heatmap_df.values
        x = heatmap_df.columns.tolist()
        y = heatmap_df.index.tolist()
        fig_heatmap = ff.create_annotated_heatmap(z=z, x=x, y=y, colorscale='RdBu',
                                                  showscale=True,
                                                  annotation_text=[[f"{val:.2f}" for val in row] for row in z],
                                                  hoverinfo='z')
        fig_heatmap.update_layout(title="Key Financial Metrics per Ticker",
                                  xaxis_title="Metric", yaxis_title="Ticker",
                                  autosize=True, margin=dict(l=40,r=40,t=40,b=40))
        st.plotly_chart(fig_heatmap, width='stretch', key="correlation_heatmap")
        
        st.subheader("🏆 Investment Score by Ticker")
        fig2 = px.bar(
            df.sort_values("Investment Score (1–100)", ascending=False),
            x="Ticker", y="Investment Score (1–100)",
            color="Investment Score (1–100)", color_continuous_scale="tempo",
            title="Investment Score Ranking", labels={"Investment Score (1–100)":"Score"}
        )
        st.plotly_chart(fig2, width='stretch', key="investment_score_chart")
        
        st.subheader("📈 5-Year Price Performance")
        fig3 = go.Figure()
        for t, prices in price_data.items():
            if t not in df["Ticker"].values: continue
            fig3.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name=t))
        fig3.update_layout(title="5-Year Stock Price History", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
        st.plotly_chart(fig3, width='stretch', key="price_performance_chart")
        
        if not df.empty:
            top_growth = df.sort_values("Rev Growth", ascending=False).iloc[0]["Ticker"]
            st.success(f"📈 Best Growth: {top_growth}")



