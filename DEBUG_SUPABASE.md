# Debug Supabase Connection

## Quick Debug Steps

### 1. Enable Debug Mode in App

In your Streamlit app, look for the sidebar checkbox:
- **"🔍 Debug Supabase Connection"**
- Check it to see what secrets are being read

### 2. Check Secret Names Match

In Streamlit Cloud secrets, make sure you have:
- `SUPABASE_URL` (not `SUPABASE_URLS` or `supabase_url`)
- `SUPABASE_KEY` (not `SUPABASE_API_KEY` or `SUPABASE_ANON_KEY`)

### 3. Use Legacy Anon Key

**IMPORTANT:** Use the **Legacy anon public key**, not the new publishable key:

1. Go to Supabase Dashboard
2. Settings → API
3. Click **"Legacy anon, service_role API keys"** tab
4. Copy the **anon public** key (starts with `eyJ...`)
5. Use that as `SUPABASE_KEY` in Streamlit secrets

### 4. Test with Simple Script

Run `test_supabase.py` to test connection:
```bash
streamlit run test_supabase.py
```

This will show you exactly what's being read and test the connection.

### 5. Verify Format

Your secrets should look like:
```toml
SUPABASE_URL = "https://mrxfkoldrtkeotkuirlp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

(Quotes are optional, code strips them automatically)

## Common Issues

- **"Invalid API key"** → Use Legacy anon key, not publishable key
- **"MISSING"** → Variable name doesn't match or not set
- **Empty values** → Quotes/whitespace issue (code handles this now)

