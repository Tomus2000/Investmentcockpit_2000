# Supabase Authentication Troubleshooting

## Error: "Invalid API key" (401)

If you're getting this error, follow these steps:

### 1. Verify You're Using the Correct Key Type

**✅ Use the `anon` public key** (NOT the service_role key)

- Go to your Supabase Dashboard
- Navigate to: **Settings → API**
- Copy the **anon public** key (starts with `eyJ...`)
- **DO NOT** use the service_role key (it's for server-side only)

### 2. Check for Extra Characters

Common issues:
- ❌ Extra spaces: `" eyJ..." ` (with spaces)
- ❌ Quotes included: `"eyJ..."` (with quotes)
- ✅ Correct: `eyJ...` (no spaces, no quotes)

**Fix:** Remove all spaces and quotes from the key in your Streamlit secrets or .env file

### 3. Verify Key is Complete

JWT tokens are long. Make sure you copied the entire key:
- Should start with: `eyJ`
- Should be very long (100+ characters)
- Should not be cut off

### 4. Check Streamlit Secrets Format

In Streamlit Cloud, your secrets should look like:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Important:**
- No extra spaces around `=`
- Quotes are optional but recommended
- No trailing slashes in URL

### 5. Verify URL Format

Your Supabase URL should be:
- ✅ `https://your-project-id.supabase.co`
- ❌ `https://your-project-id.supabase.co/` (no trailing slash)
- ❌ `http://...` (must be https)

### 6. Regenerate Key if Needed

If you suspect the key is compromised or wrong:

1. Go to Supabase Dashboard → Settings → API
2. Click "Reset" next to the anon key
3. Copy the new key
4. Update it in Streamlit secrets or .env file
5. Redeploy/restart your app

### 7. Test Your Keys

You can test if your keys work by running this in Python:

```python
from supabase import create_client

url = "your_supabase_url"
key = "your_anon_key"

client = create_client(url, key)
# Try a simple query
result = client.table('portfolio_positions').select('*').limit(1).execute()
print("✅ Connection successful!")
```

## Quick Checklist

- [ ] Using `anon` public key (not service_role)
- [ ] Key starts with `eyJ`
- [ ] No extra spaces or quotes
- [ ] Key is complete (not cut off)
- [ ] URL is correct format (https://...supabase.co)
- [ ] Updated in Streamlit secrets (if on Cloud)
- [ ] Restarted app after updating secrets

## Still Not Working?

1. Double-check the key in Supabase dashboard
2. Copy it fresh (don't use old cached version)
3. Make sure you're looking at the correct Supabase project
4. Verify RLS policies allow access (see `supabase_setup.sql`)

