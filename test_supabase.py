"""
Simple test script to verify Supabase connection
Run this to test if your Supabase credentials work
"""

import streamlit as st
from supabase import create_client, Client

st.title("🔍 Supabase Connection Test")

# Try to get secrets
try:
    if hasattr(st, 'secrets'):
        url = st.secrets.get("SUPABASE_URL", "NOT_FOUND")
        key = st.secrets.get("SUPABASE_KEY", "NOT_FOUND")
        
        st.write("### Secrets from Streamlit:")
        st.write(f"**URL found:** {bool(url) and url != 'NOT_FOUND'}")
        st.write(f"**KEY found:** {bool(key) and key != 'NOT_FOUND'}")
        
        if url and url != "NOT_FOUND":
            st.write(f"**URL:** `{str(url)[:50]}...`")
        if key and key != "NOT_FOUND":
            st.write(f"**KEY prefix:** `{str(key)[:20]}...`")
        
        # Clean the values
        url_clean = str(url).strip().strip('"').strip("'") if url != "NOT_FOUND" else ""
        key_clean = str(key).strip().strip('"').strip("'") if key != "NOT_FOUND" else ""
        
        if url_clean and key_clean:
            st.write("---")
            st.write("### Testing Connection:")
            
            try:
                client: Client = create_client(url_clean, key_clean)
                st.success("✅ Client created successfully!")
                
                # Test query
                try:
                    st.write("Testing query to `portfolio_positions` table...")
                    res = client.table("portfolio_positions").select("*").limit(1).execute()
                    st.success("✅ Query successful!")
                    st.write(f"**Status:** OK")
                    st.write(f"**Rows returned:** {len(res.data) if hasattr(res, 'data') else 0}")
                    if hasattr(res, 'data') and res.data:
                        st.json(res.data[0])
                except Exception as query_error:
                    error_msg = str(query_error)
                    if "401" in error_msg or "Invalid API key" in error_msg or "invalid API key" in error_msg.lower():
                        st.error("❌ **Authentication Failed**")
                        st.error("The API key is invalid. Make sure you're using the **Legacy anon public key**")
                        st.info("""
                        **How to fix:**
                        1. Go to Supabase Dashboard → Settings → API
                        2. Click **"Legacy anon, service_role API keys"** tab
                        3. Copy the **anon public** key (starts with `eyJ...`)
                        4. Update it in Streamlit secrets
                        """)
                    else:
                        st.warning(f"⚠️ Query error (might be OK if table doesn't exist): {error_msg[:200]}")
                        
            except Exception as e:
                st.error(f"❌ Failed to create client: {e}")
                st.info("Check your URL and KEY format")
        else:
            st.error("❌ Missing URL or KEY")
            st.info("Make sure SUPABASE_URL and SUPABASE_KEY are set in Streamlit secrets")
    else:
        st.error("Streamlit secrets not available")
        st.info("This test only works on Streamlit Cloud or with secrets.toml file")
        
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

