# How to Force Streamlit Cloud to Reload (Fix Cached Code)

## Signs You're Using Cached Code:
- Version number hasn't updated (still shows old version)
- Severity is wrong (e.g., showing 5.0 instead of 6.8 for "significant bleeding")
- Risk scores don't match expected values
- Debug info shows old behavior

## Methods to Force Reload:

### Method 1: Manual Restart (Most Reliable)
1. Go to https://share.streamlit.io/
2. Find your app
3. Click the **three dots (⋮)** next to your app
4. Click **"Reboot app"** or **"Restart app"**
5. Wait 30-60 seconds
6. Refresh your browser

### Method 2: Clear Browser Cache
1. Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. Or clear browser cache completely
3. Close and reopen browser

### Method 3: Wait for Auto-Reload
- Streamlit Cloud usually auto-reloads within 2-5 minutes of a git push
- Check the version number - if it's updated, the code has reloaded

### Method 4: Make a Dummy Change
If restart doesn't work, I can add a timestamp or change a comment to force a file hash change.

## Verify It Worked:
1. Check version number - should show "Version 4.3.1"
2. Test with: "significant bleeding from a cut that won't stop"
3. Check Debug Info:
   - Severity should be **6.8/10** (not 5.0)
   - Risk should be **~52%** (not 21%)

## If Still Not Working:
The code includes `importlib.reload()` to force Python to reload modules. If it's still cached after restart, there might be a deeper caching issue. In that case, we may need to:
- Add a `.streamlit/config.toml` with cache settings
- Or rename the module files to force reload

