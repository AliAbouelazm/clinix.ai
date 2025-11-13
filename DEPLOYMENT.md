# Deploying clinix.ai to Streamlit Cloud

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `ai-clinic-layer` (or `clinix-ai`)
3. Make it **Public** (required for free Streamlit Cloud)
4. **DO NOT** initialize with README, .gitignore, or license
5. Click "Create repository"

## Step 2: Push to GitHub

Run these commands (replace `YOUR_USERNAME` with your GitHub username):

```bash
cd /Users/aliabouelazm/Desktop/projects/ai_clinic_layer
git remote add origin https://github.com/YOUR_USERNAME/ai-clinic-layer.git
git branch -M main
git push -u origin main
```

Or if you prefer SSH:
```bash
git remote add origin git@github.com:YOUR_USERNAME/ai-clinic-layer.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account if not already connected
4. Select repository: `ai-clinic-layer` (or whatever you named it)
5. Main file path: `src/app/streamlit_app.py`
6. Click "Deploy"

## Step 4: Configure Environment Variables (Optional)

If you want to use LLM APIs, add these in Streamlit Cloud settings:
- `LLM_PROVIDER` = `openai` or `anthropic`
- `OPENAI_API_KEY` = your key (if using OpenAI)
- `ANTHROPIC_API_KEY` = your key (if using Anthropic)

## Note

The model will need to be trained on Streamlit Cloud. You can either:
- Add a button in the app to train the model
- Or commit the trained model (not recommended for large files)
- Or train it automatically on first run

The app will work with the mock parser if no API keys are provided.
