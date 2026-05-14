# EchoAI – Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended – Free)

### Step 1 — Push to GitHub
```bash
cd Soukhya_project
git init
git add .
git commit -m "EchoAI initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/echoai.git
git push -u origin main
```

> If `data/` or `models/` files are > 100 MB, install Git LFS first:
> ```bash
> git lfs install
> git lfs track "models/*.pkl" "data/*.xlsx"
> git add .gitattributes
> ```

### Step 2 — Deploy
1. Go to **https://share.streamlit.io**
2. Click **New app**
3. Select your repo → branch `main` → main file `app.py`
4. Click **Deploy**

Live URL: `https://<your-app>.streamlit.app`

The classifier, dashboard, and network views are all inside **one app** — no separate API needed.

---

## Option 2: Hugging Face Spaces (Free, More RAM)

1. Create a Space at https://huggingface.co/new-space
2. Choose **Streamlit** SDK
3. Push the repo to the Space's git remote
4. Done — public URL provided automatically

---

## Option 3: Render.com

Add a `render.yaml`:
```yaml
services:
  - type: web
    name: echoai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## Files added for hosting
| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Dark theme + server settings |
| `runtime.txt` | Forces Python 3.11 on hosting platforms |
| `packages.txt` | System packages (apt) for Streamlit Cloud |
| `.gitignore` | Excludes local-only files |

---

## What the hosted app includes (single deployment)
- **Overview** page with KPIs and charts
- **🎯 Classifier** page (replaces the local API + index.html)
- **Sentiment, Topics, Clustering, Network, Polarization, Data Explorer** pages
- All run from a single `app.py` — no FastAPI server needed in the cloud
