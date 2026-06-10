# Deploying the Football Predictor

The app is a single Docker image: a multi-stage build compiles the React
frontend, then a FastAPI backend serves both the API and the built frontend on
one port (`$PORT`). Pick **one** of the two paths below.

---

## Option A — Cloud host (stable URL, runs 24/7 even when your PC is off)

Best when you want a link that always works. Uses the `Dockerfile` in this repo.

**Prerequisites:** a GitHub account and a host account (Render shown here).

1. Push this repo to GitHub:
   ```bash
   git remote add origin https://github.com/<you>/football-predictor.git
   git push -u origin main
   ```
2. On **render.com** → *New* → *Blueprint* → connect the repo. Render reads
   `render.yaml` and builds the Dockerfile automatically.
   - Or *New* → *Web Service* → *Docker* → pick the repo (no blueprint needed).
3. (Optional) Set env var `API_FOOTBALL_KEY` to enable the live club "Today" tab.
4. You get a stable URL like `https://football-predictor.onrender.com`.

**Memory note:** the ML models (XGBoost + ~50k internationals + league models)
are memory-hungry. Render's **free** tier (512 MB) may OOM once several models
are cached; the **starter** tier (or Railway/Fly with ≥1 GB) is safer.
Free tiers also cold-start (first hit after idle is slow).

Equivalent hosts using the same Dockerfile: **Railway**, **Fly.io**,
**Google Cloud Run**.

---

## Option B — Cloudflare Named Tunnel (free, runs on your PC, no memory limits)

Best when you're fine with the app running on your own machine (it must be on
to serve traffic) and you want to avoid cloud memory limits. Gives a stable
hostname on a domain you control in Cloudflare.

**Prerequisites:** a free Cloudflare account and a domain added to it.

```bash
cloudflared login                                  # opens browser to authorize
cloudflared tunnel create football-predictor       # creates a named tunnel
cloudflared tunnel route dns football-predictor predictor.yourdomain.com
cloudflared tunnel run --url http://localhost:8000 football-predictor
```

Keep the backend running locally (`uvicorn main:app --port 8000`) and the
tunnel will serve it at `https://predictor.yourdomain.com` reliably (unlike the
throwaway `trycloudflare.com` quick tunnels).

---

## Local (no deploy)

```bash
# backend (also serves the built frontend at http://localhost:8000)
cd backend && pip install -r requirements.txt && uvicorn main:app --port 8000
# build the frontend first so the backend can serve it:
cd frontend && npm install && npm run build
```
