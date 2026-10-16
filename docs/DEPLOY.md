# Deploying

The server is one Docker image (`Dockerfile`, CPU build) used in two roles:

| Role | Where it runs | What it does |
|---|---|---|
| **Site server** | The kiosk laptop / mini-PC at each jobsite | Kiosk, foreman dashboard, face models, PPE check, emergency muster. Works with **no internet**. |
| **Central server** (optional) | Any cloud host | Receives attendance from every site and shows it under **All sites**. Never receives face data or PINs. |

## 1. Try it locally

```powershell
$env:ADMIN_PASSWORD = "change-me"
$env:SYNC_TOKEN = (python -c "import secrets; print(secrets.token_urlsafe(24))")
docker compose --env-file NUL up --build        # NUL: don't read .env (see note below)
```

| | |
|---|---|
| Site kiosk | http://localhost:8000/kiosk |
| Site dashboard | http://localhost:8000/dashboard |
| Central dashboard | http://localhost:8001/dashboard#sites |

To see offline mode, run `docker compose stop central`. The site keeps checking people in, and its
dashboard badge says "Offline · N waiting to sync". Then run `docker compose start central`, and
the queue syncs within about a minute.

*Note:* Compose reads `.env` from the project folder automatically. If that file isn't in plain
`KEY=value` form (for example, wrapped in `"""`), Compose fails to parse it. `--env-file NUL`
(`/dev/null` on macOS/Linux) skips it.

## 2. Settings

| Variable | Role | Meaning |
|---|---|---|
| `ADMIN_PASSWORD` | both | Dashboard password (user `foreman`). The dashboard is disabled without it. |
| `TZ` | both | **The jobsite's time zone**, e.g. `America/Chicago`. Check-in times and daily hours are local time. |
| `DB_PATH` | both | Default `/data/jobsite.db`. Keep `/data` on a volume. |
| `SITE_ID` | site | Short name for this site, e.g. `site-a` |
| `CENTRAL_URL` | site | Central server's address. Must be `https://` |
| `SYNC_TOKEN` | site | This site's secret, shared with the central server |
| `SITE_TOKENS` | central | Sites allowed to push: `site-a:token-a,site-b:token-b` |
| `SYNC_ALLOW_HTTP` | site | `1` allows `http://` sync. Only for a private network like Compose's |

Generate tokens with `python -c "import secrets; print(secrets.token_urlsafe(24))"`.

## 3. HTTPS is required for the camera

Browsers only open the camera on `https://` pages or on `localhost`. So:

- **Kiosk on the same machine as the site server:** open `http://localhost:8000/kiosk`. No
  certificate is needed. This is the simplest jobsite setup.
- **Kiosk on a tablet reaching the server over the network, or a public demo:** put the server
  behind HTTPS. Most hosts terminate TLS for you. On your own machine, a reverse proxy such as
  Caddy (`caddy reverse-proxy --from your.domain --to localhost:8000`) gets a certificate
  automatically. The container already runs uvicorn with `--proxy-headers`.

## 4. The PPE detector

The trained weights aren't in git or in the image, because Ultralytics and the dataset are
AGPL-3.0. Train them once (`python scripts/train_ppe.py`, about 7 minutes on a GPU), then mount the folder:

```
docker run ... -v ./models:/app/models:ro jobsite-checkin
```

Without the weights the kiosk works normally and PPE shows as "not checked".

## 5. Hosting a public demo

The face models need about 1.5 GB of RAM, so free tiers with 512 MB won't run it.

- **Hugging Face Spaces** (Docker SDK): free CPU Spaces have 16 GB RAM and HTTPS. Create a Docker
  Space, push this repo, and set `ADMIN_PASSWORD` and `TZ` as Space secrets. The app must listen
  on port 7860 there, so add `--port 7860` to the CMD or set `app_port: 8000` in the Space's
  README metadata.
- **Render / Railway / Fly.io**: deploy the Dockerfile on a plan with at least 2 GB RAM. Attach a
  persistent volume at `/data`.

For a public demo, use a fresh database with `python scripts/seed_demo.py`. Its made-up crew has
PIN 1111 and no face data. Visitors' faces are only processed in memory unless someone enrolls.
Consider leaving `ADMIN_PASSWORD` private and sharing only the kiosk link.
