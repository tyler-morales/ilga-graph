# Vultr deployment guide (ILGA Graph)

Step-by-step guide to deploy the ILGA Graph app on a Vultr VPS (Ubuntu), including corrections for common pitfalls. For generic deployment concepts see [Deployment](deployment.md).

---

## Prerequisites

- A Vultr server (Ubuntu 22.04 or 24.04).
- Your SSH key added in Vultr (or the password for the user Vultr shows).
- Domain (e.g. `landofkei.org`) with an A record pointing to the server IP, or plan to use the raw IP for testing.

---

## Step 1: SSH into the server

**Important:** Vultr often gives you a non-root user (e.g. `linuxuser`), not `root`. Use the **username shown in the Vultr dashboard** for your server.

```bash
ssh linuxuser@YOUR_SERVER_IP
```

Example: `ssh linuxuser@45.76.21.216`. If you use `root` and the dashboard says `linuxuser`, you will get "Permission denied" when pasting the password — switch to the correct username.

Enter the password when prompted (or use SSH key auth if configured).

---

## Step 2: One-time server setup

**2a.** Update the system:

```bash
sudo apt update && sudo apt upgrade -y
```

**2b.** Install Git, Python, venv, pip, and Nginx.

On **Ubuntu 24.04** the default Python is 3.12; the stock repos do not include `python3.11`. Use the **system Python** (the app supports Python 3.10+):

```bash
sudo apt install -y git python3 python3-venv python3-pip nginx
```

**2c.** Confirm Python:

```bash
python3 --version
```

You should see something like `Python 3.12.x`. Use `python3` (and `python3 -m venv`) for all following steps.

---

## Step 3: Clone the repo

**3a.** From your home directory:

```bash
cd ~
git clone https://github.com/tyler-morales/ilga-graph.git
```

This creates a folder named `ilga-graph` (with a hyphen). For a private repo, use a Personal Access Token as the password when Git prompts, or set up SSH keys on the server.

**3b.** Enter the project and confirm layout:

```bash
cd ilga-graph
ls -la
```

You should see `src/`, `pyproject.toml`, `Procfile`, `README.md`, etc.

---

## Step 4: Python virtual environment and dependencies

**4a.** Create the venv (from project root `~/ilga-graph`):

```bash
python3 -m venv .venv
```

**4b.** Activate the venv (required before any `pip` install):

```bash
source .venv/bin/activate
```

Your prompt should show `(.venv)`. If you run `pip install -e .` **without** activating first, you will get an "externally-managed-environment" error because the system Python is protected.

**4c.** Install the project:

```bash
pip install -e .
```

**4d.** Confirm:

```bash
pip show ilga-graph
```

You should see the package listed with location under `~/ilga-graph`.

---

## Step 5: Cache, data directory, and production `.env`

**5a.** Create the data directory at **project root** (app writes `data/ilga.db` here):

```bash
mkdir -p data
```

**5b.** Copy and edit environment file:

```bash
cp .env.example .env
vim .env   # or nano .env
```

Set at least:

| Variable | Value |
|----------|--------|
| `ILGA_PROFILE` | `prod` |
| `ILGA_LOAD_ONLY` | `1` |
| `ILGA_CORS_ORIGINS` | Your public URL, e.g. `https://landofkei.org` |
| `ILGA_AUTH_SECRET` | Long random string (e.g. `openssl rand -hex 32` on your Mac, paste result) |
| `ILGA_API_KEY` | (Recommended.) Another random string for API protection; browser pages still work without it. |

Generate two **different** secrets: one for `ILGA_AUTH_SECRET` (signs session cookies) and one for `ILGA_API_KEY` (authenticates API clients). Do not reuse the same value.

**5c.** Populate the cache:

- **Option A — From your Mac:** If you have a full `cache/` locally (e.g. after `make scrape`), from your **Mac** in the project directory run:
  ```bash
  rsync -avz --progress cache/ linuxuser@YOUR_SERVER_IP:~/ilga-graph/cache/
  ```
- **Option B:** Create an empty `cache/` on the server and scrape or copy data later; the app will start but may report `ready: false` until cache has the expected files (e.g. `members.json`, `bills.json`, `committees.json`).

---

## Step 6: Systemd service (app runs on boot and restarts on failure)

**6a.** Create the service file (replace `linuxuser` and paths if your user or project path differs):

```bash
sudo vim /etc/systemd/system/ilga-graph.service
```

Paste (adjust `User`, `Group`, and paths if needed):

```ini
[Unit]
Description=ILGA Graph FastAPI app
After=network.target

[Service]
Type=simple
User=linuxuser
Group=linuxuser
WorkingDirectory=/home/linuxuser/ilga-graph
EnvironmentFile=/home/linuxuser/ilga-graph/.env
Environment=ILGA_LOAD_ONLY=1
Environment=ILGA_PROFILE=prod
ExecStart=/home/linuxuser/ilga-graph/.venv/bin/uvicorn ilga_graph.main:app --app-dir src --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Save and exit.

**6b.** Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ilga-graph
sudo systemctl start ilga-graph
```

**6c.** Check status:

```bash
sudo systemctl status ilga-graph
```

Press `q` to exit the pager. You should see `active (running)`.

**6d.** Startup time: With a full cache (e.g. 180 members, 11k+ bills), the app can take **around two minutes** to finish startup (load data, compute analytics, export vault). Uvicorn only binds to port 8000 **after** the FastAPI lifespan completes. Do not assume the app is broken if `curl` fails in the first minute.

Wait until the logs show:

- `Application startup complete.`
- `Uvicorn running on http://127.0.0.1:8000`

Then test:

```bash
curl http://127.0.0.1:8000/health
```

To watch logs until startup completes:

```bash
sudo journalctl -u ilga-graph -f
```

Then run `curl` once you see "Uvicorn running on".

---

## Step 7: Nginx reverse proxy

**7a.** Create the Nginx site config (use your domain or IP for `server_name`):

```bash
sudo vim /etc/nginx/sites-available/ilga-graph
```

Example for domain `landofkei.org`:

```nginx
server {
    listen 80;
    server_name landofkei.org;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

For IP-only testing, set `server_name 45.76.21.216;` (or your server IP).

**7b.** Enable the site and test:

```bash
sudo ln -s /etc/nginx/sites-available/ilga-graph /etc/nginx/sites-enabled/
sudo nginx -t
```

**7c.** Reload Nginx:

```bash
sudo systemctl reload nginx
```

**7d.** From your browser or another machine, open `http://landofkei.org/health` (or `http://YOUR_IP/health`). You should get the same JSON as from `curl` on the server.

---

## Step 8: HTTPS with Let's Encrypt (Certbot)

**8a.** Open firewall ports **before** running Certbot. If UFW is active and only port 22 is allowed, Let's Encrypt cannot reach your server on port 80 and you will see:

`Certbot failed to authenticate ... Timeout during connect (likely firewall problem)`

Allow HTTP and HTTPS:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
sudo ufw status
```

**8b.** Install Certbot and the Nginx plugin:

```bash
sudo apt install -y certbot python3-certbot-nginx
```

**8c.** Obtain a certificate and let Certbot configure Nginx (replace with your domain):

```bash
sudo certbot --nginx -d landofkei.org
```

- Provide an email for renewal and security notices.
- Agree to the Terms of Service.
- When asked, choose to **redirect HTTP to HTTPS** (recommended).

**8d.** Test in the browser: `https://landofkei.org/health` should return the health JSON with a valid lock icon.

**8e.** Confirm automatic renewal:

```bash
sudo certbot renew --dry-run
```

---

## What lives on the server (and what you transfer)

Because `.gitignore` excludes several paths, it's easy to wonder what else must be copied. Here's what the server has and what (if anything) you still need.

| Item | On server? | Notes |
|------|------------|--------|
| **Code** | Yes | From `git clone`; `mocks/dev/` is committed so it's there. |
| **`.venv/`** | Yes | You created it on the server and ran `pip install -e .`. |
| **`.env`** | Yes | You created and edited it on the server (never commit this). |
| **`cache/`** | Yes | You transferred it via `rsync` (members, bills, committees, etc.). If your cache includes `scorecards.json`, `moneyball.json`, and `zip_to_district.json`, the app uses them; otherwise it computes or loads as configured. |
| **`data/ilga.db`** | Yes, created at startup | The app runs `init_db()` on first startup and creates `data/ilga.db` (and the `data/` directory) automatically. So the **database exists** on the server; it is just **empty** (no users, no outreach events) until someone signs in or records outreach. You do **not** need to copy a DB from your Mac unless you want to **migrate** existing local users or outreach data — in that case you would copy `data/ilga.db` from your Mac to the server (with the app stopped) and ensure file ownership matches the service user. |
| **`ILGA_Graph_Vault/`** | No, and not required | Generated by the exporter for Obsidian; the web app does not need it. |
| **`processed/`** (ML) | Optional | Gitignored. Startup may say "ML intelligence: not available (run make ml-run)". If you want the ML intelligence features on the server, run the ML pipeline there or copy `processed/` from your Mac. |

**Summary:** You already transferred the only required gitignored data (**cache/**). The **database is there** — it was created when the app first started; it's empty until users sign in or log outreach. Nothing else is required for the front end and API to run. Optionally: copy `data/ilga.db` from Mac if you need to migrate local auth/outreach data; add `processed/` (or run `make ml-run` on the server) if you want ML intelligence.

**Quick sanity check:** Open `https://landofkei.org/advocacy`, enter an Illinois ZIP (e.g. 60608), and confirm you see legislator cards. Try signing in (or creating an account) and recording a call or email — that will write to `data/ilga.db` on the server and confirms the full stack is working.

### Where is the database?

The SQLite file is **inside** the `data/` directory:

```bash
ls -la ~/ilga-graph/data/
```

You should see `ilga.db` (created by `init_db()` on first app startup). Full path: `/home/linuxuser/ilga-graph/data/ilga.db`. To inspect or backup: `sqlite3 ~/ilga-graph/data/ilga.db` or copy the file off the server.

### Getting ML data on the server

The app’s ML intelligence (e.g. prediction table, coalitions) reads from the **`processed/`** directory (gitignored). You have two options:

**Option A — Copy from your Mac** (fastest if you already ran the pipeline locally):

From your **Mac**, in the project root (with `processed/` populated after `make ml-run`):

```bash
rsync -avz processed/ linuxuser@YOUR_SERVER_IP:~/ilga-graph/processed/
```

Then restart the app so it picks up the files: `sudo systemctl restart ilga-graph`.

**Option B — Run the ML pipeline on the server:**

On the server, with the venv active and from `~/ilga-graph`:

```bash
pip install -e ".[ml]"
make ml-run
```

This can be slow and memory-heavy (full cache + sklearn, etc.). To skip hyperparameter tuning for a faster run: `ILGA_ML_SKIP_TUNE=1 make ml-run`. Then restart the app.

---

## Post-deploy checklist

- [ ] `ILGA_CORS_ORIGINS` in `.env` includes your public URL (e.g. `https://landofkei.org`).
- [ ] `ILGA_AUTH_SECRET` and `ILGA_API_KEY` are set to distinct random values and are not committed.
- [ ] Cache is populated (or you accept `ready: false` until it is).
- [ ] `data/` (or `ILGA_DB_PATH` directory) is writable by the service user.
- [ ] Optional: configure `ILGA_SMTP_*` for verification emails; see [Environment variables](environment-variables.md) and [Email (Brevo)](email-brevo.md).

---

## Automated deploy (CI/CD)

On **push to `main`**, after CI (lint and test) passes, GitHub Actions runs a deploy job that SSHs to this server, pulls the latest code, installs dependencies, and restarts the app. You do not need to SSH in and run these steps manually.

### One-time setup

1. **GitHub Secrets** (repo → Settings → Secrets and variables → Actions): add
   - `DEPLOY_HOST` — server IP or hostname (e.g. `45.76.21.216` or `landofkei.org`)
   - `DEPLOY_USER` — SSH user (e.g. `linuxuser`)
   - `DEPLOY_SSH_KEY` — private key contents for a dedicated deploy key (no passphrase)

2. **Deploy key on the server:** Generate an SSH key pair on your machine (e.g. `ssh-keygen -t ed25519 -C "github-deploy" -f deploy_key -N ""`). Add the **public** key to the server user’s `~/.ssh/authorized_keys` (same user as `DEPLOY_USER`).

3. **Passwordless sudo for restart:** So the deploy can run `systemctl restart ilga-graph` without a password, create a sudoers file:
   ```bash
   sudo visudo -f /etc/sudoers.d/ilga-graph-deploy
   ```
   Add one line (replace `linuxuser` with your deploy user):
   ```
   linuxuser ALL=(ALL) NOPASSWD: /bin/systemctl restart ilga-graph
   ```
   Save and exit. Restrict this to the deploy user and key in practice.

The workflow runs `cd ~/ilga-graph && bash scripts/deploy-on-server.sh`. If your app directory or user differs, you can run that script manually after SSH (see Useful commands) or add a `DEPLOY_PATH` secret and update the workflow.

---

## Automated daily scrape (offload from your Mac)

You can run the incremental scrape on the Vultr server on a schedule so you don’t have to run it on your Mac.

**1. One-time: allow the app user to restart the service without a password**

If you haven’t already (e.g. for CI deploy), add a sudoers rule so the user that runs cron can restart the app:

```bash
sudo visudo -f /etc/sudoers.d/ilga-graph-deploy
```

Add one line (replace `linuxuser` with your server user):

```
linuxuser ALL=(ALL) NOPASSWD: /bin/systemctl restart ilga-graph
```

**2. Optional: create a log directory**

```bash
mkdir -p ~/ilga-graph/logs
```

**3. Install a cron job**

Run `crontab -e` as the same user that owns the repo (e.g. `linuxuser`). Add a line to run the scrape daily (e.g. 3:00 AM server time):

```cron
0 3 * * * /home/linuxuser/ilga-graph/scripts/scrape-on-server.sh >> /home/linuxuser/ilga-graph/logs/scrape.log 2>&1
```

Adjust paths if your project lives elsewhere (e.g. `/home/YOUR_USER/ilga-graph`).

**What the script does**

- `scripts/scrape-on-server.sh`: activates the project venv, runs `scripts/scrape.py --fast` (incremental: members + bills + votes + slips, no full re-scrape), optionally runs the ML pipeline, then runs `sudo systemctl restart ilga-graph` so the app reloads the updated cache.

**Notes**

- The app loads cache only at startup, so restarting after the scrape is required for new data to appear.
- Scrape and ML use `|| true` so a failure does not block the restart.
- To run the scrape once by hand: `cd ~/ilga-graph && ./scripts/scrape-on-server.sh`.

---

## Useful commands

| Task | Command |
|------|--------|
| Restart the app | `sudo systemctl restart ilga-graph` |
| Run scrape once (then restart) | `cd ~/ilga-graph && ./scripts/scrape-on-server.sh` |
| View scrape log (if cron configured) | `tail -f ~/ilga-graph/logs/scrape.log` |
| View app logs (live) | `sudo journalctl -u ilga-graph -f` |
| Last 80 log lines | `sudo journalctl -u ilga-graph -n 80 --no-pager` |
| After pulling new code (manual) | `cd ~/ilga-graph && bash scripts/deploy-on-server.sh` (or `git pull && sudo systemctl restart ilga-graph`) |
| Test certificate renewal | `sudo certbot renew --dry-run` |
| Edit Nginx config | `sudo vim /etc/nginx/sites-available/ilga-graph` then `sudo nginx -t && sudo systemctl reload nginx` |

---

## Troubleshooting

- **SSH "Permission denied"** — Use the username shown in the Vultr dashboard (e.g. `linuxuser`), not `root`, unless your instance is configured for root.
- **"externally-managed-environment" when running pip** — You did not activate the venv. Run `source .venv/bin/activate` before `pip install -e .`.
- **curl to 127.0.0.1:8000 fails right after start** — Startup can take ~2 minutes with a full cache. Wait for "Application startup complete" and "Uvicorn running on" in `journalctl -u ilga-graph`, then retry.
- **Certbot timeout / "likely firewall problem"** — UFW or Vultr firewall was blocking port 80. Allow 80 and 443 (`sudo ufw allow 80/tcp` and `443/tcp`, reload), then run `sudo certbot --nginx -d yourdomain.org` again.
- **https:// loads nothing** — HTTPS is only available after Step 8. Use `http://` until Certbot has been run, or run Certbot and choose redirect so HTTP sends users to HTTPS.
