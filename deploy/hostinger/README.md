# Triven Cinema on Hostinger VPS

The current production target is one Hostinger Linux VPS for the web/API/orchestration layer and Modal for LTX-2.5 GPU inference.

```text
Internet
  |
  v
Host Nginx :80/:443
  |-------------------------------|
  v                               v
127.0.0.1:3333                127.0.0.1:3334
Next.js                         FastAPI
                                   |
                                   +-- persistent jobs/billing/integration SQLite state
                                   +-- generated media on VPS storage
                                   +-- Gemini planner / local fallback
                                   +-- Modal API -> B200 -> LTX-2.5
```

The Docker services never bind FastAPI/Next.js to a public interface. This VPS already uses host Nginx for other Triven domains, so the production Compose file intentionally contains **no Caddy service**.

## 1. Requirements

Ubuntu LTS, Docker Engine, Docker Compose v2, Nginx and Certbot:

```bash
sudo apt update
sudo apt install -y ca-certificates curl git ufw nginx certbot python3-certbot-nginx
```

Verify:

```bash
docker --version
docker compose version
nginx -v
```

## 2. Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status
```

Do not expose 3000, 8000, 3333 or 3334 publicly. `3333` and `3334` are loopback-only debug/upstream ports.

## 3. DNS and HTTPS

The demo is pinned to:

```text
devansh.info
```

Point its A record to the VPS public IPv4. Remove stale AAAA records unless IPv6 is correctly configured.

Install the Nginx site:

```bash
sudo cp deploy/hostinger/nginx.triven-cinema.conf /etc/nginx/sites-available/triven-cinema
sudo ln -sf /etc/nginx/sites-available/triven-cinema /etc/nginx/sites-enabled/triven-cinema
sudo nginx -t
sudo systemctl reload nginx
```

Verify HTTP routing, then issue/attach the certificate:

```bash
curl -I http://devansh.info
sudo certbot --nginx -d devansh.info
```

Existing `cinema.triven.ai` / `dial.triven.ai` Nginx sites can stay enabled because they use different `server_name` values.

## 4. Configure `.env`

```bash
cp .env.production.example .env
chmod 600 .env
nano .env
```

Minimum render configuration:

```env
TRIVEN_DOMAIN="devansh.info"
APP_ENV="production"
DEBUG=false
FRONTEND_URL="https://devansh.info"
CORS_ORIGINS=""
AUTH_ENABLED=true
DEMO_AUTH_SHOW_OTP=true
TRIVEN_SECRET_KEY="use-a-random-secret-of-at-least-32-characters"
VIDEO_PROVIDER="modal"
GEMINI_API_KEY="..."
MODAL_TOKEN_ID="..."
MODAL_TOKEN_SECRET="..."
```

For billing and YouTube, follow `docs/AI_VIDEO_FACTORY.md`.

Use a dedicated Modal production token. Never put server secrets into `NEXT_PUBLIC_*` variables.

## 5. Preflight

```bash
python3 scripts/production_preflight.py
```

Fix every `[FAIL]` before deploying. The script expects host Nginx to own public 80/443.

Demo OTP is supported in production: `DEMO_AUTH_SHOW_OTP=true` displays the code on the login screen and skips SMTP. This mode does not verify mailbox ownership. To use email verification instead, set `DEMO_AUTH_SHOW_OTP=false` and configure `SMTP_HOST`, `SMTP_FROM_EMAIL`, credentials, and TLS/SSL as listed in `.env.production.example`. Production requires a session-signing key of at least 32 characters in either mode. Preserve the existing signing key when it is already strong; changing it invalidates sessions and affects encrypted integrations.

## 6. Deploy application containers

Deploy the compatible Modal worker in section 7 first. Application deployment calls its CPU preflight before replacing containers; an old worker or missing/inaccessible weights will stop rollout before GPU allocation.

```bash
./scripts/deploy_hostinger.sh
```

Or directly:

```bash
docker compose -f docker-compose.production.yml up -d --build api web maintenance
```

Status:

```bash
./scripts/status_hostinger.sh
```

## 7. Modal deployment

Redeploy Modal whenever `modal/app.py`, `modal/ltx_worker.py`, GPU type, LTX pipeline flags, or model runtime configuration changes:

```bash
set -a
source .env
set +a
modal deploy modal/app.py
```

The reliability update adds the CPU `preflight` function (protocol version 2). Existing weights are reused; download only missing or incomplete model files. After worker deployment, verify all enabled recipes:

```bash
python scripts/check_inference.py --all
```

This validates model-file access, sizes and worker compatibility. It does not allocate a GPU or verify output quality.

## 8. Duration profiles

Defaults:

```text
preview: 10s/scene
1080p delivery: 30s/scene
4K delivery: 15s/scene
factory total: 300s
```

Long Modal scenes use upstream LTX-2.5 temporal windows with overlap/blending inside one inference invocation. Strict continuity between story scenes uses the final frame of the previous scene as frame-0 conditioning for the next scene.

## 9. Persistent state and cleanup

`./storage` is bind-mounted into the API and maintenance containers. It contains generated media plus job/billing/integration state. The maintenance service runs backups and cleanup every six hours.

Manual backup:

```bash
docker compose -f docker-compose.production.yml exec -T api python /app/scripts/backup_state.py
```

Manual cleanup dry run:

```bash
docker compose -f docker-compose.production.yml exec -T api python /app/scripts/cleanup_storage.py
```

Enable Hostinger snapshots/backups as well. Local backups do not protect against total VPS/disk loss.

## 10. Updates

### One-time migration for formerly tracked runtime files

Before pulling the reliability update onto an existing server, wait for active jobs to finish, stop writes, and archive the live databases and uploaded references outside the repository. The old tracked SQLite files may have uncommitted production changes; archive them before stashing. Keep the archive until the deployment and account history are verified.

```bash
cd ~/triven-cinema
docker compose -f docker-compose.production.yml stop api maintenance
python3 scripts/backup_state.py
TRIVEN_STATE_BACKUP="$HOME/triven-state-before-untracking-$(date +%Y%m%dT%H%M%S).tar.gz"
tar -czf "$TRIVEN_STATE_BACKUP" storage/auth storage/chats storage/elements
chmod 600 "$TRIVEN_STATE_BACKUP"
git stash push -m "Runtime state before storage migration" -- storage/auth storage/chats storage/elements
git pull --ff-only
tar -xzf "$TRIVEN_STATE_BACKUP" -C .
```

Do not apply the runtime-state stash after the pull: restoring from the archive keeps those files untracked. If the pull fails, restore the archive before restarting the old application. Generated videos stay in the unchanged `storage/generated` bind mount. This sequence does not remove generated media.

For demo login, set `AUTH_ENABLED=true` and `DEMO_AUTH_SHOW_OTP=true` in the server's existing `.env`; do not replace that file or its signing key. SMTP is not required in demo mode. Refresh the Nginx template (it replaces untrusted forwarded IP headers for login rate limiting), deploy Modal separately when its code changes, then rebuild the application:

```bash
cd ~/triven-cinema
git pull --ff-only
./scripts/deploy_hostinger.sh
```

If Modal worker code changed:

```bash
set -a; source .env; set +a
modal deploy modal/app.py
```

## 11. Production checks

```bash
curl http://127.0.0.1:3334/api/v1/health/ready
curl -I http://127.0.0.1:3333
curl -I https://devansh.info
curl https://devansh.info/api/v1/health
```

Then run one short paid smoke render before a 15s/30s production render.

## Scaling boundary

This is a hardened single-VPS architecture, not a horizontally scalable multi-tenant control plane. Before multiple API instances, migrate SQLite state to Postgres, the in-process job executor to a durable queue, generated media to S3/R2, and signed-browser workspace identity to real customer authentication/RBAC.
