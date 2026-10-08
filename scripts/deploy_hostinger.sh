#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
COMPOSE=(docker compose -f docker-compose.production.yml)
DEPLOY_WORKER=false
case "${1:-}" in
  "") ;;
  --deploy-worker) DEPLOY_WORKER=true ;;
  *) echo "Usage: $0 [--deploy-worker]" >&2; exit 2 ;;
esac

if [ ! -f .env ]; then
  echo "Missing .env. Copy .env.production.example to .env and configure it first." >&2
  exit 1
fi

mkdir -p storage/generated storage/jobs storage/metrics storage/backups storage/logs
chmod 700 storage/backups || true

python3 scripts/production_preflight.py

# Capture a consistent SQLite/metrics backup before replacing containers.
python3 scripts/backup_state.py

"${COMPOSE[@]}" build --pull
# Deploy from the new API image so the worker code, pinned LTX ref, Modal token
# and Modal environment are exactly those used by this VPS. Do not use a local
# developer's active Modal profile, which can point at a different workspace.
if [ "$DEPLOY_WORKER" = true ]; then
  "${COMPOSE[@]}" run --rm --no-deps api python -m modal deploy modal/app.py --strategy rolling
fi
# Validate the deployed CPU worker and model manifest before replacing the API.
# This check performs no GPU render.
"${COMPOSE[@]}" run --rm --no-deps api python /app/scripts/check_inference.py --all --require-current-worker
"${COMPOSE[@]}" up -d --remove-orphans

echo "Waiting for API readiness..."
for _ in $(seq 1 60); do
  if "${COMPOSE[@]}" exec -T api \
      curl -fsS http://127.0.0.1:8000/api/v1/health/ready >/dev/null 2>&1; then
    echo "API is ready."
    "${COMPOSE[@]}" ps

    domain="devansh.info"
    if [ -n "$domain" ]; then
      echo "Checking public HTTPS endpoint: https://$domain/api/v1/health/ready"
      if curl -fsS --connect-timeout 8 --max-time 15 \
          "https://$domain/api/v1/health/ready" >/dev/null 2>&1; then
        echo "Public HTTPS health check passed."
      else
        echo "Deployment verification failed: public HTTPS is not reachable." >&2
        echo "Check DNS A/AAAA records, Hostinger firewall/UFW, Nginx config, and Certbot certificate status." >&2
        exit 1
      fi
    fi
    exit 0
  fi
  sleep 2
done

echo "API did not become ready in time." >&2
"${COMPOSE[@]}" ps >&2
"${COMPOSE[@]}" logs --tail=160 api >&2
exit 1
