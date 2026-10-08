#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/services/api"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

echo "[1/4] Python syntax"
"$PYTHON_BIN" -m compileall -q services/api/app inference modal scripts tests

echo "[2/4] Unit tests"
"$PYTHON_BIN" scripts/run_isolated_tests.py

echo "[3/4] Shell scripts"
for file in scripts/*.sh; do
  bash -n "$file"
done

echo "[4/4] Frontend build"
if [ -d apps/web/node_modules ]; then
  (cd apps/web && npm run lint && npm test && npm run build)
else
  echo "Skipping frontend build: run 'cd apps/web && npm ci' first."
fi

echo "Verification complete."
