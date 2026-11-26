#!/usr/bin/env bash
# End-to-end smoke test of the Docker stack on synthetic data (run in CI; also works locally).
#
#   1. build the image and start MLflow + API      -> /health: no model, /predict: 503
#   2. register the baseline on synthetic train/val -> /health: version 1, /predict: the known median
#
# Usage: scripts/docker_smoke.sh   (API_PORT overrides the host port, as in `make up`)
# Uses its own compose project and removes its containers and volume on exit, so it never touches
# the registry of a stack you already run.
set -euo pipefail

cd "$(dirname "$0")/.."
export COMPOSE_PROJECT_NAME=h1b-smoke
export H1B_IMAGE=h1b-salary:smoke   # don't overwrite the h1b-salary:dev image of `make up`
export API_PORT="${API_PORT:-8000}"
export MLFLOW_PORT="${MLFLOW_PORT:-5001}"
PYTHON="${PYTHON:-python}"
API="http://localhost:${API_PORT}"
DATA_DIR="$(mktemp -d)"
export H1B_PROCESSED_DIR="$DATA_DIR"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then docker compose logs --tail=80 mlflow api || true; fi
  docker compose --profile jobs down -v --remove-orphans >/dev/null 2>&1 || true
  docker image rm "$H1B_IMAGE" >/dev/null 2>&1 || true   # layers stay in the build cache
  rm -rf "$DATA_DIR"
  exit "$status"
}
trap cleanup EXIT

PAYLOAD='{"records": [{"soc_code": "15-1252", "worksite_state": "CA", "worksite_city": "San Jose",
  "job_title": "Software Engineer", "employer_name": "Acme Corp", "naics_2": "54", "full_time": null}]}'

# JSON field check: check <json> <python expression on d> <message>
check() {
  "$PYTHON" -c "import json, sys; d = json.loads(sys.argv[1]); assert $2, sys.argv[1]" "$1" \
    || { echo "FAIL: $3"; exit 1; }
  echo "ok: $3"
}

echo "== 1. start the stack (no champion yet)"
"$PYTHON" scripts/make_synthetic_processed.py "$DATA_DIR"
docker compose build api
docker compose up -d --wait mlflow api

check "$(curl -fsS "$API/health")" 'd["model_loaded"] is False' "/health reports no model"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/predict" -H 'Content-Type: application/json' -d "$PAYLOAD")
[ "$code" = "503" ] || { echo "FAIL: /predict before registration returned $code, expected 503"; exit 1; }
echo "ok: /predict returns 503 before registration"

echo "== 2. register the baseline on synthetic data"
docker compose run --rm register
docker compose restart api
docker compose up -d --wait api

check "$(curl -fsS "$API/health")" 'd["model_loaded"] is True and d["model_version"] == "1"' "/health serves version 1"
pred=$(curl -fsS -X POST "$API/predict" -H 'Content-Type: application/json' -d "$PAYLOAD")
check "$pred" 'abs(d["predictions"][0] - 150000) < 1' "/predict returns the (15-1252, CA) train median, 150000"
echo "Docker smoke test passed"
