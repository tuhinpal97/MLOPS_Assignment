#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .codespace-logs

if [[ ! -f artifacts/model/model.joblib ]]; then
  echo "Model artifact missing; training before service startup..."
  python -m src.train --tracking-uri file:./mlruns
fi

if [[ ! -d mlruns ]]; then
  echo "MLflow runs missing; creating experiment data..."
  python -m src.train --tracking-uri file:./mlruns
fi

# Reuse an already-running MLflow UI instead of starting a second server on port 5000.
if curl -fsS --max-time 3 http://127.0.0.1:5000/ >/dev/null 2>&1; then
  echo "MLflow is already responding on port 5000; reusing the existing server."
elif ss -ltn 2>/dev/null | awk 'NR > 1 && $4 ~ /:5000$/ { found=1 } END { exit !found }'; then
  echo "WARNING: Port 5000 is occupied, but MLflow did not respond to HTTP."
  echo "Inspect the listener with: ss -ltnp | grep ':5000'"
  echo "If it is a stale MLflow process, stop that PID and rerun this script."
else
  echo "Starting MLflow on port 5000..."
  nohup mlflow ui \
    --backend-store-uri ./mlruns \
    --host 0.0.0.0 \
    --port 5000 \
    > .codespace-logs/mlflow.log 2>&1 &
fi

echo "Starting FastAPI, Prometheus and Grafana with Docker Compose..."
docker compose up -d --build

for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "Generating sample API traffic for Prometheus/Grafana..."
  for _ in {1..8}; do
    curl -fsS -X POST http://127.0.0.1:8000/predict \
      -H 'Content-Type: application/json' \
      --data @sample_request.json >/dev/null || true
  done
fi

echo
printf '%s\n' \
  "Cloud lab is ready." \
  "FastAPI:    port 8000 (/docs, /health, /predict, /metrics)" \
  "MLflow:     port 5000" \
  "Prometheus: port 9090" \
  "Grafana:    port 3000 (anonymous Admin enabled for this educational Codespace)" \
  "Use the Codespaces PORTS tab to open each service."
