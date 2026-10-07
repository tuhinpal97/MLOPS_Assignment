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

if ! pgrep -f "mlflow ui --backend-store-uri" >/dev/null 2>&1; then
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
