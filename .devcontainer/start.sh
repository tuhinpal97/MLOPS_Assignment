#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .codespace-logs

if [[ ! -f artifacts/model/model.joblib ]]; then
  echo "Model artifact missing; training before service startup..."
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

echo
printf '%s\n' \
  "Cloud lab is starting." \
  "FastAPI:    port 8000 (/docs, /health, /predict, /metrics)" \
  "MLflow:     port 5000" \
  "Prometheus: port 9090" \
  "Grafana:    port 3000 (anonymous Admin enabled for this educational Codespace)" \
  "Use the Codespaces PORTS tab to open each service."
