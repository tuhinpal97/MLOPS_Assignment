#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing project dependencies"
pip install -r requirements.txt

echo "==> Preparing dataset"
python scripts/download_data.py

echo "==> Generating EDA artifacts"
python -m src.eda

echo "==> Training models and creating MLflow experiment data"
rm -rf mlruns
python -m src.train --tracking-uri file:./mlruns

echo "==> Codespaces setup complete"
echo "FastAPI, MLflow, Prometheus and Grafana will start automatically."
