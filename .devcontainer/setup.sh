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

if [[ ! -f artifacts/model/model.joblib ]]; then
  echo "==> Training model and creating local MLflow runs"
  python -m src.train --tracking-uri file:./mlruns
else
  echo "==> Existing model artifact found; skipping initial training"
fi

echo "==> Codespaces setup complete"
echo "The services will start automatically whenever the Codespace starts."
