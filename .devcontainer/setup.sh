#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> Checking the Codespaces Python runtime"
python --version
python - <<'PY'
import sys

expected = (3, 11)
actual = sys.version_info[:2]
if actual != expected:
    raise SystemExit(
        "ERROR: This repository's Codespaces image is configured for Python 3.11, "
        f"but the active runtime is Python {sys.version.split()[0]}. "
        "Rebuild the Codespace container so .devcontainer/devcontainer.json is applied."
    )
PY

# PyPI's file host can be slow or briefly unreachable. Use longer timeouts and
# retries so a transient package-download delay does not fail the whole setup.
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
export PIP_DISABLE_PIP_VERSION_CHECK=1
PIP_NETWORK_ARGS=(--timeout 120 --retries 10)

echo "==> Upgrading pip (with network retries)"
python -m pip install "${PIP_NETWORK_ARGS[@]}" --upgrade pip

echo "==> Installing project dependencies (with network retries)"
python -m pip install "${PIP_NETWORK_ARGS[@]}" -r requirements.txt

echo "==> Preparing dataset"
python scripts/download_data.py

echo "==> Generating EDA artifacts"
python -m src.eda

echo "==> Training models and creating MLflow experiment data"
rm -rf mlruns
python -m src.train --tracking-uri file:./mlruns

echo "==> Codespaces setup complete"
echo "FastAPI, MLflow, Prometheus and Grafana will start automatically."
