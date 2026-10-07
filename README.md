# Heart Disease MLOps Assignment - AIML ZG523

End-to-end MLOps implementation for the UCI Cleveland Heart Disease classification problem. The project covers data acquisition and EDA, reusable preprocessing, two tuned classifiers, MLflow experiment tracking, reproducible model packaging, unit tests, GitHub Actions CI, FastAPI serving, Docker, Kubernetes deployment, and Prometheus/Grafana monitoring.

> Educational use only. This model is not a medical device and must not be used for diagnosis or treatment decisions.

## 1. Architecture

![Architecture](artifacts/plots/architecture.png)

The implementation deliberately uses technologies covered by the course handout: MLflow for experimentation, Docker for packaging, a model-serving microservice, Kubernetes for deployment, and Prometheus/Grafana for monitoring.

## 2. Repository structure

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   ├── metrics/
│   ├── model/
│   └── plots/
├── data/
│   ├── raw/processed.cleveland.data
│   └── processed/heart_clean.csv
├── k8s/
├── monitoring/
├── notebooks/
├── report/
├── screenshots/
├── scripts/download_data.py
├── src/
│   ├── api/
│   ├── monitoring/
│   ├── data.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── sample_request.json
```

## 3. Clean setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Prepare the dataset and generate EDA artifacts:

```bash
python scripts/download_data.py
python -m src.eda
```

The download script first uses the official UCI URL and has a public mirror fallback. Missing values represented by `?` are parsed as NaN. The original UCI `num` target (0-4) is converted to a binary label: `0 -> no disease`, `1-4 -> disease present`.

## 4. Model training and MLflow

Start the MLflow UI in one terminal:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Train and log experiments in another terminal:

```bash
python -m src.train --tracking-uri file:./mlruns
```

Open `http://127.0.0.1:5000`. The training job performs 5-fold stratified cross-validation, hyperparameter tuning, holdout evaluation, and logs parameters, metrics, plots, and the fitted model for both Logistic Regression and Random Forest.

### Reference results from the included deterministic split

| Model | CV Accuracy | CV Precision | CV Recall | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8470 | 0.8783 | 0.7838 | **0.9065** | 0.8689 | 0.8125 | 0.9286 | 0.9578 |
| Random Forest | 0.8179 | 0.8083 | 0.8008 | 0.8961 | **0.8852** | 0.8182 | **0.9643** | **0.9589** |

The champion is **Logistic Regression** because the model-selection rule is fixed in advance as mean 5-fold CV ROC-AUC on the training split. This avoids selecting a model by looking at the holdout test set. After selection, the champion pipeline is refit on the full cleaned dataset and saved as `artifacts/model/model.joblib`.

## 5. Run tests and lint

```bash
ruff check src tests scripts
pytest -q
```

The current project tests data cleaning/target construction, missing-value-safe preprocessing, and the API response contract.

## 6. Run the API locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Example response:

```json
{
  "prediction": 1,
  "label": "heart_disease_risk",
  "confidence": 0.539543,
  "model_name": "logistic_regression"
}
```

Prometheus metrics are exposed at `GET /metrics`. API request logs are structured JSON and intentionally do not record raw patient feature values.

## 7. Docker and monitoring

Build only the API container:

```bash
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
```

Or start API + Prometheus + Grafana together:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default local credentials depend on the Grafana image; change them for any non-local use)

Add Prometheus as a Grafana data source using `http://prometheus:9090`. Useful queries include:

```text
rate(heart_api_requests_total[1m])
histogram_quantile(0.95, sum(rate(heart_api_request_latency_seconds_bucket[5m])) by (le))
heart_predictions_total
```

## 8. Kubernetes deployment

Edit the image name in `k8s/deployment.yaml`, then:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods,svc
```

For Minikube, use `minikube tunnel` so the `LoadBalancer` service obtains a reachable address. Read `k8s/README.md` for the complete local sequence.

## 9. CI/CD

GitHub Actions runs on pushes and pull requests to `main`:

1. install the pinned Python environment;
2. lint with Ruff;
3. run Pytest and emit JUnit XML;
4. prepare the UCI dataset and EDA plots;
5. train/log both models with MLflow;
6. build the Docker image;
7. upload model, metrics, plots and MLflow files as workflow artifacts.

![CI/CD workflow](artifacts/plots/cicd_workflow.png)

## 10. Optional drift check

The handout also covers production drift. A simple KS-test-based batch check is included:

```bash
python -m src.monitoring.drift_check --current path/to/current_batch.csv
```

This is a demonstration alerting signal, not a complete production drift policy.

## 11. Evidence required before final submission

The code and report are complete, but screenshots/video and a public repository/deployment URL must be captured from your own environment/account. Follow `screenshots/README.md`. This avoids fabricating execution evidence.

## 12. Reproducibility notes

- Raw data is retained and processed data is generated deterministically.
- Preprocessing is inside the persisted scikit-learn pipeline, preventing training/serving skew.
- Random state is fixed to 42 where applicable.
- Dependencies are pinned.
- The champion selection metric is defined before looking at holdout performance.
- The production artifact is a single fitted preprocessing + estimator pipeline.

## 13. Dataset citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C52P4X.
