# Short Video Recording Script

A concise 5-7 minute recording can demonstrate the complete pipeline without unnecessary detail.

## 1. Repository and architecture (30-45 sec)
Show the repository tree and `artifacts/plots/architecture.png`. Explain the flow: UCI data -> cleaning/EDA -> reusable preprocessing -> two candidate models -> MLflow -> champion artifact -> FastAPI -> Docker/Kubernetes -> Prometheus/Grafana.

## 2. Data and EDA (45 sec)
Run or show:
```bash
python scripts/download_data.py
python -m src.eda
```
Open the class-balance, histogram, and correlation artifacts. Mention the 303 Cleveland records, binary target, and in-pipeline handling of missing `ca`/`thal` values.

## 3. Training and MLflow (60-75 sec)
Show MLflow UI and run:
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
python -m src.train --tracking-uri file:./mlruns
```
Compare Logistic Regression and Random Forest. Explain that Logistic Regression is selected by the predeclared 5-fold CV ROC-AUC criterion rather than by holdout performance.

## 4. Tests and CI (45 sec)
Run:
```bash
pytest -q
```
Then show the successful GitHub Actions workflow and its uploaded artifacts. Mention that lint/test/training/Docker failures stop the workflow.

## 5. API and Docker (60 sec)
Run:
```bash
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
Point out prediction, confidence, and model name in the response.

## 6. Kubernetes deployment (45-60 sec)
Show:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods,svc
```
Explain two replicas, health probes, resource requests/limits, and LoadBalancer exposure. Call `/predict` through the Kubernetes endpoint.

## 7. Monitoring (45-60 sec)
Run the local monitoring stack if not already running:
```bash
docker compose up --build
```
Show `/metrics`, Prometheus targets/query results, and the Grafana dashboard. Mention request count, latency, prediction count, JSON logging, and the optional batch drift check.

## 8. Closing (15-20 sec)
State that the persisted artifact contains preprocessing plus the classifier, dependencies are pinned, CI retains artifacts/logs, and deployment evidence is included in the report.
