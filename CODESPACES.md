# Run the Complete MLOps Assignment in GitHub Codespaces

This repository is configured so the project can be demonstrated entirely in GitHub without installing Python, Docker, MLflow, Prometheus, Grafana, kubectl or Minikube on your local computer.

## Start the cloud environment

1. Open the repository on GitHub.
2. Select **Code** -> **Codespaces** -> **Create codespace on main**.
3. Wait for the development container setup to complete.

The Codespace automatically:

- creates a Python 3.11 environment;
- installs all dependencies;
- downloads/prepares the UCI Cleveland Heart Disease dataset;
- regenerates EDA artifacts;
- trains Logistic Regression and Random Forest;
- creates local MLflow experiment runs;
- starts the FastAPI service in Docker;
- starts MLflow;
- starts Prometheus;
- starts Grafana with Prometheus already configured as the default data source;
- provisions the assignment dashboard;
- generates sample prediction traffic so monitoring charts have data.

## Open the services

Open the **PORTS** tab in the Codespace. The following ports are forwarded automatically:

| Port | Service | What to capture |
|---:|---|---|
| 8000 | FastAPI | `/docs`, `/health`, `/predict`, `/metrics` |
| 5000 | MLflow | experiment runs, candidate metrics and artifacts |
| 9090 | Prometheus | Targets page and PromQL queries |
| 3000 | Grafana | pre-provisioned Heart Disease API dashboard |
| 8080 | Kubernetes demo | API served from the Minikube deployment |

For FastAPI interactive documentation, open port 8000 and append `/docs`.

## Test a prediction from the Codespace terminal

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @sample_request.json
```

Health and metrics:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics | head
```

## MLflow

Port 5000 opens the MLflow UI. The Codespace setup runs the training workflow so the `heart-disease-classification` experiment is visible with both candidate model runs and their metrics/artifacts.

## Prometheus

Open port 9090, then check **Status -> Targets**. The `heart-api` target should be UP.

Useful queries:

```text
sum(rate(heart_api_requests_total[1m]))
heart_predictions_total
histogram_quantile(0.95, sum(rate(heart_api_request_latency_seconds_bucket[5m])) by (le))
```

## Grafana

Open port 3000. For this temporary educational Codespace, anonymous Admin access is enabled so no login setup is required. The Prometheus data source and the `Heart Disease API - MLOps Assignment` dashboard are provisioned automatically.

Do not reuse this anonymous-admin configuration for a real production deployment.

## Kubernetes / Minikube demo

The Codespace contains kubectl and Minikube tooling. To create a local Kubernetes deployment inside the cloud Codespace, run:

```bash
bash .devcontainer/k8s-demo.sh
```

The script:

1. starts Minikube with the Docker driver;
2. builds the API image inside Minikube;
3. deploys two API replicas;
4. waits for the rollout to finish;
5. prints pods/services;
6. forwards the Kubernetes Service to Codespaces port 8080.

Then open port 8080 from the PORTS tab. You can also capture:

```bash
kubectl get pods
kubectl get deployments
kubectl get svc
kubectl describe deployment heart-api
```

> Minikube requires more resources than the basic API/monitoring stack. If the smallest Codespace machine does not have enough memory, recreate the Codespace with a larger machine type before running the Kubernetes demo.

## Useful service commands

Check Docker services:

```bash
docker compose ps
```

View API logs:

```bash
docker compose logs -f api
```

Restart the application stack:

```bash
docker compose restart
```

Rebuild it:

```bash
docker compose up -d --build
```

Check MLflow log:

```bash
cat .codespace-logs/mlflow.log
```

## Assignment screenshots

Recommended screenshot sequence:

1. GitHub Actions successful CI run.
2. MLflow experiment list with both models.
3. MLflow selected model metrics/artifacts.
4. FastAPI Swagger `/docs` plus successful `/predict` response.
5. Docker `docker compose ps` showing running services.
6. Prometheus Targets showing `heart-api` UP.
7. Grafana monitoring dashboard.
8. `kubectl get pods,svc` after the Minikube deployment.
9. Kubernetes-served API opened through Codespaces port 8080.

All of these can be produced from GitHub/Codespaces without installing anything locally.
