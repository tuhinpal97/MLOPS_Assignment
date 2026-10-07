# Required evidence screenshots

Capture these after running the project in your own environment. Do not fabricate them.

1. `01_mlflow_runs.png` - MLflow experiment table showing Logistic Regression and Random Forest runs.
2. `02_mlflow_champion.png` - champion run parameters, metrics, artifacts/model.
3. `03_github_actions.png` - successful CI job with lint, pytest, training and Docker build.
4. `04_docker_predict.png` - terminal/Postman showing Docker container `/predict` response.
5. `05_kubernetes.png` - `kubectl get pods,svc` with two Ready pods and LoadBalancer service.
6. `06_k8s_predict.png` - prediction through Kubernetes service/ingress.
7. `07_prometheus.png` - Prometheus target UP and `heart_api_requests_total` query.
8. `08_grafana.png` - dashboard panel(s) for request rate/latency/prediction counts.
9. `09_video_link.txt` - link to the short overall-pipeline recording.
