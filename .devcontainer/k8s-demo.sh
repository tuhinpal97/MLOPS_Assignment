#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p .codespace-logs

if ! minikube status >/dev/null 2>&1; then
  echo "==> Starting Minikube"
  minikube start --driver=docker --cpus=2 --memory=4096
fi

echo "==> Building API image inside Minikube"
eval "$(minikube docker-env)"
docker build -t heart-disease-api:codespace .

echo "==> Applying Kubernetes manifest"
kubectl apply -f k8s/codespaces-deployment.yaml
kubectl rollout status deployment/heart-api --timeout=180s

echo "==> Current Kubernetes resources"
kubectl get pods,svc

if pgrep -f "kubectl port-forward service/heart-api 8080:80" >/dev/null 2>&1; then
  pkill -f "kubectl port-forward service/heart-api 8080:80" || true
  sleep 1
fi

nohup kubectl port-forward service/heart-api 8080:80 \
  > .codespace-logs/k8s-port-forward.log 2>&1 &

sleep 3
curl -fsS http://127.0.0.1:8080/health || true

echo
echo "Kubernetes demo is ready on Codespaces port 8080."
echo "Open the PORTS tab and launch port 8080."
