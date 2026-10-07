# Local Kubernetes deployment (Minikube)

```bash
minikube start
# Build locally and make image visible to Minikube:
eval $(minikube docker-env)   # Linux/macOS shell
# On Windows PowerShell: minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t YOUR_DOCKERHUB_USER/heart-disease-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml
kubectl get pods,svc
minikube tunnel
```

Then call the EXTERNAL-IP shown by `kubectl get svc heart-api`. If Docker Desktop Kubernetes is used,
apply the same manifest and use the LoadBalancer address assigned by Docker Desktop.
