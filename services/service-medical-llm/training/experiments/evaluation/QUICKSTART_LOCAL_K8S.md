# Quickstart: Local Serving + Kubernetes

This quickstart runs the implementation artifacts added for this project.

## 1) Start local server

Start your inference server first (for example vLLM OpenAI-compatible server).

Expected endpoints:

- `GET /health`
- `POST /v1/chat/completions`

## 2) Run baseline benchmark

```powershell
python model_training/evaluation/local_serving/benchmark_inference.py --model <MODEL_ID>
```

## 3) Create local Kubernetes cluster

```powershell
kind create cluster --name llm-local
```

## 4) Apply manifests

```powershell
kubectl apply -f model_training/evaluation/k8s/configmap.yaml
kubectl apply -f model_training/evaluation/k8s/deployment.yaml
kubectl apply -f model_training/evaluation/k8s/service.yaml
kubectl get pods -w
```

## 5) Port-forward and test

```powershell
kubectl port-forward service/llm-inference-service 8001:8001
```

In another terminal:

```powershell
python model_training/evaluation/local_serving/eval_after_deploy.py --model <MODEL_ID>
```

## 6) Follow checklist

Use:

- `model_training/evaluation/SOLUTION_EXECUTION_CHECKLIST.md`

## 7) Cleanup

```powershell
kubectl delete -f model_training/evaluation/k8s/service.yaml
kubectl delete -f model_training/evaluation/k8s/deployment.yaml
kubectl delete -f model_training/evaluation/k8s/configmap.yaml
kind delete cluster --name llm-local
```
