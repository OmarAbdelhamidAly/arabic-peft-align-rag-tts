# Lightweight Kubernetes Deployment (8 GB RAM)

This folder provides a minimal, laptop-friendly Kubernetes deployment for local LLM inference.

## Scope

- single replica only
- no autoscaling
- no service mesh
- no full Kubeflow stack

This is intentionally small to fit constrained hardware and to demonstrate deployment concepts.

## Prerequisites

- Docker Desktop (WSL2 backend enabled)
- `kubectl`
- one local Kubernetes option:
  - `kind` (recommended), or
  - `minikube`

## Suggested Flow

1. Start local Kubernetes cluster.
2. Run local inference server container image.
3. Apply manifests in this directory.
4. Port-forward and test endpoint.
5. Run benchmark and post-deploy evaluation scripts.

## kind Quick Start

```powershell
kind create cluster --name llm-local
kubectl cluster-info
```

## Apply Manifests

```powershell
kubectl apply -f model_training/evaluation/k8s/configmap.yaml
kubectl apply -f model_training/evaluation/k8s/deployment.yaml
kubectl apply -f model_training/evaluation/k8s/service.yaml
kubectl get pods -n default -w
```

## Access Service

```powershell
kubectl port-forward service/llm-inference-service 8001:8001
```

Then test:

```powershell
curl http://127.0.0.1:8001/health
```

## Notes for vLLM

If you are using vLLM, keep model/context limits small for local RAM safety. Start with:

- reduced max model length
- reduced max batched tokens
- single worker/process

## Cleanup

```powershell
kubectl delete -f model_training/evaluation/k8s/service.yaml
kubectl delete -f model_training/evaluation/k8s/deployment.yaml
kubectl delete -f model_training/evaluation/k8s/configmap.yaml
kind delete cluster --name llm-local
```

