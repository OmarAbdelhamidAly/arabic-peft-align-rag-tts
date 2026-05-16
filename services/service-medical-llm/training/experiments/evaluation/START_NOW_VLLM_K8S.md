# Start Now: vLLM + Kubernetes (Laptop Demo)

This is the fastest path to run a real demo on your personal laptop.

## 0) One-time reality check

- `kubectl`: available
- `minikube`: available
- Docker daemon: currently not running

You must open Docker Desktop first.

## 1) Start Docker Desktop

Make sure Docker is running, then verify:

```powershell
docker ps
```

## 2) Start local Kubernetes

```powershell
minikube start --driver=docker --memory=4096 --cpus=2
kubectl get nodes
```

## 3) Deploy vLLM on Kubernetes

```powershell
kubectl apply -f model_training/evaluation/k8s/configmap.yaml
kubectl apply -f model_training/evaluation/k8s/deployment.yaml
kubectl apply -f model_training/evaluation/k8s/service.yaml
kubectl get pods -w
```

When pod is `Running` and `READY 1/1`, continue.

## 4) Expose endpoint

```powershell
kubectl port-forward service/llm-inference-service 8001:8001
```

Keep this terminal open.

## 5) Quick health + smoke test

In a second terminal:

```powershell
curl http://127.0.0.1:8001/health
python model_training/evaluation/local_serving/eval_after_deploy.py --model unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit
```

## 6) Optional benchmark

```powershell
python model_training/evaluation/local_serving/benchmark_inference.py --model unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit --requests 10 --max-tokens 80
```

## KServe note

On 8 GB RAM, full KServe stack (`Knative` + `Istio`) is often too heavy locally.
Recommended approach:

1. Finish this vLLM-on-K8s demo first.
2. Add KServe as a design + optional cloud PoC step.

## If pod fails with OOM

- reduce `MAX_MODEL_LEN` to `768` in `k8s/configmap.yaml`
- keep `MAX_NUM_SEQS=1`
- restart deployment:

```powershell
kubectl rollout restart deployment/llm-inference
```
