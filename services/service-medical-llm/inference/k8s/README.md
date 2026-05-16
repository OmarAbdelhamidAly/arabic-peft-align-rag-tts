# Kubernetes Deployment for service-inference

Lightweight K8s deployment for local LLM inference.

## Apply

```bash
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Access

```bash
kubectl port-forward svc/llm-inference-service 8001:8001
```

## Notes

- Configured for 8GB RAM laptops
- Uses 3B parameter model (4-bit quantized)
- Ready for migration to cloud with same manifests
