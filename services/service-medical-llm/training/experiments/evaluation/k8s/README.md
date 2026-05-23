# 🚀 Serving the Arabic Medical LLM on Kubernetes

This folder provides Kubernetes manifests to serve your fine-tuned and aligned Arabic Medical LLM via the **vLLM Inference Engine** in two ways:
1. **Lightweight Deployment (Standard K8s):** Best for simple local setups on low-resource machines.
2. **KServe InferenceService:** Standard cloud/enterprise deployment using KServe CRDs.

---

## 🛠️ Prerequisites

Before executing the manifests, ensure you have:
*   **Docker Desktop** (WSL2 backend enabled on Windows).
*   `kubectl` CLI.
*   A local Kubernetes cluster like **Kind** or **Minikube**.
*   (Optional) **KServe Control Plane** installed on your cluster. For KServe setup details, refer to the [KServe Quickstart](https://kserve.github.io/website/master/get_started/).

---

## 🏗️ Method 1: Lightweight Deployment (Standard K8s)

This method runs vLLM directly as a standard Kubernetes Deployment and exposes it via a ClusterIP Service.

### 1. Apply Manifests
Run these commands from the `training` root directory:
```powershell
kubectl apply -f experiments/evaluation/k8s/configmap.yaml
kubectl apply -f experiments/evaluation/k8s/deployment.yaml
kubectl apply -f experiments/evaluation/k8s/service.yaml
```

### 2. Verify Pod Status
Wait until the pod is `Running` and the readiness probe passes:
```powershell
kubectl get pods -n default -w
```

### 3. Port Forwarding
Expose the service port to your localhost:
```powershell
kubectl port-forward service/llm-inference-service 8001:8001
```

### 4. Test the OpenAI-compatible Endpoint
Run this from your terminal:
```powershell
curl http://127.0.0.1:8001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    "messages": [
      {"role": "system", "content": "أنت طبيب نفسي خبير."},
      {"role": "user", "content": "أشعر بالقلق والتوتر الدائم."}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

---

## 🏭 Method 2: KServe Deployment (InferenceService)

KServe simplifies model lifecycle management, autoscaling, and versioning.

### 1. Edit the Model ID
Before applying the manifest, edit [kserve_inferenceservice.yaml](file:///E:/FineTuning/services/service-medical-llm/training/experiments/evaluation/k8s/kserve_inferenceservice.yaml) and replace `OmarAbdelhamidAly/Arabic-Medical-LLM-Qwen-3B` with your own Hugging Face model repository ID.

### 2. Apply KServe Manifest
```powershell
kubectl apply -f experiments/evaluation/k8s/kserve_inferenceservice.yaml
```

### 3. Track Status
Verify that the InferenceService URL is generated and the predictor status becomes `Ready`:
```powershell
kubectl get inferenceservice arabic-medical-llm
```

### 4. Port Forwarding & Testing
Port-forward the KServe ingress gateway (usually Knative ingress-gateway or KServe local gateway):
```powershell
# If using KServe local gateway:
kubectl port-forward -n kserve service/kserve-local-gateway 8080:80

# If using standard port forwarding on the pod directly:
kubectl port-forward service/arabic-medical-llm-predictor-default 8080:8080
```

Then send a test request:
```powershell
curl http://127.0.0.1:8080/v1/chat/completions `
  -H "Host: arabic-medical-llm.default.example.com" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "OmarAbdelhamidAly/Arabic-Medical-LLM-Qwen-3B",
    "messages": [
      {"role": "system", "content": "أنت طبيب نفسي خبير."},
      {"role": "user", "content": "أشعر بالقلق والتوتر الدائم."}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

---

## 🧹 Cleanup

To delete deployments and clean up your cluster:

```powershell
# For Method 1:
kubectl delete -f experiments/evaluation/k8s/service.yaml
kubectl delete -f experiments/evaluation/k8s/deployment.yaml
kubectl delete -f experiments/evaluation/k8s/configmap.yaml

# For Method 2:
kubectl delete -f experiments/evaluation/k8s/kserve_inferenceservice.yaml
```
