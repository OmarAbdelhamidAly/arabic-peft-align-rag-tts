# 🔁 Kubeflow Pipeline — Arabic Medical LLM

This folder contains the **Kubeflow Pipelines (KFP SDK v2)** definition for the full training workflow.

## Architecture

```
Kubeflow Pipelines orchestrates the training workflow:

  ┌─────────────────────────────────────────────────────────────────┐
  │                   Kubeflow Pipeline                             │
  │                                                                 │
  │  [Validate Data] → [SFT Training] → [SimPO Alignment]          │
  │                                           ↓                    │
  │                                    [Merge Model]               │
  │                                           ↓                    │
  │                                 [Push to PVC / S3]             │
  │                                           ↓                    │
  │                              KServe InferenceService           │
  │                           (vLLM runtime, GPU, port 8080)       │
  └─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Full KFP v2 pipeline with 5 components |

## Quick Start

```bash
# 1. Install KFP SDK
pip install kfp==2.7.0

# 2. Compile pipeline to YAML
python pipeline.py --compile --output training_pipeline.yaml

# 3. Submit to Kubeflow cluster
kfp run create \
  --experiment-name arabic-llm \
  --pipeline-package-path training_pipeline.yaml
```

## Pipeline Components

| Step | Component | Tool |
|------|-----------|------|
| 1 | `validate_dataset` | pandas |
| 2 | `run_sft` | Unsloth + TRL SFTTrainer |
| 3 | `run_alignment` | TRL SimPO/DPO/IPO/KTO/ORPO |
| 4 | `merge_model` | Unsloth `save_pretrained_merged` |
| 5 | `push_model` | PVC copy or S3 upload (boto3) |

## After Training → KServe Deployment

Once the pipeline pushes the merged model to the PVC:

```bash
# Deploy with KServe (vLLM runtime)
kubectl apply -f ../k8s/inference_service.yaml -n kserve-inference

# Check status
kubectl get inferenceservice arabic-medical-llm -n kserve-inference

# Test endpoint
curl http://arabic-medical-llm.kserve-inference.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "arabic-medical-llm", "messages": [{"role": "user", "content": "ما هي أعراض السكري؟"}]}'
```

## MLOps Stack Summary

| Tool | Role |
|------|------|
| **Kubeflow Pipelines** | Orchestrate training runs, track experiments |
| **vLLM** | High-throughput LLM serving engine (PagedAttention + continuous batching) |
| **KServe** | Kubernetes-native model serving (autoscaling, canary deployments, monitoring) |
