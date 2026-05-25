# 🔁 Kubeflow Pipeline — Arabic Medical LLM

This folder contains the **Kubeflow Pipelines (KFP SDK v2)** definition and helper utilities for the full training workflow.

## Architecture

```text
Kubeflow Pipelines orchestrates the training workflow:

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                               Kubeflow Pipeline                                 │
  │                                                                                 │
  │  [Validate Data] → [SFT Training] ──┐                                           │
  │                                     ├──→ [6x Parallel Alignments]               │
  │  (pandas check)     (Unsloth SFT)   │    (DPO/IPO/KTO/ORPO/SimPO/RLOO)          │
  │                                     │                 ↓                         │
  │                                     └────────→ [Select Best Model]              │
  │                                                       ↓                         │
  │                                                 [Merge Model]                   │
  │                                                 (16-bit Merge)                  │
  │                                                       ↓                         │
  │                                             [Push to HuggingFace]               │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

## Folder Structure & Files

| File | Description |
|------|-------------|
| [pipeline.py](file:///e:/FineTuning/services/service-medical-llm/training/kubeflow/pipeline.py) | Full KFP v2 pipeline definition with 6 components and DAG structure. |
| [validate_pipeline.py](file:///e:/FineTuning/services/service-medical-llm/training/kubeflow/validate_pipeline.py) | Validation dry-run tool to check Python dependencies, dataset schemas, notebooks, and MLflow connection. |
| [test_sft.py](file:///e:/FineTuning/services/service-medical-llm/training/kubeflow/test_sft.py) | Local execution test harness for the SFT training stage via Papermill. |
| [test_alignment.py](file:///e:/FineTuning/services/service-medical-llm/training/kubeflow/test_alignment.py) | Local execution test harness for the alignment training stage (e.g. SimPO, DPO) via Papermill. |
| [training_pipeline.yaml](file:///e:/FineTuning/services/service-medical-llm/training/kubeflow/training_pipeline.yaml) | The compiled Kubeflow pipeline schema file, ready to upload to the Kubeflow cluster. |

## Quick Start

### 1. Compile the Pipeline
If you modify `pipeline.py`, compile the DAG to YAML from within the `kubeflow` directory:
```bash
cd kubeflow
python pipeline.py --compile --output training_pipeline.yaml
```

### 2. Submit to Kubeflow Cluster
Submit the compiled `training_pipeline.yaml` package to your Kubeflow Dashboard or via the CLI:
```bash
kfp run create \
  --experiment-name arabic-medical-llm \
  --pipeline-package-path training_pipeline.yaml
```

---

## Pipeline Components

| Step | Component Name | Action / Tool |
|------|----------------|---------------|
| 1 | `validate_dataset` | Validates dataset presence and structure using `pandas`. |
| 2 | `run_sft` | Trains LoRA adapters using `Unsloth` + TRL `SFTTrainer` inside SFT notebook via Papermill. |
| 3 | `run_alignment` | Runs 6 parallel alignment runs (DPO/IPO/KTO/ORPO/SimPO/RLOO) via Papermill and logs metrics to MLflow. |
| 4 | `select_best_model` | Queries MLflow to select the model adapter with the highest reward margin. |
| 5 | `merge_model` | Merges the best adapter back to 16-bit weights using `Unsloth`. |
| 6 | `push_to_huggingface` | Uploads the final merged model folder directly to Hugging Face Hub. |

---

## Post-Training → KServe Serving Deployment

Once the pipeline successfully finishes and pushes the merged model to Hugging Face or your local storage:

```bash
# Deploy with KServe (vLLM runtime)
kubectl apply -f ../experiments/evaluation/k8s/kserve_inferenceservice.yaml -n kserve-inference

# Check deployment status
kubectl get inferenceservice arabic-medical-llm -n kserve-inference

# Send a test inference request
curl http://arabic-medical-llm.kserve-inference.svc.cluster.local/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "arabic-medical-llm", "messages": [{"role": "user", "content": "ما هي أعراض السكري؟"}]}'
```
