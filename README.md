<div align="center">

# 🗣️ Arabic Mental Health LLM — Full Production Pipeline

### PEFT · Alignment · RAG · TTS Fine-Tuning · Microservices · MLOps

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![vLLM](https://img.shields.io/badge/vLLM-412991?style=flat-square&logoColor=white)](https://github.com/vllm-project/vllm)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC2626?style=flat-square&logoColor=white)](https://qdrant.tech)
[![Kubeflow](https://img.shields.io/badge/Kubeflow-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kubeflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![KServe](https://img.shields.io/badge/KServe-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kserve.github.io)

<br/>

> **End-to-end Arabic mental health voice assistant** — from raw data to a deployable, voice-enabled RAG system.
> Built with Clean Architecture and Microservices across three independently deployable services.

</div>

---

## 🗺️ System Architecture

```text
                    ┌─────────────────────────────────────────┐
                    │              User / Client               │
                    └──────────────────┬──────────────────────┘
                                       │  Arabic question (HTTP)
                    ┌─────────────────────────────────────────┐
                    │ 1. service-rag                          │
                    │  [Online API] Retrieve → Generate       │
                    │  [Offline] Fine-tune LLM for RAG        │
                    └──────┬────────────────────┬─────────────┘
                           │                    │
          ┌────────────────▼────────┐  ┌────────▼────────────────┐
          │ 2. service-medical-llm  │  │ 3. service-tts          │
          │                         │  │                         │
          │ [Online Inference API]  │  │ [Online Synthesis API]  │
          │  Serve 16-bit via vLLM  │  │  Serve XTTS-v2 Voice    │
          │                         │  │                         │
          │ [Offline Training]      │  │ [Offline Fine-Tuning]   │
          │  SFT → Parallel Align   │  │  Prepare → Train → Eval │
          └─────────────────────────┘  └─────────────────────────┘

                    ┌──────────────────────┐
                    │    Qdrant Vector DB   │
                    │  Mental Health Docs   │
                    └──────────────────────┘
```

---

## 📂 Repository Structure

```
arabic-peft-align-rag-tts/
│
├── services/
│   │
│   ├── service-medical-llm/              ← Core LLM Service (Training + Inference)
│   │   ├── training/
│   │   │   ├── kubeflow/                 ← KFP pipeline (Papermill + MLflow)
│   │   │   ├── experiments/              ← Research notebooks + trained artifacts ✅
│   │   │   │   ├── 01_sft/               ← SFT notebook + qwen_medical_arabic_lora
│   │   │   │   └── 02_post_training/     ← Alignment notebooks (6 parallel methods)
│   │   │   └── data/                     ← JSON preference datasets
│   │   └── inference/                    ← OpenAI-compatible REST API
│   │       ├── app/                      ← HuggingFaceLoader + FastAPI
│   │       └── k8s/                      ← vLLM Deployment + KServe InferenceService ✅
│   │
│   ├── service-tts/                      ← TTS fine-tuning + synthesis API
│   │   ├── app/                          ← XTTS-v2 adapter + FastAPI
│   │   ├── fine_tuning/                  ← prepare_data, train_xtts, evaluate
│   │   └── data/tts/wavs/                ← Arabic TTS audio dataset
│   │
│   └── service-rag/                      ← RAG orchestrator
│       ├── app/                          ← E5 embedder + Qdrant retriever + LLM client
│       └── scripts/ingest_documents.py
│
├── docker-compose.yml                    ← Orchestrates all online services + Qdrant
└── training_pipeline.yaml                ← Compiled Kubeflow Pipeline ✅
```

---

## 📊 Service Status

| # | Core Service | Modes | Status & Description |
|:-:|:-------------|:------|:---------------------|
| 1 | **`service-medical-llm`** | **Offline (Training)** <br> **Online (Inference)** | ✅ **Done** <br> 🟡 **Code ready** <br> _Unified service for fine-tuning via KFP and serving via vLLM._ |
| 2 | **`service-tts`** | **Offline (Fine-Tuning)** <br> **Online (API)** | 🔴 **Not started** <br> 🟡 **Skeleton ready** <br> _Voice cloning & synthesis using XTTS-v2._ |
| 3 | **`service-rag`** | **Offline (Fine-Tuning)** <br> **Online (API)** | 🔴 **Not started** <br> 🔴 **Stub** <br> _RAG orchestrator + LLM fine-tuning for complex mental health docs._ |

---

## 📈 Project Phases & Evolution

The development of this project is structured into three distinct phases, evolving from manual experimentation to a fully autonomous AI system. For a deep dive into the engineering mindset and "boring details" of each phase, read the dedicated documentation below:

| Phase | Description | Read More |
|:------|:------------|:----------|
| **Phase 1: The R&D Foundation** | Manual data generation, SFT, parallel alignments, local LLM-as-a-judge evaluation, and standard model serving. Focuses on proving model quality and discovering the best architecture. | [📖 Read Phase 1](docs/PHASE_1_RND.md) |
| **Phase 2: MLOps Automation** | Transforming the R&D notebooks into a robust Kubeflow Pipeline using Papermill and MLflow. Zero human intervention from dataset to Hugging Face deployment. | [📖 Read Phase 2](docs/PHASE_2_MLOPS.md) |
| **Phase 3: The Data Flywheel** | Active learning loop. The system monitors live RAG queries, detects weaknesses, generates targeted "hard" datasets, and triggers Phase 2 automatically to self-improve. | [📖 Read Phase 3](docs/PHASE_3_FLYWHEEL.md) |

---

## 📚 Service Documentation

لكل خدمة ملف `README` خاص بها يشرح تفاصيل هيكلتها وكيفية تشغيلها بشكل مستقل:

| Service | Documentation Link |
|:--------|:-------------------|
| **`service-medical-llm`** | [`services/service-medical-llm/README.md`](services/service-medical-llm/README.md) |
| **`service-tts`** | [`services/service-tts/README.md`](services/service-tts/README.md) |
| **`service-rag`** | [`services/service-rag/README.md`](services/service-rag/README.md) |

---

## 🔄 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE A — LLM Training                                   ✅ DONE   │
├─────────────────────────────────────────────────────────────────────┤
│  1. Collect & curate Arabic mental health data                            │
│  2. Generate SFT dataset + 6 preference alignment datasets          │
│  3. SFT notebook → qwen_medical_arabic_lora                         │
│  4. Parallel alignment via Papermill:                               │
│     DPO · IPO · KTO · ORPO · SimPO · RLOO                          │
│  5. MLflow tracking → select best adapter                           │
│  6. Merge to 16-bit → Push to HuggingFace Hub                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE B — TTS Fine-Tuning                                🔴 TODO   │
├─────────────────────────────────────────────────────────────────────┤
│  1. Select Arabic mental health speech dataset                            │
│  2. prepare_data.py → normalize + segment audio                     │
│  3. train_xtts.py → fine-tune XTTS-v2                               │
│  4. evaluate_tts.py → MOS + WER metrics                             │
│  5. Load checkpoint into service-tts API                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE C — Full System Integration                        🔴 TODO   │
├─────────────────────────────────────────────────────────────────────┤
│  1. Start service-medical-llm locally (vLLM / KServe)               │
│  2. POST /v1/chat/completions → verify LLM responses                │
│  3. Ingest Arabic mental health PDFs → Qdrant via ingest_documents.py     │
│  4. POST /rag/query → retrieval + grounded generation               │
│  5. Arabic text answer + synthesized audio response                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Training — Alignment Methods

Six alignment strategies run **in parallel** via Papermill, all tracked in MLflow:

| Method | Type | Key Property |
|:-------|:-----|:-------------|
| **DPO** | Offline RL | Direct preference optimization, no reward model |
| **IPO** | Offline RL | Identity mapping — avoids reward over-optimization |
| **KTO** | Binary feedback | Learns from unpaired human feedback signals |
| **ORPO** | Joint training | Merges SFT + alignment into a single training step |
| **SimPO** | Reference-free | Sequence-level reward, no reference model needed |
| **RLOO** | Online RL | Leave-one-out policy gradient, reduced variance |

Best adapter is selected by MLflow reward margin metrics → merged → pushed to Hub.

---

## ☸️ MLOps Stack

| Tool | Role | Path |
|:-----|:-----|:-----|
| **Kubeflow Pipelines** | Orchestrate: validate → SFT → align → merge → push | `training/kubeflow/pipeline.py` |
| **Papermill** | Execute Jupyter notebooks natively inside KFP steps | `Dockerfile` |
| **MLflow** | Track hyperparameters, loss curves, and reward margins | `pipeline.py` |
| **vLLM** | High-throughput LLM serving with PagedAttention | `inference/k8s/deployment.yaml` |
| **KServe** | Kubernetes-native model serving + autoscaling | `inference/k8s/inference_service.yaml` |

### Deployment Flow

```
Kubeflow Pipeline → Papermill executes SFT notebook
         ↓
6 parallel alignment notebooks tracked in MLflow
         ↓
Best adapter selected → merged to 16-bit → pushed to HuggingFace Hub
         ↓
KServe InferenceService (vLLM runtime) pulls from HuggingFace
         ↓
OpenAI-compatible endpoint at /v1/chat/completions
```

### Compile & Submit Pipeline

```bash
conda activate unsloth_env
cd services/service-medical-llm/training/kubeflow

# Compile
python pipeline.py --compile --output ../../../../training_pipeline.yaml

# Submit
kfp run create \
  --experiment-name arabic-llm \
  --pipeline-package-path ../../../../training_pipeline.yaml
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|:------|:-----------|
| **LLM Training** | Unsloth, TRL, PEFT, Transformers, bitsandbytes |
| **Alignment** | DPO, IPO, KTO, ORPO, SimPO, RLOO (via TRL) |
| **Pipeline Orchestration** | Kubeflow Pipelines, Papermill |
| **Experiment Tracking** | MLflow |
| **TTS Fine-Tuning** | Coqui TTS (XTTS-v2), librosa, soundfile |
| **Embeddings** | sentence-transformers `multilingual-e5-large` |
| **Vector DB** | Qdrant |
| **LLM Serving** | vLLM (PagedAttention), KServe |
| **APIs** | FastAPI, uvicorn, Pydantic v2 |
| **Containerization** | Docker, docker-compose |

---

## ⚡ Quick Start

### Run LLM inference service

```bash
cd services/service-medical-llm/inference
MODEL_PATH=../training/experiments/merged_model_16bit \
uvicorn app.interfaces.api.main:app --port 8001
```

### Start Qdrant only

```bash
docker-compose up qdrant
```

### Run all services

```bash
docker-compose up --build
```

### Ingest documents into Qdrant

```bash
python services/service-rag/scripts/ingest_documents.py \
  --source ./data/medical_docs/ \
  --collection arabic_mental_health
```

### Query the full pipeline

```bash
curl -X POST http://localhost:8003/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي أعراض الاكتئاب؟", "tts": true}'
```

---

## 🗒️ Notes

- All alignment experiments use the same SFT-initialized base — a fair apples-to-apples comparison.
- The `training_pipeline.yaml` is a **pre-compiled** KFP pipeline; you can submit it directly without re-running `pipeline.py`.
- `service-rag` uses an **OpenAI-compatible client** to call `service-medical-llm`, making it straightforward to swap in any other vLLM-served model.
- TTS is **optional per request** — set `"tts": false` in the RAG query to skip audio synthesis and return text only.