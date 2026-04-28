# 🧠 Arabic LLM Fine-Tuning Roadmap
### PEFT · Alignment · TTS · RAG — Arabic Medical Domain

> **Goal:** Master PEFT and alignment techniques (DPO, IPO, KTO, ORPO, RLHF, RLAIF), select the best model, fine-tune it for Arabic Text-to-Speech, then build a RAG pipeline that outputs speech — all following **Clean Architecture** with **Microservices**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Phase 1 — Dataset Selection](#-phase-1--dataset-selection)
- [Phase 2 — Base Model Selection](#-phase-2--base-model-selection)
- [Phase 3 — PEFT & Alignment Experiments](#-phase-3--peft--alignment-experiments)
- [Phase 4 — TTS Fine-Tuning](#-phase-4--tts-fine-tuning)
- [Phase 5 — RAG + Speech Pipeline](#-phase-5--rag--speech-pipeline)
- [Evaluation Strategy](#-evaluation-strategy)
- [Tools & Frameworks](#-tools--frameworks)
- [📚 Recommended Books](#-recommended-books)
- [Timeline](#-timeline)

---

## 🎯 Project Overview

A structured learning journey to master modern LLM fine-tuning in the **Arabic Medical domain**, built as a production-grade **microservices** system with **Clean Architecture**.

| Stage | What |
|-------|------|
| 1 | Data collection, translation & cleaning |
| 2 | SFT with LoRA / QLoRA |
| 3 | Alignment: DPO · IPO · KTO · ORPO · RLHF · RLAIF |
| 4 | Model selection via evaluation benchmarks |
| 5 | TTS fine-tuning (XTTS-v2) |
| 6 | RAG + Speech pipeline (end-to-end) |

---

## 🏗️ System Architecture

```
                        ┌──────────────────────────────────┐
                        │         User / Client            │
                        └────────────────┬─────────────────┘
                                         │ HTTP
                        ┌────────────────▼─────────────────┐
                        │       service-rag  :8003         │
                        │  Retrieval → LLM → TTS           │
                        └──────┬──────────────┬────────────┘
                               │              │
             ┌─────────────────▼──┐    ┌──────▼──────────────┐
             │ service-inference  │    │    service-tts       │
             │   :8001            │    │      :8002           │
             │ Fine-tuned LLM API │    │ Fine-tuned XTTS-v2   │
             └────────────────────┘    └──────────────────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │          Qdrant Vector DB           │
             │  (Arabic Medical Document Corpus)   │
             └────────────────────────────────────┘

             ┌────────────────────────────────────┐
             │      service-training  (offline)   │
             │  SFT → DPO → IPO → KTO → ORPO     │
             │  RLHF (PPO) → RLAIF               │
             └────────────────────────────────────┘

             ┌────────────────────────────────────┐
             │   service-vision  (🚧 planned)     │
             │  Medical image OCR / VQA           │
             │  Feeds text into service-rag       │
             └────────────────────────────────────┘
```

### Clean Architecture per Service

Each service strictly follows:
```
service-xxx/
├── app/
│   ├── domain/            ← Pure business logic. No external deps.
│   │   ├── entities/      ← Dataclasses / Pydantic models
│   │   └── interfaces/    ← Abstract base classes (ABCs)
│   ├── application/       ← Use cases (orchestrate domain)
│   │   └── use_cases/
│   ├── infrastructure/    ← External: HuggingFace, Qdrant, WandB
│   │   └── models/
│   └── interfaces/        ← Entry points: FastAPI / CLI
│       └── api/ or cli/
└── tests/
```

---

## 📁 Project Structure

```
arabic-peft-align-rag-tts/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── services/
│   ├── service-training/          ← PEFT + all alignment experiments
│   │   └── app/
│   │       ├── domain/interfaces/base_trainer.py
│   │       ├── application/use_cases/
│   │       ├── infrastructure/models/
│   │       └── interfaces/cli/
│   │
│   ├── service-inference/         ← Fine-tuned LLM REST API
│   │   └── app/
│   │       ├── domain/
│   │       ├── application/use_cases/
│   │       ├── infrastructure/models/
│   │       └── interfaces/api/app.py
│   │
│   ├── service-tts/               ← XTTS-v2 fine-tuning + synthesis API
│   │   └── app/
│   │       ├── domain/
│   │       ├── application/use_cases/
│   │       ├── infrastructure/models/
│   │       └── interfaces/api/app.py
│   │
│   ├── service-rag/               ← Retrieval + orchestration
│   │   └── app/
│   │       ├── domain/
│   │       ├── application/use_cases/
│   │       ├── infrastructure/
│   │       │   ├── vectorstore/   ← Qdrant adapter
│   │       │   └── embeddings/    ← multilingual-e5-large adapter
│   │       └── interfaces/api/app.py
│   │
│   └── service-vision/            ← 🚧 Planned: Medical VQA / OCR
│       └── README.md
│
├── shared/
│   ├── configs/                   ← Shared training / inference configs
│   ├── schemas/                   ← Shared Pydantic schemas
│   └── utils/                     ← Shared helpers (logging, metrics)
│
├── infrastructure/
│   ├── docker/                    ← Per-service Dockerfiles
│   └── docker-compose.yml         ← All services orchestration
│
├── data/
│   ├── raw/                       ← Original downloaded datasets
│   ├── processed/                 ← Cleaned & normalized
│   ├── sft/                       ← Instruction-following pairs
│   ├── preference/                ← Chosen / rejected pairs (alignment)
│   └── tts/                       ← Audio + transcripts
│
└── docs/                          ← Architecture diagrams, research notes
```

---

## 📊 Phase 1 — Dataset Selection

### Domain: **Arabic Medical**

| Reason | Detail |
|--------|--------|
| Scarce Arabic data | Fine-tuning impact is measurable and clear |
| Real RAG value | Retrieve from clinical guidelines & research |
| Real TTS value | Read reports, prescriptions, consultations aloud |

---

### 🗂️ SFT / Instruction Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **MedAraBench** | 19 medical specialties, MCQ, Arabic | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/MedAraBench) |
| **ArabicMedicalInstruct** | Arabic medical instruction pairs | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/ArabicMedicalInstruct) |
| **Shifaa Medical Consultations** | Patient-doctor Arabic consultations | [🤗 HuggingFace](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) |
| **Shifaa Mental Health** | Arabic mental health consultations | [🤗 HuggingFace](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations) |
| **MedArabiQ** | 7-task Arabic medical benchmark | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/MedArabiQ) |
| **Arabic Wikipedia** | General Arabic pretraining corpus | [🤗 HuggingFace](https://huggingface.co/datasets/wikipedia/viewer/20220301.ar) |

---

### 🏆 Alignment / Preference Datasets

| Dataset | Description | Type | Link |
|---------|-------------|------|------|
| **Arabic RLHF Preference** | Chosen/rejected Arabic pairs | DPO / RLHF | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/Arabic-preference-data-RLHF) |
| **Argilla DPO Mix Arabic** | 7k Arabic DPO-formatted pairs | DPO / IPO / ORPO | [🤗 HuggingFace](https://huggingface.co/datasets/2A2I/argilla-dpo-mix-7k-arabic) |
| **User Feedback DPO** | Real user feedback preference data | DPO / KTO | [🤗 HuggingFace](https://huggingface.co/datasets/farabi-lab/user-feedback-dpo) |
| **UltraFeedback Binarized** | High-quality preference data | All alignment | [🤗 HuggingFace](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |

---

### 🔊 TTS / Speech Datasets

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **Common Voice 17 (ar)** | Mozilla crowd-sourced Arabic | ~67 hrs | [🤗 HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) |
| **Arabic Speech Corpus** | Formal MSA, studio quality | 3.7 hrs | [🤗 HuggingFace](https://huggingface.co/datasets/arbml/arabic_speech_corpus) |
| **ClArTTS** | Classical Arabic TTS dataset | ~14 hrs | [🤗 HuggingFace](https://huggingface.co/datasets/MBZUAI/ClArTTS) |

---

## 🤖 Phase 2 — Base Model Selection

| # | Model | Params | Arabic Quality | VRAM (4-bit) | Notes |
|---|-------|--------|----------------|--------------|-------|
| ⭐ | **Qwen2.5-7B-Instruct** | 7B | ⭐⭐⭐⭐⭐ | ~6 GB | Best Arabic, 128k ctx |
| ⭐ | **LLaMA-3.1-8B-Instruct** | 8B | ⭐⭐⭐⭐ | ~6 GB | Best ecosystem |
| ✅ | **Gemma-2-9B-it** | 9B | ⭐⭐⭐⭐ | ~7 GB | Strong instruction following |
| ✅ | **Mistral-7B-v0.3** | 7B | ⭐⭐⭐ | ~5 GB | Fast, good baseline |
| 🔬 | **AceGPT-7B** | 7B | ⭐⭐⭐⭐⭐ | ~6 GB | Purpose-built Arabic |
| 🔬 | **ALLaM-7B** | 7B | ⭐⭐⭐⭐⭐ | ~6 GB | SDAIA Arabic model |

**HuggingFace Links:**
```
Qwen2.5-7B-Instruct   → https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
LLaMA-3.1-8B-Instruct → https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Gemma-2-9B-it         → https://huggingface.co/google/gemma-2-9b-it
Mistral-7B-v0.3       → https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
AceGPT-7B             → https://huggingface.co/FreedomIntelligence/AceGPT-7B
ALLaM-7B              → https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview
```

> 💡 **Start with `Qwen2.5-7B-Instruct`** — best Arabic out-of-the-box, 128k context, trains efficiently on a single GPU with QLoRA.

---

## 🔬 Phase 3 — PEFT & Alignment Experiments

### Alignment Techniques

| Technique | TRL Trainer | Data Format | Difficulty | Notes |
|-----------|-------------|-------------|------------|-------|
| **SFT + LoRA** | `SFTTrainer` | `{instruction, output}` | ⭐⭐ | Always first step |
| **DPO** | `DPOTrainer` | `{prompt, chosen, rejected}` | ⭐⭐⭐ | Most stable, start here |
| **IPO** | `DPOTrainer(loss_type="ipo")` | Same as DPO | ⭐⭐⭐ | DPO improvement |
| **KTO** | `KTOTrainer` | `{prompt, completion, label}` | ⭐⭐ | Easiest — binary feedback |
| **ORPO** | `ORPOTrainer` | `{prompt, chosen, rejected}` | ⭐⭐ | SFT + alignment in one shot |
| **RLHF (PPO)** | `PPOTrainer` | Reward model needed | ⭐⭐⭐⭐⭐ | Most powerful |
| **RLAIF** | `PPOTrainer` + LLM judge | LLM generates rewards | ⭐⭐⭐⭐ | No human annotations |

### Experiment Order

```
exp_01_sft_lora    → Baseline adapter
exp_02_sft_qlora   → Memory-efficient baseline
exp_03_dpo         → First alignment run
exp_04_ipo         → DPO variant
exp_05_kto         → Binary feedback alignment
exp_06_orpo        → Combined SFT+align
exp_07_rlhf_ppo    → Full RL alignment
exp_08_rlaif       → LLM-as-judge alignment
```

### QLoRA Config (Recommended Start)

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

---

## 🔊 Phase 4 — TTS Fine-Tuning

### TTS Model Choices

| Model | Quality | Fine-tunable | VRAM | Link |
|-------|---------|--------------|------|------|
| ⭐ **XTTS-v2** | ⭐⭐⭐⭐⭐ | ✅ | ~4 GB | [🤗 HuggingFace](https://huggingface.co/coqui/XTTS-v2) |
| ✅ **SpeechT5** | ⭐⭐⭐⭐ | ✅ | ~3 GB | [🤗 HuggingFace](https://huggingface.co/microsoft/speecht5_tts) |
| ✅ **MMS-TTS Arabic** | ⭐⭐⭐⭐ | ✅ | ~2 GB | [🤗 HuggingFace](https://huggingface.co/facebook/mms-tts-ara) |
| 🔬 **Parler-TTS** | ⭐⭐⭐⭐ | ✅ | ~4 GB | [🤗 HuggingFace](https://huggingface.co/parler-tts/parler-tts-mini-v1) |

> 💡 **Use `XTTS-v2`** — supports Arabic, voice cloning, and domain-specific fine-tuning.

---

## 🔍 Phase 5 — RAG + Speech Pipeline

```
User Query
    ↓
service-rag → Qdrant (multilingual-e5-large embeddings)
    ↓
Retrieved Arabic Medical Docs
    ↓
service-inference → Fine-tuned Qwen2.5-7B (generates answer)
    ↓
service-tts → Fine-tuned XTTS-v2 (synthesizes speech)
    ↓
🔊 Arabic Medical Audio Response
```

### RAG Stack

| Component | Tool |
|-----------|------|
| Document Loader | `PyMuPDF` + `LangChain` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Embedding Model | `intfloat/multilingual-e5-large` |
| Vector Store | `Qdrant` |
| LLM | Fine-tuned `Qwen2.5-7B` via `service-inference` |
| TTS | Fine-tuned `XTTS-v2` via `service-tts` |

---

## 📏 Evaluation Strategy

### LLM

| Metric | Tool |
|--------|------|
| ROUGE-L | `rouge-score` |
| BERTScore | `bert-score` |
| Arabic MMLU / MedAraBench | Benchmark suites |
| LLM-as-Judge (1–10) | GPT-4o / Claude |

### TTS

| Metric | Tool |
|--------|------|
| WER / CER | `jiwer` |
| MOS (1–5) | Human evaluation |
| RTF | Manual timing |

---

## 🛠️ Tools & Frameworks

```
Training      → transformers · trl · peft · accelerate · bitsandbytes
Experiment    → wandb
TTS           → TTS (Coqui) · speechbrain · librosa
RAG           → langchain · qdrant-client · sentence-transformers
Serving       → fastapi · uvicorn
Infra         → docker · docker-compose
Compute       → Kaggle (free T4×2) · Colab Pro · Vast.ai
```

---

## 📚 Recommended Books

### 🔥 Must-Read (Advanced & Recent)

| # | Book | Author | Year | Focus | Why Read It |
|---|------|--------|------|-------|-------------|
| 1 | **[LLM Engineer's Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072)** | Paul Iusztin & Maxime Labonne | 2024 | End-to-end LLM production | Best hands-on book for the full fine-tuning lifecycle: data → training → eval → deployment |
| 2 | **[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)** | Sebastian Raschka | 2024 | Deep architecture understanding | Builds LLM from zero in PyTorch — essential before fine-tuning anything |
| 3 | **[Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)** | Jay Alammar & Maarten Grootendorst | 2024 | Practical applications | Beautiful visual explanations + practical code |
| 4 | **[Foundations of Large Language Models](https://arxiv.org/abs/2501.09223)** | Tong Xiao & Jingbo Zhu | 2025 | Theory + RLHF alignment | Academic rigor on pre-training, RLHF, and evaluation — free on arXiv |
| 5 | **[A Practical Guide to RLHF](https://www.oreilly.com/library/view/a-practical-guide/9781098140304/)** | O'Reilly | 2024 | RLHF / PPO / Reward modeling | The only book dedicated 100% to alignment — covers reward modeling, PPO, preference learning |

### 📖 Supplementary Resources

| Resource | Type | Link |
|----------|------|------|
| **Maxime Labonne's LLM Course** | Free GitHub course | [github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course) |
| **Sebastian Raschka's "Ahead of AI"** | Newsletter / blog | [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com) |
| **Hugging Face TRL Docs** | Official docs | [huggingface.co/docs/trl](https://huggingface.co/docs/trl) |
| **Andrej Karpathy — Let's build GPT** | YouTube | [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| **RLHF Paper (Ouyang et al., 2022)** | Original InstructGPT paper | [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) |
| **DPO Paper (Rafailov et al., 2023)** | Direct Preference Optimization | [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) |
| **ORPO Paper (Hong et al., 2024)** | ORPO alignment | [arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691) |
| **KTO Paper (Ethayarajh et al., 2024)** | KTO alignment | [arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306) |

### 📑 Reading Order

```
1. Karpathy YouTube  →  Understand the fundamentals
2. Build LLM Scratch →  Code-level architecture knowledge
3. Hands-On LLMs     →  Practical fine-tuning applications
4. LLM Eng Handbook  →  Production pipeline (your main reference)
5. RLHF Guide        →  Deep alignment techniques
6. Foundations LLMs  →  Theory grounding (read alongside experiments)
```

---

## 📅 Timeline

| Week | Phase | Task |
|------|-------|------|
| 1–2 | Data | Collect, translate & clean Arabic medical dataset |
| 3 | Setup | Service-training env, base model loading, QLoRA config |
| 4–5 | SFT | LoRA + QLoRA SFT on Qwen2.5-7B + LLaMA-3.1-8B |
| 6 | Alignment | DPO + ORPO |
| 7 | Alignment | IPO + KTO |
| 8 | Alignment | RLHF (PPO) |
| 9 | Alignment | RLAIF (LLM-as-Judge) |
| 10 | Eval | Full comparison → select best model |
| 11–12 | TTS | Fine-tune XTTS-v2 on Arabic medical speech |
| 13 | RAG | Qdrant setup + embedding pipeline |
| 14 | Integration | Wire service-rag → service-inference → service-tts |
| 15 | Polish | End-to-end testing, documentation, demo |

---

## 🔗 Quick Reference

### Models
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [LLaMA-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [AceGPT-7B](https://huggingface.co/FreedomIntelligence/AceGPT-7B) · [ALLaM-7B](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview) · [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

### Datasets
- [MedAraBench](https://huggingface.co/datasets/FreedomIntelligence/MedAraBench) · [Shifaa Consultations](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) · [Arabic RLHF](https://huggingface.co/datasets/FreedomIntelligence/Arabic-preference-data-RLHF) · [Common Voice AR](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) · [ClArTTS](https://huggingface.co/datasets/MBZUAI/ClArTTS)

### Papers
- [DPO](https://arxiv.org/abs/2305.18290) · [ORPO](https://arxiv.org/abs/2403.07691) · [KTO](https://arxiv.org/abs/2402.01306) · [InstructGPT/RLHF](https://arxiv.org/abs/2203.02155) · [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314)

---

*Built with ❤️ as a learning journey through the full LLM fine-tuning + alignment + TTS + RAG stack.*
