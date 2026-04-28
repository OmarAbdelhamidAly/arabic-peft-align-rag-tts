# 🧠 Arabic LLM Fine-Tuning Roadmap
### PEFT · Alignment · TTS · RAG — Arabic Medical Domain

> **Goal:** Master PEFT and alignment techniques (DPO, IPO, KTO, ORPO, RLHF, RLAIF), select the best model, fine-tune it for Arabic Text-to-Speech, then build a RAG pipeline that outputs speech.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Phase 1 — Dataset Selection](#-phase-1--dataset-selection)
- [Phase 2 — Base Model Selection](#-phase-2--base-model-selection)
- [Phase 3 — PEFT & Alignment Experiments](#-phase-3--peft--alignment-experiments)
- [Phase 4 — TTS Fine-Tuning](#-phase-4--tts-fine-tuning)
- [Phase 5 — RAG + Speech Pipeline](#-phase-5--rag--speech-pipeline)
- [Evaluation Strategy](#-evaluation-strategy)
- [Tools & Frameworks](#-tools--frameworks)
- [Timeline](#-timeline)

---

## 🎯 Project Overview

This project is a structured learning journey to master modern LLM fine-tuning techniques in the **Arabic Medical domain**. The pipeline goes through:

1. **Data Collection & Translation** → Arabic Medical Instruction Dataset
2. **SFT with PEFT** (LoRA / QLoRA) → Domain-adapted base model
3. **Alignment Experiments** → DPO · IPO · KTO · ORPO · RLHF · RLAIF
4. **Model Selection** → Best model via evaluation benchmarks
5. **TTS Fine-Tuning** → Arabic domain-specific speech synthesis
6. **RAG + TTS Pipeline** → Retrieval-augmented speech output system

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Full Pipeline                          │
│                                                          │
│  Arabic Medical Corpus                                   │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────┐      ┌──────────────────────┐      │
│  │  SFT (LoRA /    │ ───► │  Alignment Stage     │      │
│  │  QLoRA)         │      │  DPO/IPO/KTO/ORPO    │      │
│  └─────────────────┘      │  RLHF/RLAIF          │      │
│                            └──────────┬───────────┘      │
│                                       │                  │
│                            ┌──────────▼───────────┐      │
│                            │   Model Evaluation   │      │
│                            │   & Selection        │      │
│                            └──────────┬───────────┘      │
│                                       │                  │
│                    ┌──────────────────▼──────────────┐   │
│                    │      TTS Fine-Tuning             │   │
│                    │   (XTTS-v2 / SpeechT5)          │   │
│                    └──────────────────┬──────────────┘   │
│                                       │                  │
│  User Query ──► RAG Retrieval ──► Fine-tuned LLM ──►    │
│                                       │                  │
│                            Fine-tuned TTS ──► 🔊 Audio  │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Phase 1 — Dataset Selection

### Domain: Arabic Medical

The medical domain was chosen because:
- High-quality Arabic medical data is scarce → fine-tuning impact is clear
- RAG is genuinely useful (retrieving from clinical documents)
- TTS adds real value (reading reports, consultations aloud)

---

### 🗂️ SFT / Instruction Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **MedAraBench** | 19 medical specialties, MCQ format, Arabic | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/MedAraBench) |
| **ArabicMedicalInstruct** | Arabic medical instruction-following pairs | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/ArabicMedicalInstruct) |
| **Shifaa Medical Consultations** | Large Arabic patient-doctor consultations | [🤗 HuggingFace](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) |
| **Shifaa Mental Health** | Arabic mental health consultations | [🤗 HuggingFace](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations) |
| **MedArabiQ** | 7-task Arabic medical benchmark dataset | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/MedArabiQ) |
| **Arabic Wikipedia** | General Arabic pretraining corpus | [🤗 HuggingFace](https://huggingface.co/datasets/wikipedia/viewer/20220301.ar) |

---

### 🏆 Alignment / Preference Datasets

| Dataset | Description | Alignment Type | Link |
|---------|-------------|----------------|------|
| **Arabic RLHF Preference** | Chosen/rejected Arabic response pairs | DPO / RLHF | [🤗 HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/Arabic-preference-data-RLHF) |
| **Argilla DPO Mix Arabic** | 7k Arabic DPO-formatted pairs | DPO / IPO / ORPO | [🤗 HuggingFace](https://huggingface.co/datasets/2A2I/argilla-dpo-mix-7k-arabic) |
| **User Feedback DPO** | Real user feedback preference data | DPO / KTO | [🤗 HuggingFace](https://huggingface.co/datasets/farabi-lab/user-feedback-dpo) |
| **UltraFeedback Arabic** | Translated UltraFeedback for Arabic | All alignment | [🤗 HuggingFace](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |

---

### 🔊 TTS / Speech Datasets

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **Common Voice 17 (Arabic)** | Mozilla crowd-sourced Arabic speech | ~67 hours | [🤗 HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) |
| **Arabic Speech Corpus** | Formal Arabic (MSA), studio quality | 3.7 hours | [🤗 HuggingFace](https://huggingface.co/datasets/arbml/arabic_speech_corpus) |
| **SADA Arabic Speech** | Diverse Arabic dialects speech | ~1000 hours | [🤗 HuggingFace](https://huggingface.co/datasets/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) |
| **ClArTTS** | Classical Arabic TTS dataset | ~14 hours | [🤗 HuggingFace](https://huggingface.co/datasets/MBZUAI/ClArTTS) |

---

### 📦 RAG Corpus (Documents for Retrieval)

| Source | Format | Content |
|--------|--------|---------|
| Arabic medical textbooks (PDF) | PDF → chunks | Clinical guidelines |
| Egyptian MOH guidelines | PDF | Public health protocols |
| PubMed Arabic abstracts | Text | Medical research |
| WHO Arabic publications | PDF | Global health docs |

> **Tip:** Use `PyMuPDF` + `LangChain` to chunk PDFs, then embed with `CAMeL-Lab/bert-base-arabic-camelbert-msa` or `intfloat/multilingual-e5-large`.

---

## 🤖 Phase 2 — Base Model Selection

### Recommended Models for LLM Fine-Tuning

| # | Model | Params | Arabic Quality | VRAM (4-bit) | Recommendation |
|---|-------|--------|----------------|--------------|----------------|
| ⭐ | **Qwen2.5-7B-Instruct** | 7B | ⭐⭐⭐⭐⭐ | ~6 GB | **Best overall for Arabic** |
| ⭐ | **LLaMA-3.1-8B-Instruct** | 8B | ⭐⭐⭐⭐ | ~6 GB | **Best ecosystem & community** |
| ✅ | **Gemma-2-9B-it** | 9B | ⭐⭐⭐⭐ | ~7 GB | Strong instruction following |
| ✅ | **Mistral-7B-Instruct-v0.3** | 7B | ⭐⭐⭐ | ~5 GB | Fast training, good baseline |
| 🔬 | **AceGPT-7B** | 7B | ⭐⭐⭐⭐⭐ | ~6 GB | Purpose-built Arabic LLM |
| 🔬 | **ALLaM-7B** | 7B | ⭐⭐⭐⭐⭐ | ~6 GB | SDAIA Arabic model |

### HuggingFace Links

```
Qwen2.5-7B-Instruct    → https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
LLaMA-3.1-8B-Instruct  → https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Gemma-2-9B-it          → https://huggingface.co/google/gemma-2-9b-it
Mistral-7B-v0.3        → https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
AceGPT-7B              → https://huggingface.co/FreedomIntelligence/AceGPT-7B
ALLaM-7B               → https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview
```

### 💡 My Recommendation

> **Start with `Qwen2.5-7B-Instruct`** — it has the best Arabic language understanding out of the box, supports long context (128k tokens), and trains efficiently with QLoRA on a single GPU. Run all alignment experiments on it, then compare against `LLaMA-3.1-8B` as a baseline.

---

## 🔬 Phase 3 — PEFT & Alignment Experiments

### 3.1 Supervised Fine-Tuning (SFT)

```python
# Using TRL + PEFT
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

**Dataset format:**
```json
{
  "instruction": "اشرح أعراض مرض السكري بشكل مبسط",
  "input": "",
  "output": "مرض السكري يتميز بارتفاع مستوى السكر في الدم..."
}
```

---

### 3.2 Alignment Techniques Overview

| Technique | Trainer (TRL) | Data Format | Difficulty | Notes |
|-----------|--------------|-------------|------------|-------|
| **DPO** | `DPOTrainer` | `{prompt, chosen, rejected}` | ⭐⭐⭐ | Start here — stable & effective |
| **IPO** | `DPOTrainer(loss_type="ipo")` | `{prompt, chosen, rejected}` | ⭐⭐⭐ | Improvement over DPO |
| **KTO** | `KTOTrainer` | `{prompt, completion, label}` | ⭐⭐ | Easiest — binary good/bad |
| **ORPO** | `ORPOTrainer` | `{prompt, chosen, rejected}` | ⭐⭐ | SFT + alignment in one step |
| **RLHF (PPO)** | `PPOTrainer` | Reward model needed | ⭐⭐⭐⭐⭐ | Most powerful, most complex |
| **RLAIF** | `PPOTrainer` + LLM judge | LLM generates rewards | ⭐⭐⭐⭐ | No human annotations needed |

---

### 3.3 DPO Example

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    beta=0.1,           # KL penalty coefficient
    loss_type="sigmoid", # DPO loss
    output_dir="./dpo-arabic-medical",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

### 3.4 ORPO Example (Recommended for Start)

```python
from trl import ORPOTrainer, ORPOConfig

# ORPO = SFT + DPO in one step (no ref model needed!)
training_args = ORPOConfig(
    lambda_orpo=0.1,
    output_dir="./orpo-arabic-medical",
    num_train_epochs=3,
)

trainer = ORPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

### 3.5 RLAIF Setup (LLM as Judge)

```python
# Use GPT-4o or Claude to generate preference labels
def generate_preference_label(prompt, response_a, response_b):
    """
    Ask an LLM to judge which response is better.
    Returns: {"chosen": response_a, "rejected": response_b} or vice versa
    """
    judge_prompt = f"""
    Given the medical question: {prompt}
    
    Response A: {response_a}
    Response B: {response_b}
    
    Which response is more accurate, helpful, and safe?
    Answer with only "A" or "B".
    """
    # Call GPT-4o / Claude API
    ...
```

---

### 3.6 Experiment Tracking

Use **Weights & Biases** to track all experiments:

```python
import wandb
wandb.init(project="arabic-medical-finetuning", name="dpo-qwen2.5-7b")
```

Run Matrix:

```
Model: Qwen2.5-7B-Instruct
Experiments:
  ├── exp_01_sft_lora
  ├── exp_02_sft_qlora
  ├── exp_03_dpo
  ├── exp_04_ipo
  ├── exp_05_kto
  ├── exp_06_orpo
  ├── exp_07_rlhf_ppo
  └── exp_08_rlaif
```

---

## 🔊 Phase 4 — TTS Fine-Tuning

### Recommended TTS Models

| Model | Language | Quality | Fine-tunable | VRAM | Link |
|-------|----------|---------|--------------|------|------|
| ⭐ **XTTS-v2** | 16 langs incl. Arabic | ⭐⭐⭐⭐⭐ | ✅ Yes | ~4 GB | [🤗 HuggingFace](https://huggingface.co/coqui/XTTS-v2) |
| ✅ **SpeechT5** | Multilingual | ⭐⭐⭐⭐ | ✅ Yes | ~3 GB | [🤗 HuggingFace](https://huggingface.co/microsoft/speecht5_tts) |
| ✅ **MMS-TTS Arabic** | Arabic native | ⭐⭐⭐⭐ | ✅ Yes | ~2 GB | [🤗 HuggingFace](https://huggingface.co/facebook/mms-tts-ara) |
| 🔬 **Parler-TTS** | Multilingual | ⭐⭐⭐⭐ | ✅ Yes | ~4 GB | [🤗 HuggingFace](https://huggingface.co/parler-tts/parler-tts-mini-v1) |

### 💡 TTS Recommendation

> **Use `XTTS-v2`** — best Arabic quality, supports voice cloning, and you can fine-tune it on your domain-specific Arabic speech data (medical terminology pronunciation).

### TTS Fine-Tuning Pipeline

```
Arabic Medical Text (from LLM) 
        │
        ▼
Common Voice Arabic + ClArTTS
        │ (domain adaptation)
        ▼
XTTS-v2 Fine-Tuning
  - Learning rate: 1e-5
  - Epochs: 50-100
  - Dataset: ~10-20 hours minimum
        │
        ▼
Domain-Specific Arabic Medical TTS
```

---

## 🔍 Phase 5 — RAG + Speech Pipeline

### Architecture

```python
# Full Pipeline
def arabic_medical_rag_tts(user_query: str) -> bytes:
    """
    1. Retrieve relevant medical documents
    2. Generate answer with fine-tuned LLM
    3. Convert answer to speech with fine-tuned TTS
    """
    
    # Step 1: Retrieval
    retrieved_docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Step 2: LLM Generation (Fine-tuned model)
    prompt = f"""بناءً على المعلومات التالية:
{context}

أجب على السؤال التالي بدقة طبية:
{user_query}
"""
    answer = finetuned_llm.generate(prompt)
    
    # Step 3: TTS Synthesis (Fine-tuned TTS)
    audio = finetuned_tts.synthesize(answer, language="ar")
    
    return audio  # Return audio bytes
```

### RAG Components

| Component | Tool | Purpose |
|-----------|------|---------|
| **Document Loader** | `PyMuPDF` / `LangChain` | Load PDFs |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Chunk Arabic text |
| **Embedding Model** | `intfloat/multilingual-e5-large` | Arabic embeddings |
| **Vector Store** | `Qdrant` / `FAISS` | Store & retrieve chunks |
| **LLM** | Fine-tuned `Qwen2.5-7B` | Generate answers |
| **TTS** | Fine-tuned `XTTS-v2` | Synthesize speech |

---

## 📏 Evaluation Strategy

### LLM Evaluation

| Metric | Tool | What it measures |
|--------|------|-----------------|
| **ROUGE-L** | `rouge-score` | N-gram overlap with reference |
| **BERTScore** | `bert-score` | Semantic similarity |
| **Arabic MMLU** | Benchmark | General Arabic knowledge |
| **MedAraBench** | Benchmark | Arabic medical knowledge |
| **LLM-as-Judge** | GPT-4o / Claude | Response quality (1-10) |

### TTS Evaluation

| Metric | Tool | What it measures |
|--------|------|-----------------|
| **WER** | `jiwer` | Word Error Rate (ASR feedback) |
| **MOS** | Human eval | Mean Opinion Score (1-5) |
| **CER** | `jiwer` | Character Error Rate |
| **RTF** | Manual | Real-Time Factor (speed) |

---

## 🛠️ Tools & Frameworks

```
Fine-tuning:
  ├── 🤗 transformers      → Model loading & training
  ├── 🤗 trl               → DPO, PPO, KTO, ORPO trainers
  ├── 🤗 peft              → LoRA, QLoRA adapters
  ├── 🤗 datasets          → Dataset loading
  ├── bitsandbytes         → 4-bit / 8-bit quantization
  └── accelerate           → Multi-GPU / gradient checkpointing

Experiment Tracking:
  └── wandb                → Metrics, loss curves, model comparison

TTS:
  ├── TTS (Coqui)          → XTTS-v2 fine-tuning
  ├── speechbrain          → Speech processing
  └── librosa              → Audio analysis

RAG:
  ├── langchain            → RAG orchestration
  ├── qdrant-client        → Vector database
  ├── sentence-transformers → Embedding models
  └── PyMuPDF              → PDF processing

Compute:
  ├── Kaggle (free T4 x2)  → SFT / DPO experiments
  ├── Google Colab Pro     → Quick experiments
  └── Vast.ai / RunPod     → Longer training runs
```

---

## 📅 Timeline

| Week | Phase | Task |
|------|-------|------|
| 1–2 | Data | Collect, translate & clean Arabic medical dataset |
| 3 | Setup | Environment, base model loading, QLoRA config |
| 4–5 | SFT | LoRA + QLoRA SFT on Qwen2.5-7B + LLaMA-3.1-8B |
| 6 | Alignment | DPO + ORPO experiments |
| 7 | Alignment | IPO + KTO experiments |
| 8 | Alignment | RLHF (PPO) experiment |
| 9 | Alignment | RLAIF (LLM-as-Judge) experiment |
| 10 | Evaluation | Full model comparison, select winner |
| 11–12 | TTS | Fine-tune XTTS-v2 on Arabic medical speech |
| 13 | RAG | Build retrieval pipeline + vector DB |
| 14 | Integration | Connect RAG → LLM → TTS end-to-end |
| 15 | Polish | Evaluation, documentation, demo |

---

## 📁 Project Structure

```
FineTuning/
│
├── README.md                    ← You are here
│
├── data/
│   ├── raw/                     ← Downloaded datasets
│   ├── processed/               ← Cleaned & formatted
│   ├── sft/                     ← Instruction pairs
│   ├── preference/              ← Chosen/rejected pairs
│   └── tts/                     ← Audio + transcripts
│
├── experiments/
│   ├── exp_01_sft_lora/
│   ├── exp_02_sft_qlora/
│   ├── exp_03_dpo/
│   ├── exp_04_ipo/
│   ├── exp_05_kto/
│   ├── exp_06_orpo/
│   ├── exp_07_rlhf/
│   └── exp_08_rlaif/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_alignment_dpo.ipynb
│   ├── 04_alignment_orpo.ipynb
│   ├── 05_alignment_rlhf.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_tts_finetuning.ipynb
│   └── 08_rag_pipeline.ipynb
│
├── src/
│   ├── data/
│   │   ├── downloader.py        ← Download datasets
│   │   ├── translator.py        ← EN→AR translation
│   │   └── formatter.py        ← Format for training
│   ├── training/
│   │   ├── sft_trainer.py
│   │   ├── dpo_trainer.py
│   │   ├── orpo_trainer.py
│   │   └── ppo_trainer.py
│   ├── evaluation/
│   │   ├── llm_eval.py
│   │   └── tts_eval.py
│   └── pipeline/
│       ├── rag_retriever.py
│       ├── tts_synthesizer.py
│       └── full_pipeline.py
│
├── configs/
│   ├── lora_config.yaml
│   ├── dpo_config.yaml
│   └── tts_config.yaml
│
└── models/
    ├── checkpoints/             ← Training checkpoints
    └── final/                  ← Best model weights
```

---

## 🔗 Quick Reference Links

### Models
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Gemma-2-9B-it](https://huggingface.co/google/gemma-2-9b-it)
- [AceGPT-7B (Arabic)](https://huggingface.co/FreedomIntelligence/AceGPT-7B)
- [ALLaM-7B (Arabic)](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview)
- [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
- [MMS-TTS Arabic](https://huggingface.co/facebook/mms-tts-ara)

### Datasets
- [MedAraBench](https://huggingface.co/datasets/FreedomIntelligence/MedAraBench)
- [ArabicMedicalInstruct](https://huggingface.co/datasets/FreedomIntelligence/ArabicMedicalInstruct)
- [Shifaa Medical Consultations](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations)
- [Arabic RLHF Preference Data](https://huggingface.co/datasets/FreedomIntelligence/Arabic-preference-data-RLHF)
- [Argilla DPO Mix Arabic](https://huggingface.co/datasets/2A2I/argilla-dpo-mix-7k-arabic)
- [Common Voice Arabic](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- [ClArTTS](https://huggingface.co/datasets/MBZUAI/ClArTTS)

### Libraries
- [TRL (Hugging Face)](https://github.com/huggingface/trl)
- [PEFT (Hugging Face)](https://github.com/huggingface/peft)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [LangChain](https://python.langchain.com)

---

## 💡 Key Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Domain | Arabic Medical | High impact, scarce data, real RAG + TTS use case |
| Base LLM | Qwen2.5-7B-Instruct | Best Arabic support, 128k context, efficient |
| First alignment | ORPO → DPO | ORPO is one-step (no ref model), DPO is most stable |
| TTS Model | XTTS-v2 | Best Arabic quality, voice cloning, fine-tunable |
| Vector DB | Qdrant | Production-grade, already in your stack |
| Embedding | multilingual-e5-large | Best Arabic semantic search |

---

*Built with ❤️ as a learning journey through the full LLM fine-tuning stack.*
