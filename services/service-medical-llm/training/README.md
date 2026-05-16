# 🧠 service-training — Offline Text-to-Text Training

> **Role:** Offline service — runs once, produces trained model artifacts.
> Orchestrates the full SFT → Alignment → Merge pipeline.

---

## ✅ Current Status

All training is **complete**. Adapters and evaluation results exist in `model_training/`.

| Phase | Adapter | Status |
|-------|---------|--------|
| SFT | `model_training/01_sft/qwen_medical_arabic_lora/` | ✅ Done |
| DPO | `model_training/02_post_training/qwen_medical_arabic_dpo/` | ✅ Done |
| IPO | `model_training/02_post_training/qwen_medical_arabic_ipo/` | ✅ Done |
| KTO | `model_training/02_post_training/qwen_medical_arabic_kto/` | ✅ Done |
| ORPO | `model_training/02_post_training/qwen_medical_arabic_orpo/` | ✅ Done |
| SimPO | `model_training/02_post_training/qwen_medical_arabic_simpo/` | ✅ Done |
| Evaluation | `model_training/evaluation/results/model_comparison.json` | ✅ Done |
| Merge | `model_training/merged_model_16bit/` | ⚠️ Run `select_and_merge.py` |

---

## 🏗️ Clean Architecture

```
service-training/
├── app/
│   ├── domain/                    ← Pure business logic (no ML imports)
│   │   ├── entities/
│   │   │   ├── training_job.py    ← TrainingJob dataclass + TrainingPhase enum
│   │   │   └── training_result.py ← TrainingResult dataclass
│   │   └── interfaces/
│   │       └── base_trainer.py    ← BaseTrainer ABC (contract for all trainers)
│   │
│   ├── application/               ← Use cases (orchestrate domain)
│   │   └── use_cases/
│   │       └── run_training_pipeline.py  ← RunTrainingPipelineUseCase
│   │
│   ├── infrastructure/            ← Concrete ML implementations
│   │   ├── trainers/
│   │   │   ├── sft_trainer.py     ← Unsloth + TRL SFTTrainer
│   │   │   └── alignment_trainer.py  ← TRL DPO/IPO/KTO/ORPO/ORPO trainers
│   │   └── merging/
│   │       └── model_merger.py    ← Unsloth save_pretrained_merged
│   │
│   └── interfaces/                ← Entry points
│       └── cli/
│           └── main.py            ← argparse CLI: sft | align | full | merge
│
├── notebooks/                     ← Original Jupyter training notebooks
│   ├── 01_sft/train_sft_optimized.ipynb
│   ├── 02_post_training/train_dpo.ipynb  (+ ipo, kto, orpo, simpo)
│   └── data_pipeline/
│
├── scripts/
│   └── select_and_merge.py        ← LLM-as-judge → pick best → merge
└── requirements.txt
```

---

## 🔄 Pipeline Flow

```
Raw Data (data/raw/)
      │
      ▼
[data_pipeline notebooks]
      │  generates SFT + alignment datasets
      ▼
[SFT Training] — Unsloth FastLanguageModel + TRL SFTTrainer
      │  output: qwen_medical_arabic_lora/ (LoRA adapter)
      ▼
[Alignment Training] — TRL DPOTrainer / KTOTrainer / ORPOTrainer
      │  output: 5 adapters (dpo, ipo, kto, orpo, simpo)
      ▼
[Evaluation] — LLM-as-judge (GPT-4o scores each model 0–10)
      │  output: model_comparison.json, model_by_category.csv
      ▼
[select_and_merge.py] — pick best adapter → merge into 16-bit
      │  output: merged_model_16bit/
      ▼
→ service-inference consumes merged_model_16bit/
```

---

## ⚙️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Base model | Qwen2.5-3B-Instruct (4-bit) | Best Arabic quality, fits in 6 GB VRAM |
| Fine-tuning | Unsloth FastLanguageModel | 2× faster than vanilla HuggingFace |
| LoRA | `peft` library | Train ~1% of params, keep base frozen |
| SFT | TRL `SFTTrainer` | Instruction following on medical Q&A |
| DPO/IPO | TRL `DPOTrainer` | Preference alignment from chosen/rejected pairs |
| KTO | TRL `KTOTrainer` | Binary feedback (easier data collection) |
| ORPO | TRL `ORPOTrainer` | SFT + alignment in a single training pass |
| Merging | Unsloth `save_pretrained_merged` | LoRA → standalone 16-bit model |
| Evaluation | LLM-as-judge (GPT-4o) | Automated quality scoring |

### Key Concepts to Understand

**QLoRA (4-bit + LoRA):**
- Base model loaded in 4-bit NormalFloat (NF4) — reduces VRAM ~4×
- LoRA adds small trainable matrices (rank r=16) to attention layers
- Only ~0.5% of parameters are trained
- Final adapter is only ~100 MB vs 6 GB for full model

**Alignment techniques:**
- **DPO**: Train directly on (prompt, chosen, rejected) — no reward model needed
- **IPO**: DPO variant with identity loss — more stable, less overfit
- **KTO**: Only needs binary (good/bad) labels — easiest data collection
- **ORPO**: Combines SFT + DPO in one pass — saves training time
- **SimPO**: DPO without reference model — simpler, often better

---

## 🚀 Usage

### Run the pipeline (if re-training):
```bash
cd services/service-training

# Full pipeline: SFT → all alignment phases
python -m app.interfaces.cli.main full

# SFT only
python -m app.interfaces.cli.main sft

# Specific alignment phases
python -m app.interfaces.cli.main align --phases dpo orpo simpo

# Merge best adapter
python -m app.interfaces.cli.main merge \
  --adapter ./outputs/02_post_training/qwen_medical_arabic_simpo \
  --output  ./outputs/merged_model_16bit
```

### Or use the notebooks directly:
```bash
jupyter notebook notebooks/01_sft/train_sft_optimized.ipynb
jupyter notebook notebooks/02_post_training/train_simpo.ipynb
python scripts/select_and_merge.py
```

---

## 📂 Key Output: `merged_model_16bit/`

This is what `service-inference` loads. It contains:
```
merged_model_16bit/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── model-00001-of-XXXX.safetensors  (sharded weights)
```

Point `service-inference` to this directory via `MODEL_PATH` env variable.
