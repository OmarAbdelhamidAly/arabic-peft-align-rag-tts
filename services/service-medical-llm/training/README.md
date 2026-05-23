# 🧠 Arabic Medical LLM: Training & Post-Training Alignment Pipeline

This directory contains the complete end-to-end **Offline Training Pipeline** for the Arabic Medical and Mental Health LLM. It manages the entire lifecycle of model creation—from synthetic data generation and Supervised Fine-Tuning (SFT) to parallel preference alignment, evaluation, automated model selection, merging, and deployment.

---

## 🏗️ Technical Architecture & Workflow

The architecture is built for **reproducibility**, **scalability**, and **resource efficiency** (optimized to run on consumer GPUs like the RTX 3070 Ti / 3060).

```
                              [ 1. Raw Data Generation ]
                                         │
                                         ▼
                            [ 2. ChatML Dataset Prep ]
                                         │
                                         ▼
                         [ 3. Validate Dataset Component ]
                                         │
                                         ▼
                       [ 4. Supervised Fine-Tuning (SFT) ]
                         (Qwen2.5-3B-Instruct + 4-bit LoRA)
                                         │
                 ┌────────┬────────┬─────┴─────┬────────┬────────┐
                 ▼        ▼        ▼           ▼        ▼        ▼
              [DPO]    [IPO]    [KTO]       [ORPO]   [SimPO]   [RLOO]
                 │        │        │           │        │        │
                 └────────┴────────┴─────┬─────┴────────┴────────┘
                                         ▼
                             [ 5. Select Best Model ]
                               (Reward Margin Query)
                                         │
                                         ▼
                            [ 6. Merge Model (16-bit) ]
                                         │
                                         ▼
                             [ 7. Push to HuggingFace ]
```

### 1. Kubeflow & Papermill Integration
Rather than refactoring interactive Jupyter Notebooks into monolithic python scripts, the pipeline uses **Papermill** inside **Kubeflow Pipelines (KFP)**. 
- **Interactive Development:** Notebooks are used for exploration and prototyping.
- **Production Execution:** Kubeflow executes the notebooks headlessly inside Docker containers, injecting parameters (like `DATASET_PATH`, `LEARNING_RATE`, `BATCH_SIZE`) dynamically into the top-level cell tagged with `"parameters"`.
- **Artifact Preservation:** The executed notebooks (containing output charts, logs, and stats) are saved as pipeline artifacts for easy inspection.

### 2. Supervised Fine-Tuning (SFT)
- **Base Model:** `Qwen/Qwen2.5-3B-Instruct` (bnb 4-bit quantized).
- **Optimization Framework:** `Unsloth` (enables 2x faster training and fits inside 8GB VRAM).
- **Format:** `ChatML` template (`<|im_start|>system...`, `<|im_start|>user...`, `<|im_start|>assistant...`).

### 3. Parallel Post-Training Preference Alignment
To steer the model towards empathetic, medically accurate, and safe Arabic responses, we run **6 preference alignment algorithms in parallel**:
1. **DPO (Direct Preference Optimization):** Directly optimizes the policy model using pairwise preference data without a separate reward model.
2. **IPO (Identity Preference Optimization):** Adds a regularizer to DPO to prevent the model from degenerating or overfitting.
3. **KTO (Kahneman-Tversky Optimization):** Works on unpaired data (binary thumbs up/down) using utility theory.
4. **ORPO (Odds Ratio Preference Optimization):** Integrates preference alignment directly into the SFT loss, eliminating the need for a two-phase SFT -> DPO process.
5. **SimPO (Simple Preference Optimization):** A reference-free algorithm that optimizes the length-normalized margin between chosen and rejected responses directly.
6. **RLOO (Reinforcement Learning with Leave-One-Out):** An on-policy RL method that optimizes generated completions using a reward model.

---

## 📂 Detailed Directory Structure

```text
training/
├── kubeflow/                     # Kubeflow pipeline orchestrator
│   ├── pipeline.py               # KFP pipeline definition & component definitions
│   ├── test_pipeline.py          # Quick pipeline validation test
│   ├── test_sft.py               # Local execution harness for the SFT component
│   └── validate_pipeline.py      # Dry-run checklist verifying files, paths, and env config
├── experiments/                  # Research, notebooks, and model configs
│   ├── data_pipeline/            # Generates synthetic Arabic datasets
│   │   ├── 01_data_generator.ipynb           # Base SFT data generator
│   │   ├── 02_alignment_data_generator.ipynb # DPO/Preference generator
│   │   ├── 03_alignment_data_generator_hard.ipynb # Hard preference mining
│   │   └── processed_data/                   # Output folder for generated datasets
│   ├── 01_sft/                   # Supervised Fine-Tuning folder
│   │   ├── train_sft_optimized.ipynb         # Main fine-tuning notebook
│   │   ├── OPTIMIZATION_REPORT.md            # Hardware optimization log
│   │   └── QUICK_REFERENCE.md                # Quick commands cheatsheet
│   ├── 02_post_training/         # Preference alignment folder
│   │   ├── train_dpo.ipynb                   # DPO notebook
│   │   ├── train_ipo.ipynb                   # IPO notebook
│   │   ├── train_kto.ipynb                   # KTO notebook
│   │   ├── train_orpo.ipynb                  # ORPO notebook
│   │   ├── train_simpo.ipynb                 # SimPO notebook
│   │   └── train_rloo_part1_reward_model.ipynb / train_rloo_part2_policy.ipynb
│   ├── evaluation/               # Model comparison, judge-LLM & serving
│   │   ├── 01_compare_5_models.ipynb         # Evaluation workbook
│   │   ├── local_serving/                    # Inference benchmark scripts
│   │   └── k8s/                              # Kubernetes manifests for vLLM deployment
│   ├── select_and_merge.py       # Standalone script to select and merge LoRA adapters
│   ├── merge_winner.py           # Helper script to load PEFT adapter and merge to 16bit
│   └── upload_to_huggingface.py  # Standalone hub publisher
├── data/                         # Hard links/copies of alignment datasets (JSON)
├── outputs/                      # Local testing output (adapters, logs, executed notebooks)
├── requirements.txt              # Standardized python dependencies
├── Dockerfile                    # Container definition for pipeline components
└── training_pipeline.yaml        # Compiled Kubeflow pipeline schema (auto-generated)
```

---

## ⚙️ Core Parameters & Environment Setup

### 1. Hardware Requirements & Optimization
- **Recommended GPU:** NVIDIA RTX 3070 Ti Laptop / RTX 3060 (minimum 8GB VRAM).
- **Quantization:** bitsandbytes 4-bit Double Quantization (`load_in_4bit=True`).
- **Optimizer:** `adamw_8bit` (reduces memory consumption by ~60% compared to standard AdamW).
- **Gradient Checkpointing:** Configured via `unsloth` (`use_gradient_checkpointing="unsloth"`) to reduce activation memory.

### 2. Local Environment Setup
To run scripts locally on Windows, set up a dedicated Conda environment:

```cmd
# Create environment
conda create -n unsloth_env python=3.10 -y
conda activate unsloth_env

# Install PyTorch via Conda (Prevents DLL load errors on Windows)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Accelerators (Must be done without dependencies to prevent torch downgrades)
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 --no-deps

# Install main requirements
pip install -r requirements.txt

# Install specific tokenizers version required by Unsloth
pip install "tokenizers==0.21.4" --no-deps
```

---

## 🏃 Execution Guide

### 1. Local Validation (Dry Run)
Before running training or compiling the pipeline, check your local setup and inputs using the validation utility:

```cmd
# Set workspace path (required for local path resolving)
set WORKSPACE_PATH=E:\FineTuning\services\service-medical-llm\training

# Run the validation checker
python kubeflow/validate_pipeline.py
```

### 2. Test Fine-Tuning (SFT) Component
To run a fast test of the SFT training loop locally through Papermill:

```cmd
cd kubeflow
python test_sft.py
```
*Outputs will be saved in `outputs/test_sft/`.*

### 3. Compiling the Pipeline for Kubeflow
If you modify `kubeflow/pipeline.py`, compile the DAG to YAML to upload it to the Kubeflow cluster:

```cmd
python kubeflow/pipeline.py --compile --output training_pipeline.yaml
```

### 4. Rebuilding the Trainer Docker Image
When updating training code or adding requirements, rebuild the container:

```cmd
docker build -t arabic-medical-llm/trainer:latest .
```

---

## 📊 Metrics & Experiment Tracking
All notebooks are configured to report training losses and evaluations to **MLflow**.
- **Tracking URI:** `http://localhost:5000` (configurable via `MLFLOW_TRACKING_URI`).
- **Best Model Selection:** The pipeline queries the MLflow client to extract the model with the highest `reward_margin` (SimPO or DPO), selects it, and executes the merge.
