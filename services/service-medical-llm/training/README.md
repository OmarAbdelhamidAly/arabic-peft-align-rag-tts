<div align="center">

# 🧠 Service Medical LLM: Training Pipeline

### SFT · Parallel Alignment · Papermill · MLflow · Unsloth

<br/>

> **This module handles the end-to-end Offline Training pipeline for the Arabic Mental Health LLM.**  
> It automates the transition from raw datasets to a fully fine-tuned, aligned, and merged 16-bit model ready for production serving via vLLM.

</div>

---

## 🏗️ Architecture Overview

The training pipeline is fully orchestrated using **Kubeflow Pipelines (KFP)**. Instead of manually running Python scripts, the pipeline uses **Papermill** to execute Jupyter Notebooks directly inside isolated Docker containers. This ensures exact reproducibility while maintaining the interactive development experience of Notebooks.

### The Pipeline Flow (DAG)

```text
                  [ 1. Validate Dataset ]
                             │
                             ▼
               [ 2. Supervised Fine-Tuning (SFT) ]
                             │
      ┌────────┬────────┬────┴───┬────────┬────────┐
      ▼        ▼        ▼        ▼        ▼        ▼
   [DPO]    [IPO]    [KTO]    [ORPO]   [SimPO]   [RLOO]
      │        │        │        │        │        │
      └────────┴────────┴────┬───┴────────┴────────┘
                             ▼
                 [ 3. Select Best Model ]
                       (via MLflow)
                             │
                             ▼
                [ 4. Merge to 16-bit Base ]
                             │
                             ▼
                 [ 5. Push to HuggingFace ]
```

---

## 📂 Directory Structure

```text
training/
├── kubeflow/
│   └── pipeline.py                 # The main KFP DAG definition
├── experiments/
│   ├── 01_sft/                     # Supervised Fine-Tuning Notebooks
│   ├── 02_post_training/           # 6 Parallel Alignment Notebooks
│   ├── data_pipeline/              # Data generation (OpenRouter/GPT-4o)
│   ├── evaluation/                 # Benchmark metrics and LLM-as-judge
│   ├── select_and_merge.py         # Standalone merging script
│   └── merged_model_16bit/         # Output directory for the final model
├── data/                           # Curated preference datasets (JSON)
├── Dockerfile                      # Builds the `arabic-medical-llm/trainer` image
└── training_pipeline.yaml          # The compiled KFP pipeline ready for deployment
```

---

## 🔬 Core Components

### 1. Data Generation (`data_pipeline/`)
We use powerful teacher models (like GPT-4o via OpenRouter) to generate highly specialized Arabic Mental Health dialogues. The pipeline generates both:
- **SFT Data:** Standard User/Assistant instructional dialogues.
- **Preference Data:** Chosen/Rejected pairs for alignment algorithms.

### 2. Supervised Fine-Tuning (SFT)
- **Framework:** `Unsloth` (for 2x faster, memory-efficient LoRA training).
- **Base Model:** `Qwen2.5-3B-Instruct`.
- **Output:** An SFT LoRA Adapter (`qwen_mental_health_arabic_lora`).

### 3. Parallel Post-Training Alignment
To ensure the highest quality model, we don't guess which alignment algorithm works best. We run **6 state-of-the-art algorithms simultaneously** on top of the SFT adapter:
1. **DPO:** Direct Preference Optimization.
2. **IPO:** Identity Preference Optimization.
3. **KTO:** Kahneman-Tversky Optimization (Unpaired feedback).
4. **ORPO:** Odds Ratio Preference Optimization.
5. **SimPO:** Simple Preference Optimization (Reference-free).
6. **RLOO:** Reinforcement Learning with Leave-One-Out.

### 4. MLflow Tracking & Selection
Every alignment run logs its Hyperparameters, Training Loss, and **Reward Margins** directly to a centralized MLflow server. The `select_best_model` component queries MLflow, compares the reward margins, and automatically selects the winning adapter (e.g., SimPO).

### 5. Merging and Deployment
The winning adapter is merged into the base model at `16-bit` precision (dequantized). Finally, the `push_to_huggingface` component securely uploads the production-ready weights to the HuggingFace Hub, making it immediately available for the Online Inference Service.

---

## 🛠️ How to Run

### Local Compilation
If you modify `pipeline.py`, you must recompile it into YAML:
```bash
conda activate unsloth_env
python kubeflow/pipeline.py --compile --output ../training_pipeline.yaml
```

### Running on Kubeflow
Once compiled, you can submit the pipeline to your Kubernetes cluster:
```bash
kfp run create \
  --experiment-name arabic-mental-health-llm \
  --pipeline-package-path ../training_pipeline.yaml
```

### Updating the Docker Image
If you add new pip packages to the Notebooks, update the `Dockerfile` and rebuild the runner image:
```bash
docker build -t arabic-medical-llm/trainer:latest .
```

---

## 🗒️ Why Papermill?
By using `papermill`, this architecture eliminates the infamous "Notebook-to-Production gap". You can write and test your experiments interactively in Jupyter, and Kubeflow will execute those exact same Notebooks in production by dynamically injecting parameters (like `learning_rate` or `dataset_path`) into the top cell.
