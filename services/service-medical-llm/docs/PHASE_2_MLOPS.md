# 🏭 Phase 2: MLOps Automation (The Factory)

> **Goal:** Eliminate manual intervention. Transform the interactive Jupyter-notebook R&D process into a robust, scalable, and reproducible automated pipeline using Kubernetes, Kubeflow, and MLflow.

## 📖 Overview
Phase 2 represents the "ML Engineer / MLOps" phase. While Phase 1 proved we can train an aligned model, Phase 2 ensures we can build it consistently, repeatedly, and efficiently. We automated the workflow by defining a Directed Acyclic Graph (DAG) in Kubeflow, allowing notebooks to run concurrently inside containerized Kubernetes environments.

---

## 🔬 Core Components & Implementation Details

### 1. Kubeflow Pipelines (KFP) Orchestration (`kubeflow/pipeline.py`)
The entire process is compiled into a KFP DAG which defines the following steps:
1.  **Validate Dataset:** Checks for the presence of preference files and basic schema requirements.
2.  **Run SFT:** Executes the supervised fine-tuning notebook via Papermill.
3.  **Run Parallel Alignments:** Spins up 6 concurrent Docker containers running DPO, IPO, KTO, ORPO, SimPO, and RLOO alignments.
4.  **Select Best Model:** Retrieves reward margins from MLflow and chooses the winning adapter.
5.  **Merge Model:** Merges the selected PEFT adapter into the base 16-bit model.
6.  **Push to HuggingFace:** Securely uploads the merged weights to the Hugging Face Hub.

*To compile the pipeline into a Kubernetes-ready YAML format:*
```cmd
python kubeflow/pipeline.py --compile --output training_pipeline.yaml
```

### 2. The "Papermill" Bridge & Environment Setup
Using **Papermill** eliminates the need to rewrite Jupyter notebooks into Python scripts.
*   **Parameters Injection:** Kubeflow injects variables like `DATASET_PATH` and `LEARNING_RATE` directly into the notebook's `"parameters"` tagged cell.
*   **Artifact Retention:** The executed notebooks (e.g., `train_sft_executed.ipynb`) are saved in the outputs directory to inspect logs and plots later.

#### ⚠️ Critical Compatibility Patch (Tokenizers Issue)
Unsloth has strict requirements on the `tokenizers` library, which conflicts with newer transformer versions. We resolved the DLL load/tokenizer mismatch error by pinning the exact version without fetching dependencies:
```cmd
pip install "tokenizers==0.21.4" --no-deps
```
*Failure to use this exact version will raise exceptions during model loading (`Exception: tokenizers>=0.21,<0.22 is required...`).*

### 3. Workspace Path Resolution
Local executions require the `WORKSPACE_PATH` environment variable to locate files:
```cmd
set WORKSPACE_PATH=E:\FineTuning\services\service-medical-llm\training
```
Without this variable, the pipeline assumes a default `/workspace` directory matching the internal Docker container workdir.

### 4. Dry-Run Validation (`kubeflow/validate_pipeline.py`)
To prevent wasting GPU resources and pipeline execution failures, we built a comprehensive dry-run validator. Run the following command before launching the pipeline:
```cmd
python kubeflow/validate_pipeline.py
```
This script checks:
*   **Python packages:** Installs and versions (Torch, Unsloth, TRL, etc.).
*   **Dataset Integrity:** Verifies if `chatml_dataset.json` exists, is valid JSON, and uses the correct ChatML schema.
*   **Notebook Tags:** Verifies if the SFT notebook contains the `"parameters"` tag cell.
*   **Mock Injection:** Executes `papermill --prepare-only` to ensure parameters can be injected.
*   **MLflow connection:** Verifies tracking server reachability.

---

## ⚡ How Phase 2 Differs from Phase 1
Phase 2 prioritizes **Infrastructure, Scale, and Reproducibility**. Instead of tweaking parameters interactively, the pipeline runs start-to-finish without human interaction. If a step fails, the pipeline raises an alert, keeping the production system robust and reliable.
