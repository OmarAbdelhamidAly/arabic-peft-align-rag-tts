# 🏭 Phase 2: MLOps Automation (The Factory)

> **Goal:** Eliminate manual intervention. Transform the chaotic Jupyter-notebook R&D process into a robust, scalable, and reproducible automated pipeline using Kubernetes, Kubeflow, and MLflow.

## 📖 Overview
Phase 2 represents the "ML Engineer / MLOps" phase. While Phase 1 proved that we *can* build a great model, Phase 2 ensures that we can build it *consistently, repeatedly, and efficiently*. We took the standalone scripts and notebooks and orchestrated them into a Directed Acyclic Graph (DAG) that runs natively in a Kubernetes cluster.

## 🔬 Boring Details & Implementation

### 1. Kubeflow Pipelines (KFP) Orchestration
Instead of running a script and waiting 5 hours to run the next one, we defined a **Kubeflow Pipeline** (`pipeline.py`). This pipeline defines the exact sequence of events:
- Validate Dataset → Run SFT → Run Parallel Alignments → Select Best Model → Merge → Push to HF.
This pipeline is compiled into a YAML file, which Kubernetes reads to provision the exact hardware (GPUs/CPUs) needed for each step, tearing them down when finished to save cloud costs.

### 2. The "Papermill" Bridge
Usually, moving from Phase 1 to Phase 2 requires rewriting all Jupyter Notebooks (`.ipynb`) into standard Python modules (`.py`). We bypassed this massive engineering overhead by using **Papermill**.
Papermill allows Kubeflow to execute the original Jupyter Notebooks natively. It injects pipeline parameters (like `dataset_path`, `learning_rate`) directly into a new cell at the top of the notebook before execution. This ensures that the exact code the Data Scientist wrote in Phase 1 is what runs in Production in Phase 2.

### 3. Parallel Execution
In Phase 1, the 6 alignment algorithms were run sequentially (one after another). In Phase 2, Kubeflow leverages Kubernetes to spin up 6 isolated Docker containers simultaneously. 
DPO, IPO, KTO, ORPO, SimPO, and RLOO all train at the exact same time, reducing a process that used to take 24 hours down to just 4 hours.

### 4. MLflow Tracking
Running 6 parallel jobs makes tracking metrics impossible without a centralized system. We integrated **MLflow** directly into the Kubeflow components. 
Every time a notebook runs, it logs:
- Hyperparameters.
- Training Loss curves.
- **Reward Margins** (the metric that determines how well the model aligns with human preferences).

### 5. Automated Selection & Deployment
Once all 6 alignment jobs finish, the `select_best_model` component queries the MLflow server via its API, retrieves the reward margins, and mathematically selects the absolute best adapter. 
The pipeline then automatically passes this adapter to the `merge_model` component, and finally to `push_to_huggingface` which uses a secure token to update the live repository. The KServe inference service then automatically pulls this new weight.

---

## 💻 Hardware Constraints & Engineering Philosophy
It is critical to note that this entire pipeline was developed and executed on a single **NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB VRAM)**. 
Because of this strict hardware limitation, large-scale distributed training on Kubernetes across multiple GPUs was impossible. However, this constraint became the primary driver for our engineering excellence. 

**The Philosophy:** If we can build an end-to-end, highly capable, aligned Arabic Mental Health LLM using a small 3B parameter model and just 8GB of VRAM, imagine the quality and scale achievable with enterprise-grade infrastructure and 70B+ models. The goal of this project isn't to train the world's most powerful AI, but to demonstrate that we can build the world's most robust and automated **AI Infrastructure and Training Pipeline**.

---

## ⚡ How it Differs from the Other Phases
**Phase 2 is highly structured and automated.** It is about *Infrastructure and Reproducibility*. If the pipeline is triggered, no human needs to touch a keyboard until the new model is live in production. It differs from Phase 1 because it sacrifices ad-hoc flexibility for absolute stability. You don't "tweak" things mid-run here; you rely on the system to do exactly what it was programmed to do.
