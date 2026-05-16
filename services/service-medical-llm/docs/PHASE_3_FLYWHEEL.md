# 🔄 Phase 3: Active Learning (The Data Flywheel)

> **Goal:** Create a self-improving AI system. Build an automated feedback loop that identifies the model's weaknesses in production, generates new synthetic data to fix those weaknesses, and retrains the model autonomously.

## 📖 Overview
Phase 3 represents the "AI Systems Architect" phase. Having a great model (Phase 1) that deploys automatically (Phase 2) is excellent, but real-world user queries will inevitably expose edge cases the model hasn't seen before. Phase 3 transforms the architecture into a living, breathing system. We implement a "Data Flywheel"—a continuous loop where usage drives data generation, data generation drives training, and training improves usage.

## 🔬 Boring Details & Implementation

### 1. Production Monitoring & Weakness Detection
When the model is live (served via `service-medical-llm` Inference API or embedded in the `service-rag` pipeline), we log user queries and the model's responses. 
An automated LLM-as-a-judge periodically evaluates these logs to flag low-confidence responses, hallucinations, or unsafe medical advice (e.g., struggling with specific regional dialects or complex psychiatric terminology).

### 2. Targeted Data Generation (The "Hard" Dataset)
Once weaknesses are identified, the system triggers the data generation pipeline (`03_alignment_data_generator_hard.ipynb`). 
Unlike the generic data generated in Phase 1, this script uses GPT-4o (or another strong teacher model) to generate highly specific **Hard Negatives** and complex scenarios directly related to the flagged weaknesses. It creates explicit Chosen/Rejected pairs targeting exactly where the previous model failed.

### 3. Automated Re-Training (Triggering Phase 2)
The newly generated specialized dataset is automatically appended to the main training corpus. 
The system then triggers a Webhook or a Kubernetes CronJob to launch the entire **Phase 2 Kubeflow Pipeline** again. 
The pipeline runs the SFT and parallel alignments (DPO, KTO, SimPO, etc.) on this expanded, much harder dataset.

### 4. Shadow Deployment & A/B Testing
Before replacing the live production model, the new merged model is deployed in "Shadow Mode". Real user traffic is duplicated and sent to both the old model and the new model. If the new model consistently scores better on the evaluation metrics (especially on the edge cases it was specifically trained to fix) without degrading performance on generic queries, it is promoted to production.

---

## 💻 Hardware Constraints & Engineering Philosophy
It is critical to note that this entire pipeline was developed and executed on a single **NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB VRAM)**. 
Because of this strict hardware limitation, large-scale distributed training on Kubernetes across multiple GPUs was impossible. However, this constraint became the primary driver for our engineering excellence. 

**The Philosophy:** If we can build an end-to-end, highly capable, aligned Arabic Mental Health LLM using a small 3B parameter model and just 8GB of VRAM, imagine the quality and scale achievable with enterprise-grade infrastructure and 70B+ models. The goal of this project isn't to train the world's most powerful AI, but to demonstrate that we can build the world's most robust and automated **AI Infrastructure and Training Pipeline**.

---

## ⚡ How it Differs from the Other Phases
**Phase 3 is autonomous and cyclical.** 
- **Phase 1** was about a human painstakingly building a model.
- **Phase 2** was about a human pressing a button to build a model automatically.
- **Phase 3** is about the system deciding *when* and *how* to build the model based on its own failures. 
It differs fundamentally because it focuses entirely on the **Feedback Loop**. The engineering challenge here is not training speed or orchestration, but designing accurate evaluation heuristics and ensuring safe, autonomous data generation that doesn't suffer from model collapse over multiple iterations.
