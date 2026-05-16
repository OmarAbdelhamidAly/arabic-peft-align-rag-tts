# 🚀 Phase 1: The R&D Foundation (Research, Development & Serving)

> **Goal:** Prove the concept. Build the best possible Arabic Mental Health LLM manually, evaluate it rigorously, and serve it efficiently.

## 📖 Overview
Phase 1 represents the "Data Scientist / AI Researcher" phase. Before automating anything, we needed to ensure that we could actually build a model that understands complex Arabic psychological terminology and aligns with human preferences. This phase is highly experimental, heavily relying on Jupyter Notebooks, manual evaluation, and finding the perfect mathematical formula for our model.

## 🔬 Boring Details & Implementation

### 1. Data Engineering & Generation
We didn't just scrape the internet. We built a synthetic data generation pipeline using powerful teacher models (like GPT-4o via OpenRouter) to generate highly specialized Arabic Mental Health dialogues. 
- **SFT Data:** Standard User/Assistant instructional dialogues (e.g., a user asking about panic attacks, the assistant answering with empathy and scientific accuracy).
- **Preference Data:** For alignment, we generated pairs of (Chosen / Rejected) answers. The rejected answers contained hallucinations, lack of empathy, or unsafe medical advice.

### 2. Supervised Fine-Tuning (SFT)
- **Base Model:** `Qwen2.5-3B-Instruct` (chosen for its strong multilingual capabilities and efficiency).
- **Framework:** `Unsloth` was used to accelerate training by 2x and reduce VRAM usage using 4-bit quantization and LoRA (Low-Rank Adaptation).
- **Output:** An SFT adapter that speaks Arabic fluently and understands the mental health domain.

### 3. The 6-Way Alignment Battle
To prevent the model from giving harmful or unhelpful advice, we had to "align" it. Instead of guessing the best algorithm, we ran 6 state-of-the-art algorithms manually:
1. **DPO (Direct Preference Optimization):** Standard preference learning without a separate reward model.
2. **IPO (Identity Preference Optimization):** Prevents the model from over-optimizing and becoming robotic.
3. **KTO (Kahneman-Tversky Optimization):** Learns from unpaired human feedback (e.g., a simple "thumbs up" or "thumbs down").
4. **ORPO (Odds Ratio Preference Optimization):** Skips the SFT step and does SFT + Alignment simultaneously.
5. **SimPO (Simple Preference Optimization):** Uses a reference-free sequence-level reward.
6. **RLOO (Reinforcement Learning with Leave-One-Out):** A low-variance online RL algorithm.

### 4. Evaluation & Merging
We used an **LLM-as-a-Judge** script (`01_compare_5_models.ipynb`) to evaluate the outputs of all models against a hidden test set. The model that scored the highest in Empathy, Accuracy, and Safety (which happened to be `SimPO`) was selected.
We then dequantized and merged this LoRA adapter into the base model to create a standalone **16-bit** model, which was uploaded to the Hugging Face Hub.

### 5. Serving (Inference)
The merged model is deployed online using **vLLM** (for PagedAttention and continuous batching, ensuring maximum throughput) and **KServe** on Kubernetes to expose a scalable, OpenAI-compatible REST API (`/v1/chat/completions`).

---

## ⚡ How it Differs from the Other Phases
**Phase 1 is manual and experimental.** It is about *discovery*. If a notebook fails, a human fixes it. If a hyperparameter needs tuning, a human changes it. It is the necessary chaotic laboratory work required before building a factory. In this phase, the primary metric is **Model Quality**, not speed or automation.
