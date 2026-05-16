# 🏥 Service Medical LLM

This service handles the core LLM pipeline for the Mental Health project. It consists of two components:

1. **Training (Offline):** Supervised Fine-Tuning (SFT) and parallel preference alignments (DPO, IPO, KTO, ORPO, SimPO, RLOO) orchestrated via Kubeflow Pipelines and Papermill, tracked via MLflow.
2. **Inference (Online):** An OpenAI-compatible REST API served via vLLM and KServe for high-throughput, low-latency text generation.

See the `training/` and `inference/` directories for detailed implementations.
