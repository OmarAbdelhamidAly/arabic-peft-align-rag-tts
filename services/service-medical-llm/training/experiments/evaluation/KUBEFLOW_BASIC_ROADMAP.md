# Kubeflow Basic Roadmap for This Project

This document is a practical, CV-friendly guide to understand Kubeflow concepts using this Arabic medical LLM project, without requiring company-grade servers.

## Goal

Build basic knowledge of:

- `Kubeflow Pipelines` (workflow orchestration)
- `KServe` (model serving on Kubernetes)
- how they map to our existing project stages

With a laptop (`8 GB RAM`), the target is **learning + lightweight proof of concept**, not full production deployment.

## Why This Is Enough for CV

You can confidently claim:

- end-to-end ML lifecycle understanding
- Kubernetes-native ML architecture basics
- Kubeflow component mapping and deployment reasoning

You should avoid claiming full production Kubeflow deployment unless you actually run it at scale.

## Project-to-Kubeflow Mapping

### Our current stages

1. Data generation and alignment dataset building
2. SFT training
3. Post-training alignment (DPO/IPO/KTO/ORPO/SimPO/RL-style)
4. Evaluation and model comparison
5. Final model serving

### Kubeflow equivalent

1. **Pipeline step:** data preparation component
2. **Pipeline step:** train component (SFT)
3. **Pipeline step:** align component (post-training)
4. **Pipeline step:** evaluate component
5. **Serving step:** KServe inference service (or local vLLM first)

## Core Concepts You Must Be Able to Explain

### 1) Pipeline vs Serving

- **Pipeline**: offline jobs (prepare, train, evaluate).
- **Serving**: online inference API for user requests.

### 2) Why Kubernetes in ML

- reproducibility (same manifests, same runtime style)
- isolation (training/inference environments separated)
- operations pattern used in production teams

### 3) Why KServe

- standardized inference deployment on Kubernetes
- versioned model endpoints
- easier production migration once infrastructure is available

### 4) Why not full Kubeflow on this laptop

- full stack needs extra components (`Istio`, `Knative`, control plane overhead)
- `8 GB RAM` is usually insufficient for a comfortable full local setup

## Learning Path (Designed for 8 GB RAM)

### Phase A: Concept + Design (must do)

Deliverables:

- architecture diagram (pipeline + serving flow)
- this mapping document
- interview notes (questions and concise answers)

### Phase B: Lightweight technical proof

Deliverables:

- local inference endpoint (vLLM or lightweight server)
- basic Kubernetes deploy (`Deployment` + `Service`) for the endpoint
- short benchmark note (latency + memory observations)

### Phase C: Optional KServe PoC

Deliverables:

- tiny KServe trial or design-only section if local resources are not enough
- decision record: why local limit exists and when to migrate to cloud

## Suggested Diagram (You can copy into README/docs)

```text
Data Generator
   -> SFT Training
   -> Post-Training Alignment
   -> Evaluation/Model Selection
   -> Final Model Artifact
   -> Local Serving (vLLM)
   -> Kubernetes Deployment
   -> (Optional) KServe Endpoint
```

## Practical Scope Definition

For this repository, "Kubeflow basic knowledge achieved" means:

- can map each notebook stage to a pipeline component
- can explain data artifact flow between stages
- can explain how final model would be served via KServe
- can articulate local hardware limits and production migration plan

## Interview Q&A (Short Form)

### Q: What does Kubeflow add to your project?

A: It structures the ML lifecycle into reproducible pipeline steps and prepares the system for Kubernetes-native serving.

### Q: Did you deploy full Kubeflow locally?

A: I implemented a lightweight local path due to `8 GB` limits, mapped all project stages to Kubeflow components, and prepared the deployment design for KServe/cloud migration.

### Q: Why use KServe instead of only FastAPI?

A: KServe gives a standard serving pattern on Kubernetes, better lifecycle handling, and cleaner transition to production model operations.

## CV Bullet Examples

- Built an end-to-end Arabic medical LLM fine-tuning workflow (data preparation, SFT, post-training alignment, and evaluation).
- Mapped the workflow to Kubeflow concepts (Pipelines for training/evaluation and KServe-style inference deployment).
- Implemented lightweight local serving and Kubernetes-ready deployment patterns under constrained hardware resources.

## Honest Scope Statement

Use this statement in presentations or portfolio notes:

> This implementation focuses on Kubeflow fundamentals and deployment design patterns in a local constrained environment. Full-scale Kubeflow/KServe production deployment is planned for cloud infrastructure.

