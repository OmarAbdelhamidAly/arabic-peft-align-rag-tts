# Solution Execution Checklist

Use this checklist to complete the full lightweight solution end-to-end.

## Stage 1: Finalize Selected Model

- [ ] Confirm final winner model from `01_compare_5_models.ipynb`
- [ ] Record model ID and tokenizer path
- [ ] Decide local serving mode (vLLM or compatible server)

## Stage 2: Local Serving Baseline

- [ ] Start local inference server on `http://127.0.0.1:8001`
- [ ] Verify health endpoint: `GET /health`
- [ ] Verify OpenAI-compatible endpoint: `POST /v1/chat/completions`
- [ ] Run benchmark:
  - `python model_training/evaluation/local_serving/benchmark_inference.py --model <MODEL_ID>`
- [ ] Save summary metrics (avg, median, p95)

## Stage 3: Kubernetes Lightweight Deploy

- [ ] Create local cluster (`kind` or `minikube`)
- [ ] Apply:
  - `configmap.yaml`
  - `deployment.yaml`
  - `service.yaml`
- [ ] Confirm pod ready
- [ ] Port-forward service to local machine
- [ ] Run smoke eval:
  - `python model_training/evaluation/local_serving/eval_after_deploy.py --model <MODEL_ID>`

## Stage 4: Kubeflow Concept Mapping

- [ ] Review `KUBEFLOW_BASIC_ROADMAP.md`
- [ ] Explain project stages as pipeline components
- [ ] Explain serving stage as KServe-ready architecture
- [ ] Prepare 3-minute explanation for interview

## Stage 5: Portfolio and CV Proof

- [ ] Add architecture snapshot to your notes
- [ ] Add 2-3 CV bullets from `cv/CV_BULLETS.md`
- [ ] Practice pitch from `interview/INTERVIEW_SCRIPT.md`

## Definition of Done

- [ ] You can run local serving
- [ ] You can run local K8s deployment
- [ ] You can explain Kubeflow concepts mapped to this project
- [ ] You have CV/interview material ready and honest
