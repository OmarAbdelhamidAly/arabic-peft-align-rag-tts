# 🤖 service-inference — LLM Inference API

> **Role:** Online service — always running, serves the fine-tuned Arabic Medical LLM via REST API.
> Provides an **OpenAI-compatible** endpoint so any client (service-rag, curl, Python) can use it.

---

## 🏗️ Clean Architecture

```
service-inference/
├── app/
│   ├── domain/                         ← Pure contracts, no ML imports
│   │   ├── entities/
│   │   │   ├── inference_request.py    ← ChatCompletionRequest (OpenAI schema)
│   │   │   └── inference_response.py   ← ChatCompletionResponse (OpenAI schema)
│   │   └── interfaces/
│   │       └── model_loader.py         ← ModelLoader ABC
│   │
│   ├── application/                    ← Business logic
│   │   └── use_cases/
│   │       └── generate_completion.py  ← GenerateCompletionUseCase
│   │
│   ├── infrastructure/                 ← ML backend implementations
│   │   └── models/
│   │       └── hf_loader.py           ← HuggingFaceLoader (local, no Docker)
│   │
│   └── interfaces/                     ← FastAPI entry point
│       └── api/
│           └── main.py                ← FastAPI app with all routes
│
├── k8s/                               ← Kubernetes manifests (KServe/vLLM)
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── Dockerfile                         ← vLLM container for production
└── requirements.txt
```

---

## 🔄 Request Flow

```
Client (curl / service-rag / Python)
          │
          │  POST /v1/chat/completions
          │  {"messages": [{"role":"user","content":"ما هي أعراض السكري؟"}]}
          ▼
   FastAPI (interfaces/api/main.py)
          │  validates request → ChatCompletionRequest
          ▼
   GenerateCompletionUseCase (application)
          │  formats messages → calls model_loader.generate()
          ▼
   HuggingFaceLoader (infrastructure)
          │  apply_chat_template → tokenize → generate → decode
          ▼
   ChatCompletionResponse
          │  wraps text in OpenAI-compatible JSON envelope
          ▼
   {"choices": [{"message": {"role":"assistant","content":"..."}}]}
```

---

## ⚙️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Web framework | FastAPI + uvicorn | Async, fast, auto-generates OpenAPI docs |
| Model loading (local) | HuggingFace `transformers` | Works without Docker, easy to debug |
| Quantization | `bitsandbytes` 4-bit | ~4× less VRAM — fits on consumer GPU |
| Model (production) | vLLM | PagedAttention — 10–24× higher throughput |
| K8s serving | KServe InferenceService | Production-grade model serving on K8s |
| Schema | Pydantic v2 | Auto-validation + OpenAPI generation |

### Why OpenAI-compatible API?

The `/v1/chat/completions` format means:
- `service-rag` calls this service the same way it would call the OpenAI API
- Any OpenAI client library works out of the box
- Switching between local model and hosted API requires only changing the URL

### Two backends — same interface

```
ModelLoader (ABC)
    ├── HuggingFaceLoader   ← Local: transformers + bitsandbytes
    │                          Use for: development, testing, no GPU server
    └── [vLLM via Docker]   ← Production: vLLM OpenAI server
                               Use for: K8s deployment, high throughput
```

---

## 🚀 Usage

### Local (no Docker):
```bash
cd services/service-inference
pip install -r requirements.txt

# Set model path to the merged model
MODEL_PATH=../../model_training/merged_model_16bit \
uvicorn app.interfaces.api.main:app --host 0.0.0.0 --port 8001 --reload
```

### Test it:
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "أنت مساعد طبي عربي متخصص."},
      {"role": "user",   "content": "ما هي أعراض مرض السكري؟"}
    ],
    "max_tokens": 256
  }'
```

### Docker (vLLM):
```bash
docker-compose up service-inference
```

### Kubernetes:
```bash
kubectl apply -f k8s/
```

---

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat inference |
| `GET`  | `/v1/models` | List loaded models |
| `POST` | `/admin/load` | Load/reload model at runtime |
| `GET`  | `/health` | Liveness probe |

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `""` | Path to merged model or HuggingFace model ID |
| `PORT` | `8001` | Server port |

---

## 📚 Key Concepts

**Chat Template:** Each model has its own special format for multi-turn conversations.
Qwen2.5 uses:
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user\nQuestion here<|im_end|>
<|im_start|>assistant\n
```
The `tokenizer.apply_chat_template()` handles this automatically.

**4-bit Inference:** The model was trained in 4-bit (QLoRA) and can be served in 4-bit too.
This means a 3B model needs only ~2–3 GB VRAM vs ~6 GB in fp16.
