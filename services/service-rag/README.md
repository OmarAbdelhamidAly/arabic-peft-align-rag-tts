# 🔍 service-rag — Retrieval-Augmented Generation API

> **Role:** Online orchestrator — the brain of the system.
> Takes an Arabic medical question, retrieves relevant documents from Qdrant,
> calls service-inference for an LLM answer, and returns text (+ optionally calls service-tts for audio).

---

## 🏗️ Clean Architecture

```
service-rag/
├── app/
│   ├── domain/                                  ← Pure contracts
│   │   ├── entities/
│   │   │   ├── query.py                         ← Query entity         ← TODO: implement
│   │   │   ├── document.py                      ← Document entity      ← TODO: implement
│   │   │   └── context.py                       ← Context entity       ← TODO: implement
│   │   └── interfaces/
│   │       ├── retriever.py                     ← Retriever ABC        ← TODO: implement
│   │       ├── llm_client.py                    ← LLMClient ABC        ← TODO: implement
│   │       └── embedder.py                      ← Embedder ABC         ← TODO: implement
│   │
│   ├── application/
│   │   └── use_cases/
│   │       └── retrieve_and_generate.py         ← RAG orchestration ✅
│   │
│   ├── infrastructure/
│   │   ├── vectorstore/
│   │   │   └── qdrant_adapter.py                ← QdrantRetriever ✅ (partial)
│   │   ├── embeddings/
│   │   │   └── e5_adapter.py                    ← E5EmbeddingAdapter  ← TODO: implement
│   │   └── llm/
│   │       └── inference_client.py              ← InferenceServiceClient ← TODO: implement
│   │
│   └── interfaces/
│       └── api/
│           └── main.py                          ← FastAPI app          ← TODO: implement
│
├── scripts/
│   └── ingest_documents.py                      ← Bulk document ingestion ← TODO: implement
├── Dockerfile
└── requirements.txt
```

---

## 🔄 Request Flow

```
Arabic Medical Question (from user or frontend)
          │
          │  POST /rag/query {"question": "ما هي أعراض مرض السكري؟"}
          ▼
   FastAPI (interfaces/api/main.py)
          │  validate → create Query entity
          ▼
   RetrieveAndGenerateUseCase (application)
          │
          ├─ [Step 1] Embed query
          │    E5EmbeddingAdapter.embed("query: ما هي أعراض...") → vector[1024]
          │
          ├─ [Step 2] Search Qdrant
          │    QdrantRetriever.retrieve(query, top_k=5) → List[Document]
          │
          ├─ [Step 3] Build Arabic prompt
          │    "السياق:\n[1] ...\n[2] ...\nالسؤال: ما هي أعراض...\nالإجابة:"
          │
          ├─ [Step 4] Generate answer
          │    InferenceServiceClient.generate(prompt) → HTTP POST service-inference
          │                                           → returns Arabic answer text
          │
          └─ [Step 5] Return response
               {"answer": "...", "sources": ["doc1.pdf", ...]}
```

---

## 🗂️ Ingestion Flow (one-time setup)

```
Arabic Medical PDF/TXT documents
          │
          ▼
scripts/ingest_documents.py
          │
          ├─ [Step 1] Parse PDFs  → PyMuPDF (handles Arabic RTL)
          ├─ [Step 2] Split text  → RecursiveCharacterTextSplitter (500 chars, 100 overlap)
          ├─ [Step 3] Embed chunks → E5EmbeddingAdapter.embed_batch() (prefix: "passage: ")
          └─ [Step 4] Upsert      → QdrantRetriever.add_documents() → Qdrant collection
```

---

## ⚙️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Vector DB | Qdrant | Fast cosine search, easy Docker deployment |
| Embedding | `intfloat/multilingual-e5-large` | Best Arabic embedding model (1024-dim) |
| LLM backend | service-inference (HTTP) | Microservice decoupling |
| PDF parsing | PyMuPDF (`fitz`) | Handles Arabic RTL text correctly |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` | Respects sentence boundaries |
| HTTP client | `httpx` (async) | Non-blocking calls to service-inference |
| Web API | FastAPI + uvicorn | Async, OpenAPI auto-docs |

---

## 📚 Research Roadmap (What to Learn Next)

### Step 1 — Understand Qdrant
Key concepts:
- **Collection**: like a database table. One collection = one domain (arabic_medical)
- **Point**: one document chunk = {id, vector, payload}
- **Payload**: metadata stored alongside the vector (source, page, content)
- **Search**: give a query vector → returns nearest points by cosine distance

```python
# Create collection (run once)
client.create_collection(
    collection_name = "arabic_medical",
    vectors_config  = VectorParams(size=1024, distance=Distance.COSINE)
)

# Upsert documents
client.upsert("arabic_medical", points=[
    PointStruct(id=1, vector=[...], payload={"content": "...", "source": "doc.pdf"})
])

# Search
results = client.search("arabic_medical", query_vector=[...], limit=5)
```

### Step 2 — Understand multilingual-e5-large prefixes
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large")

# Query embedding (prefix "query: " is REQUIRED)
q_vec = model.encode("query: ما هي أعراض مرض السكري؟", normalize_embeddings=True)

# Document embedding (prefix "passage: " is REQUIRED)
d_vec = model.encode("passage: السكري هو مرض...", normalize_embeddings=True)
```

### Step 3 — Implement InferenceServiceClient
```python
import httpx

async with httpx.AsyncClient(timeout=60.0) as client:
    resp = await client.post(
        f"{self.base_url}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
    )
    return resp.json()["choices"][0]["message"]["content"]
```

### Step 4 — Arabic RAG prompt engineering
The system prompt and context format matter greatly for Arabic:
```
أنت مساعد طبي عربي متخصص. استخدم المعلومات التالية فقط للإجابة.

السياق:
[1] {doc1_content}
[2] {doc2_content}

السؤال: {question}

الإجابة:
```

---

## 🚀 Usage

### Start dependencies:
```bash
docker-compose up qdrant          # Vector DB
docker-compose up service-inference  # LLM backend
```

### Ingest documents (once):
```bash
python scripts/ingest_documents.py \
  --source ./data/medical_docs \
  --collection arabic_medical
```

### Run the API:
```bash
INFERENCE_URL=http://localhost:8001 \
QDRANT_HOST=localhost \
uvicorn app.interfaces.api.main:app --host 0.0.0.0 --port 8003
```

### Test:
```bash
curl -X POST http://localhost:8003/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي الجرعة الموصى بها من الميتفورمين؟"}'
```

---

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/rag/query` | Arabic Q&A with retrieved context |
| `POST` | `/rag/ingest` | Upload + index a document |
| `GET`  | `/health` | Liveness probe |

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_URL` | `http://localhost:8001` | service-inference URL |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Embedding model |
| `COLLECTION_NAME` | `arabic_medical` | Qdrant collection |
