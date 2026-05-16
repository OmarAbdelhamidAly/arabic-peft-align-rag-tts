"""
FastAPI Entry Point: service-rag
----------------------------------
TODO: Implement the RAG orchestration API.

Endpoints to build:
  POST /rag/query   ← main endpoint: takes Arabic question → returns answer + sources
  POST /rag/ingest  ← admin: upload PDF/text documents → embed → store in Qdrant
  GET  /health      ← liveness probe

Flow for /rag/query:
  1. Receive Arabic question
  2. Embed question using E5EmbeddingAdapter
  3. Search Qdrant for top-k similar document chunks
  4. Build Arabic prompt with retrieved context
  5. Call service-inference via InferenceServiceClient
  6. Return answer + source documents

Flow for /rag/ingest:
  1. Receive PDF/text file
  2. Parse text (PyMuPDF for PDFs)
  3. Split into chunks (RecursiveCharacterTextSplitter)
  4. Embed all chunks (E5EmbeddingAdapter batch)
  5. Upsert into Qdrant collection

Research questions:
  - How to handle Arabic PDF text extraction? (PyMuPDF handles RTL)
  - What chunk size? (try 500–1000 chars with 100–200 overlap)
  - How to create the Qdrant collection on first run?
  - Should ingest be sync or async? (async is better for large files)

Useful libraries: fastapi, qdrant-client, pymupdf, langchain-text-splitters, httpx
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title       = "service-rag",
    description = "Arabic Medical RAG — Retrieval-Augmented Generation API",
    version     = "0.1.0",
)


class QueryRequest(BaseModel):
    """TODO: Define the full request schema."""
    question: str
    top_k:    int = 5


class QueryResponse(BaseModel):
    """TODO: Define the full response schema."""
    answer:  str
    sources: List[str]


@app.post("/rag/query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    TODO: Implement the full RAG pipeline:
      embed → retrieve → build prompt → generate → return
    """
    raise NotImplementedError("Implement RAG pipeline")


@app.post("/rag/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    TODO: Implement document ingestion:
      parse PDF → chunk → embed → upsert to Qdrant
    """
    raise NotImplementedError("Implement document ingestion")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag"}
