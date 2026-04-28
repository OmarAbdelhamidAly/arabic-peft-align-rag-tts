"""
service-rag: FastAPI entry point.
Retrieves relevant Arabic medical docs and passes to LLM → TTS.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Arabic Medical RAG Service",
    description="Retrieval-Augmented Generation over Arabic medical corpus",
    version="0.1.0",
)


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    output_speech: bool = True  # if True, pipes answer to TTS service


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    audio_url: str | None = None  # populated when output_speech=True


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=RAGResponse)
async def query(request: RAGRequest) -> RAGResponse:
    # TODO: wire retriever → LLM → TTS use cases
    raise NotImplementedError("Inject RAG pipeline via use case.")
