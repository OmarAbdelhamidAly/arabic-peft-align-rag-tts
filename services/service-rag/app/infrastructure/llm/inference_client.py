"""
Infrastructure: InferenceServiceClient
-----------------------------------------
TODO: HTTP client that calls service-inference to generate LLM responses.

How it works:
  service-rag → (HTTP POST) → service-inference:8001/v1/chat/completions
                             → returns OpenAI-compatible JSON response

Why not import the model directly here?
  service-rag and service-inference are separate microservices.
  Keeping them decoupled means you can scale them independently,
  upgrade the LLM without touching the RAG logic, etc.

Steps to implement:
  1. Use httpx (async HTTP client) — NOT requests (blocking)
  2. POST to http://service-inference:8001/v1/chat/completions
  3. Parse the response JSON → extract choices[0].message.content
  4. Handle timeouts (~60s) and connection errors gracefully
  5. Add retry logic (tenacity library is great for this)

Environment variable:
  INFERENCE_URL=http://service-inference:8001  (set in docker-compose)

Useful libraries: httpx, tenacity
"""

import os
from ...domain.interfaces.llm_client import LLMClient


class InferenceServiceClient(LLMClient):
    """HTTP client to service-inference for LLM generation."""

    def __init__(self, base_url: str = ""):
        self.base_url = base_url or os.getenv("INFERENCE_URL", "http://localhost:8001")

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        TODO: POST to /v1/chat/completions and return the generated text.

        Research httpx.AsyncClient for async HTTP calls.
        """
        raise NotImplementedError("Implement using httpx.AsyncClient")
