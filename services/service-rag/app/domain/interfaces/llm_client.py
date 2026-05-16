"""
Domain Interface: LLMClient
-----------------------------
TODO: Abstract contract for calling the LLM (service-inference).

Research questions:
  - How do we call service-inference from service-rag?
    → HTTP client (httpx) to POST /v1/chat/completions
  - Should we use async HTTP? (Yes — FastAPI is async)
  - What timeout is reasonable for LLM generation? (~60 seconds)
  - Should we retry on failure?
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract contract for the LLM generation backend."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a response for the given prompt.
        The concrete implementation calls service-inference HTTP API.
        """
        ...
