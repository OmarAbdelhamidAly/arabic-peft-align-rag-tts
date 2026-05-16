"""
Domain Interface: Embedder
---------------------------
TODO: Abstract contract for text embedding models.

Research questions:
  - Why multilingual-e5-large?
    → 1024-dim embeddings, trained on 100 languages including Arabic
    → Strong cross-lingual retrieval (query in Arabic → retrieve Arabic docs)
  - How to call it? (sentence-transformers library)
  - Should we embed queries and documents differently?
    → YES: e5 models use prefixes:
      query:   "query: <text>"
      passage: "passage: <text>"
  - Batch size for embedding large document corpora? (~32–64)
"""

from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """Abstract contract for text embedding."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed a single text string → vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in one batch call."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...
