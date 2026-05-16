"""
Infrastructure: E5EmbeddingAdapter
-------------------------------------
TODO: Implement text embedding using intfloat/multilingual-e5-large.

Why multilingual-e5-large?
  - 1024-dimensional embeddings
  - Trained on 100+ languages including Arabic
  - Handles cross-lingual retrieval well
  - State-of-the-art on MTEB Arabic benchmarks

How to use it (research this):
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer("intfloat/multilingual-e5-large")

  # IMPORTANT: e5 models require prefixes!
  query_embedding = model.encode("query: ما هي أعراض مرض السكري؟")
  doc_embedding   = model.encode("passage: السكري هو مرض مزمن...")

Steps to implement:
  1. Load the model (once, at startup)
  2. For queries: prepend "query: " before encoding
  3. For documents: prepend "passage: " before encoding
  4. Normalize embeddings (L2) for cosine similarity
  5. Return as List[float] for Qdrant

Useful libraries: sentence-transformers, torch
"""

from ...domain.interfaces.embedder import Embedder
from typing import List


class E5EmbeddingAdapter(Embedder):
    """Concrete embedder using multilingual-e5-large."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self._model_name = model_name
        self._model      = None  # TODO: load SentenceTransformer here

    async def embed(self, text: str) -> List[float]:
        """TODO: Encode a single text with query prefix."""
        raise NotImplementedError("Implement using sentence-transformers")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """TODO: Encode a batch of passage texts."""
        raise NotImplementedError("Implement batch encoding")

    def dimension(self) -> int:
        return 1024  # multilingual-e5-large output dimension
