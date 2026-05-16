"""
Domain Interface: Retriever
-----------------------------
TODO: Abstract contract for any vector store retrieval backend.

Research questions:
  - What distance metric does Qdrant use? (Cosine by default)
  - What vector dimension does multilingual-e5-large produce? (1024)
  - Should we use dense retrieval only, or add sparse (BM25) too?
    → Hybrid search (dense + sparse) often outperforms either alone
  - How do we handle the Qdrant collection creation on first run?
"""

from abc import ABC, abstractmethod
from typing import List
from ..entities.query import Query
from ..entities.document import Document


class Retriever(ABC):
    """Abstract contract for vector store retrieval."""

    @abstractmethod
    async def retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        """Search the vector store and return top_k documents."""
        ...

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Index documents into the vector store."""
        ...
