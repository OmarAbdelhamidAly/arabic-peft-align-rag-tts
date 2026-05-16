from typing import List

from qdrant_client import QdrantClient

from ...domain.entities.query import Query
from ...domain.entities.document import Document
from ...domain.interfaces.retriever import Retriever


class QdrantRetriever(Retriever):
    """Qdrant vector store adapter implementing Retriever interface."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "arabic_medical",
        embedding_provider=None
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
    
    async def retrieve(self, query: Query, top_k: int = 5) -> List[Document]:
        # Get query embedding
        query_vector = await self.embedding_provider.embed(query.text)
        
        # Search Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Map to Document entities
        documents = []
        for point in results:
            documents.append(Document(
                id=str(point.id),
                content=point.payload.get("content", ""),
                source=point.payload.get("source", "unknown"),
                score=point.score,
                metadata=point.payload
            ))
        
        return documents
    
    async def add_documents(self, documents: List[Document]) -> None:
        # Implementation for adding documents
        pass
