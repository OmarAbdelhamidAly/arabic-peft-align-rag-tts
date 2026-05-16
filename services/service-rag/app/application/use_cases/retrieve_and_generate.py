from dataclasses import dataclass

from ...domain.entities.query import Query
from ...domain.interfaces.retriever import Retriever
from ...domain.interfaces.llm_client import LLMClient


@dataclass
class RAGResponse:
    answer: str
    sources: list
    context_used: str


class RetrieveAndGenerateUseCase:
    """Orchestrates RAG pipeline: retrieve -> build prompt -> generate."""
    
    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        system_prompt: str = None
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """أنت مساعد طبي عربي متخصص. استخدم المعلومات المقدمة للإجابة.
إذا لم تكن المعلومات كافية، قدّم نصائح عامة وأوصِ باستشارة طبيب."""
    
    async def execute(self, query: Query) -> RAGResponse:
        # 1. Validate query
        if not query.validate():
            raise ValueError("Invalid query")
        
        # 2. Retrieve documents
        documents = await self.retriever.retrieve(query, top_k=5)
        
        # 3. Build context
        context = self._build_context(documents)
        
        # 4. Build prompt
        prompt = self._build_prompt(query.text, context)
        
        # 5. Generate answer
        answer = await self.llm_client.generate(prompt, max_tokens=512)
        
        return RAGResponse(
            answer=answer,
            sources=[doc.source for doc in documents],
            context_used=context
        )
    
    def _build_context(self, documents) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.content}")
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""السياق:
{context}

السؤال: {question}

الإجابة:"""
