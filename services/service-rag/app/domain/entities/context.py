"""
Domain Entity: Context
-----------------------
TODO: Aggregates retrieved documents into a prompt context block.

Research questions:
  - What is the maximum context length for Qwen2.5-3B? (128k tokens)
  - How many retrieved documents should we include? (typically 3–5)
  - How should we rank/filter documents before including them?
  - Should we include document scores in the prompt?
"""

from dataclasses import dataclass, field
from typing import List
from .document import Document


@dataclass
class Context:
    """Aggregated retrieval context for the LLM prompt."""
    documents: List[Document] = field(default_factory=list)
    max_chars: int = 4000

    def build_prompt_block(self) -> str:
        """TODO: Format all documents into a single context string."""
        raise NotImplementedError
