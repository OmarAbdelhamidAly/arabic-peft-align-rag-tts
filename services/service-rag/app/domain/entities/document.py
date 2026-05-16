"""
Domain Entity: Document
------------------------
TODO: Represents a retrieved Arabic medical document chunk.

Research questions:
  - What chunk size works best for Arabic medical text?
    (LangChain default: 1000 chars with 200 overlap)
  - Should chunks overlap? How much?
  - How do we handle Arabic RTL text in chunk boundaries?
  - What metadata should we store per chunk?
    (source_file, page_number, section, specialty)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Document:
    """A retrieved document chunk from the vector store."""
    id:       str
    content:  str
    source:   str
    score:    float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """TODO: Format document for inclusion in LLM prompt."""
        return f"[{self.source}]\n{self.content}"
