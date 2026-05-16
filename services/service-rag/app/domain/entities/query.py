"""
Domain Entity: Query
---------------------
TODO: Represents a user's Arabic medical question.

Research questions:
  - Should we normalize Arabic text before embedding?
    (remove tashkeel, normalize hamza, etc.)
  - Should the query carry metadata like session_id or language?
  - Do we need query expansion for better retrieval?
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Query:
    """A user query to the RAG pipeline."""
    text:       str
    language:   str = "ar"
    session_id: Optional[str] = None

    def validate(self) -> bool:
        """TODO: Add Arabic text validation logic."""
        return bool(self.text and self.text.strip())
