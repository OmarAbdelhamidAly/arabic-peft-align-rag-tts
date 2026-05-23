from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ChatMessage:
    """Chat message entity."""
    role: str  # system, user, assistant
    content: str


@dataclass
class ChatRequest:
    """Chat completion request entity."""
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    
    def validate(self) -> bool:
        return bool(self.messages and len(self.messages) > 0)
