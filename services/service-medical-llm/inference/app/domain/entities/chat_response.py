from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ChatChoice:
    """Single chat completion choice."""
    index: int
    message: Dict[str, str]
    finish_reason: str


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    """Chat completion response entity."""
    id: str
    model: str
    choices: List[ChatChoice]
    usage: Usage
    created: int
