"""
Domain Entity: InferenceRequest
---------------------------------
OpenAI-compatible chat completion request schema.

Keeping this in the domain layer means the API layer (FastAPI)
and the infrastructure layer (model loader) both reference the
same canonical shape — no duplication.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    """A single chat turn."""
    role:    str   # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible /v1/chat/completions request body.

    Keeping it compatible means the RAG service (or any client)
    can talk to this service the same way it would talk to OpenAI.
    """
    model:       str            = "arabic-medical-llm"
    messages:    List[Message]
    max_tokens:  int            = Field(default=512,  ge=1, le=4096)
    temperature: float          = Field(default=0.7,  ge=0.0, le=2.0)
    top_p:       float          = Field(default=0.95, ge=0.0, le=1.0)
    stream:      bool           = False
    stop:        Optional[List[str]] = None
