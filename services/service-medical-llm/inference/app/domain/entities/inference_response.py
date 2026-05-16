"""
Domain Entity: InferenceResponse
----------------------------------
OpenAI-compatible chat completion response schema.
"""

from pydantic import BaseModel
from typing import List, Optional
import time
import uuid


class MessageContent(BaseModel):
    role:    str
    content: str


class Choice(BaseModel):
    index:         int
    message:       MessageContent
    finish_reason: str   # "stop" | "length" | "error"


class UsageStats(BaseModel):
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response envelope."""
    id:      str            = ""
    object:  str            = "chat.completion"
    created: int            = 0
    model:   str            = "arabic-medical-llm"
    choices: List[Choice]   = []
    usage:   Optional[UsageStats] = None

    @classmethod
    def from_text(cls, text: str, model: str, prompt_tokens: int = 0) -> "ChatCompletionResponse":
        """Factory: wrap a plain text response into the OpenAI envelope."""
        completion_tokens = len(text.split())
        return cls(
            id      = f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created = int(time.time()),
            model   = model,
            choices = [Choice(
                index         = 0,
                message       = MessageContent(role="assistant", content=text),
                finish_reason = "stop",
            )],
            usage = UsageStats(
                prompt_tokens     = prompt_tokens,
                completion_tokens = completion_tokens,
                total_tokens      = prompt_tokens + completion_tokens,
            ),
        )
