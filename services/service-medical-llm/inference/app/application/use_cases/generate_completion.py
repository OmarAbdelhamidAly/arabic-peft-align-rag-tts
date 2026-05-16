"""
Use Case: GenerateCompletionUseCase
--------------------------------------
Orchestrates a single LLM inference call.

Responsibilities:
  1. Validate the request
  2. Format messages into a prompt (apply chat template)
  3. Delegate generation to the ModelLoader (infrastructure)
  4. Wrap the raw text into an OpenAI-compatible response

This class has zero awareness of which ML framework is being used.
"""

import logging
from ...domain.entities.inference_request import ChatCompletionRequest
from ...domain.entities.inference_response import ChatCompletionResponse
from ...domain.interfaces.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class GenerateCompletionUseCase:
    """
    Single use case: take a ChatCompletionRequest → return ChatCompletionResponse.

    Depends on ModelLoader (abstract), injected at startup time.
    """

    def __init__(self, model_loader: ModelLoader):
        self.model = model_loader

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Execute one inference call.

        Args:
            request: Validated ChatCompletionRequest.

        Returns:
            OpenAI-compatible ChatCompletionResponse.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError:   If the request has no messages.
        """
        if not self.model.is_loaded():
            raise RuntimeError("Model is not loaded. Call /admin/load first.")

        if not request.messages:
            raise ValueError("Request must have at least one message.")

        logger.info(f"[INFERENCE] {len(request.messages)} messages | max_tokens={request.max_tokens}")

        # Convert Pydantic messages → plain dicts for the model loader
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Count prompt tokens (rough estimate)
        prompt_tokens = sum(len(m["content"].split()) for m in messages)

        # Delegate to infrastructure
        generated_text = self.model.generate(
            messages    = messages,
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
            top_p       = request.top_p,
        )

        return ChatCompletionResponse.from_text(
            text          = generated_text,
            model         = self.model.model_name(),
            prompt_tokens = prompt_tokens,
        )
