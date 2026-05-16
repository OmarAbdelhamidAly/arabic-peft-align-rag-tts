"""
Domain Interface: ModelLoader
-------------------------------
Abstract contract for loading and querying an LLM.

Why here (domain layer)?
  The GenerateCompletionUseCase depends on this interface,
  not on any specific backend (HuggingFace, vLLM, OpenAI API).
  This enables:
    - Local testing with HuggingFace transformers
    - Production serving with vLLM (K8s/KServe)
    - Mocking in unit tests
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class ModelLoader(ABC):

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def generate(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.95,
    ) -> str:
        """
        Generate a response for a list of chat messages.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p:       Nucleus sampling probability.

        Returns:
            Generated text string.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the model is ready to serve."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Returns the loaded model identifier."""
        ...
