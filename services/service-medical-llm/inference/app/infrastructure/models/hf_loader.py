"""
Infrastructure: HuggingFaceLoader
-----------------------------------
Concrete ModelLoader using HuggingFace transformers directly.

When to use this:
  - Local development / testing (no Docker, no vLLM needed)
  - CPU inference (slow but functional for demos)
  - GPU inference on a single machine

For production (K8s/KServe):
  → Use the vLLM Dockerfile instead. vLLM serves an OpenAI-compatible
    API on port 8001 natively, so service-inference simply wraps it.

Architecture note:
  This class lives in the infrastructure layer because it imports
  torch and transformers — external ML dependencies.
"""

import logging
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ...domain.interfaces.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class HuggingFaceLoader(ModelLoader):
    """
    Loads and serves the fine-tuned Qwen model using HuggingFace transformers.

    Supports both:
      - 4-bit (bitsandbytes): smaller VRAM footprint
      - 16-bit (bfloat16):   faster on modern GPUs
    """

    def __init__(self, load_in_4bit: bool = True):
        self._model      = None
        self._tokenizer  = None
        self._pipeline   = None
        self._model_path = ""
        self._load_4bit  = load_in_4bit

    # ── ModelLoader interface ─────────────────────────────────────────────────

    def load(self, model_path: str) -> None:
        """
        Load the merged 16-bit model (or any HuggingFace model).

        Args:
            model_path: Local path or HuggingFace model ID.
                        Default: reads MODEL_PATH env var.
        """
        self._model_path = model_path or os.getenv("MODEL_PATH", "")
        if not self._model_path:
            raise ValueError("MODEL_PATH env variable or model_path argument is required.")

        logger.info(f"[HF] Loading model from: {self._model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        if self._load_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = torch.bfloat16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                quantization_config = bnb_config,
                device_map          = "auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype = torch.bfloat16,
                device_map  = "auto",
            )

        self._pipeline = pipeline(
            "text-generation",
            model     = self._model,
            tokenizer = self._tokenizer,
        )
        logger.info("[HF] Model loaded and ready.")

    def generate(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        top_p:       float = 0.95,
    ) -> str:
        """Generate a response using the chat template."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Apply the model's built-in chat template (handles Arabic RTL correctly)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize         = False,
            add_generation_prompt = True,
        )

        output = self._pipeline(
            prompt,
            max_new_tokens = max_tokens,
            temperature    = temperature,
            top_p          = top_p,
            do_sample      = temperature > 0,
            return_full_text = False,   # return only the generated part
        )

        return output[0]["generated_text"].strip()

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def model_name(self) -> str:
        return self._model_path
