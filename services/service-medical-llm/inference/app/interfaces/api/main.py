"""
FastAPI Entry Point: service-inference
-----------------------------------------
Serves the fine-tuned Arabic Medical LLM via an OpenAI-compatible REST API.

Endpoints:
  POST  /v1/chat/completions   ← main inference endpoint
  GET   /v1/models             ← list loaded models
  POST  /admin/load            ← (re)load a model at runtime
  GET   /health                ← liveness probe

Startup:
  Set MODEL_PATH env variable to the merged model directory, e.g.:
    MODEL_PATH=../../model_training/merged_model_16bit uvicorn app.interfaces.api.main:app

Architecture:
  FastAPI → GenerateCompletionUseCase → HuggingFaceLoader
  The use case and model loader are created once at startup (singleton).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from ...domain.entities.inference_request import ChatCompletionRequest
from ...domain.entities.inference_response import ChatCompletionResponse
from ...application.use_cases.generate_completion import GenerateCompletionUseCase
from ...infrastructure.models.hf_loader import HuggingFaceLoader

logger = logging.getLogger(__name__)

# ── Dependency wiring (singleton) ─────────────────────────────────────────────

_loader:   HuggingFaceLoader      | None = None
_use_case: GenerateCompletionUseCase | None = None


def get_use_case() -> GenerateCompletionUseCase:
    if _use_case is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /admin/load first.")
    return _use_case


# ── Lifespan: auto-load model if MODEL_PATH is set ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loader, _use_case
    model_path = os.getenv("MODEL_PATH", "")
    if model_path:
        logger.info(f"[STARTUP] Auto-loading model from: {model_path}")
        _loader   = HuggingFaceLoader(load_in_4bit=True)
        _loader.load(model_path)
        _use_case = GenerateCompletionUseCase(_loader)
        logger.info("[STARTUP] Model ready.")
    else:
        logger.warning("[STARTUP] MODEL_PATH not set. POST /admin/load to load a model.")
    yield
    logger.info("[SHUTDOWN] Cleaning up.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "service-inference",
    description = "Arabic Medical LLM — OpenAI-compatible inference API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request:  ChatCompletionRequest,
    use_case: GenerateCompletionUseCase = Depends(get_use_case),
):
    """
    OpenAI-compatible chat completion endpoint.

    Example request body:
    {
      "messages": [
        {"role": "system", "content": "أنت مساعد طبي عربي متخصص."},
        {"role": "user",   "content": "ما هي أعراض مرض السكري؟"}
      ],
      "max_tokens": 512,
      "temperature": 0.7
    }
    """
    try:
        return await use_case.execute(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error.")


@app.get("/v1/models")
async def list_models():
    """List currently loaded models (OpenAI-compatible)."""
    if _loader and _loader.is_loaded():
        return {
            "object": "list",
            "data": [{"id": _loader.model_name(), "object": "model"}],
        }
    return {"object": "list", "data": []}


@app.post("/admin/load")
async def load_model(model_path: str):
    """
    Load (or reload) a model at runtime.
    Useful for switching between different adapters without restarting.
    """
    global _loader, _use_case
    try:
        _loader   = HuggingFaceLoader(load_in_4bit=True)
        _loader.load(model_path)
        _use_case = GenerateCompletionUseCase(_loader)
        return {"status": "loaded", "model": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Liveness probe for Docker / Kubernetes."""
    return {
        "status":       "ok",
        "service":      "inference",
        "model_loaded": _loader.is_loaded() if _loader else False,
    }
