"""
service-inference: FastAPI entry point.
Exposes the fine-tuned LLM as a REST API.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Arabic Medical LLM Inference",
    description="Fine-tuned Arabic medical LLM inference service",
    version="0.1.0",
)


class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    generated_text: str
    model_name: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest) -> InferenceResponse:
    # TODO: wire to infrastructure/models layer
    raise NotImplementedError("Inject fine-tuned model via use case.")
