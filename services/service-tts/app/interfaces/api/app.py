"""
service-tts: FastAPI entry point.
Converts Arabic text to speech using fine-tuned XTTS-v2.
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(
    title="Arabic Medical TTS Service",
    description="Fine-tuned XTTS-v2 for Arabic medical speech synthesis",
    version="0.1.0",
)


class TTSRequest(BaseModel):
    text: str
    language: str = "ar"
    speaker_wav: str | None = None  # path to reference voice (voice cloning)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(request: TTSRequest) -> StreamingResponse:
    # TODO: wire to infrastructure/models layer (XTTS-v2)
    raise NotImplementedError("Inject fine-tuned TTS model via use case.")
