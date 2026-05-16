from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import hashlib
import io

from ....domain.entities.tts_request import TTSRequest
from ....application.use_cases.synthesize_speech import SynthesizeSpeechUseCase
from ....infrastructure.models.xtts_v2_adapter import XTTSv2Adapter


app = FastAPI(title="Service TTS", version="0.1.0")


# Pydantic schemas
class TTSRequestSchema(BaseModel):
    text: str
    language: str = "ar"
    speaker_wav: str | None = None  # Path or reference to cloned voice


class TTSResponseSchema(BaseModel):
    audio_url: str
    duration_seconds: float
    sample_rate: int


# Singleton dependencies
_tts_adapter = None
_use_case = None


def get_use_case():
    global _tts_adapter, _use_case
    
    if _use_case is None:
        _tts_adapter = XTTSv2Adapter()
        _use_case = SynthesizeSpeechUseCase(synthesizer=_tts_adapter)
    
    return _use_case


@app.post("/tts/synthesize")
async def synthesize(request: TTSRequestSchema):
    """Generate speech from text."""
    try:
        tts_request = TTSRequest(
            text=request.text,
            language=request.language,
            speaker_wav=request.speaker_wav
        )
        
        use_case = get_use_case()
        result = await use_case.execute(tts_request)
        
        # Stream audio response
        audio = result.audio
        audio_stream = io.BytesIO(audio.audio_bytes)
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers={
                "X-Duration": str(audio.duration_seconds),
                "X-Sample-Rate": str(audio.sample_rate)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/clone-voice")
async def clone_voice(
    speaker_id: str,
    sample_audio: UploadFile = File(...)
):
    """Clone voice from audio sample."""
    # Save uploaded file temporarily
    temp_path = f"/tmp/{speaker_id}_{sample_audio.filename}"
    with open(temp_path, "wb") as f:
        f.write(await sample_audio.read())
    
    # TODO: Implement voice cloning use case
    
    return {"speaker_id": speaker_id, "status": "cloned"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tts"}


# Fine-tuning endpoints
@app.post("/admin/fine-tune/prepare")
async def prepare_fine_tuning():
    """Prepare model for domain-specific fine-tuning."""
    adapter = XTTSv2Adapter()
    adapter.prepare_for_fine_tuning("./fine_tuning_output")
    return {"status": "prepared", "output_path": "./fine_tuning_output"}


@app.post("/admin/fine-tune/load")
async def load_fine_tuned(checkpoint_path: str):
    """Load fine-tuned model weights."""
    global _tts_adapter, _use_case
    
    _tts_adapter = XTTSv2Adapter(fine_tuned_path=checkpoint_path)
    _use_case = SynthesizeSpeechUseCase(synthesizer=_tts_adapter)
    
    return {"status": "loaded", "checkpoint": checkpoint_path}
