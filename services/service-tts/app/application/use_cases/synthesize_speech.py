from dataclasses import dataclass

from ...domain.entities.tts_request import TTSRequest
from ...domain.entities.audio_output import AudioOutput
from ...domain.interfaces.synthesizer import Synthesizer


@dataclass
class SynthesisResult:
    audio: AudioOutput
    text_length: int
    processing_time_ms: float


class SynthesizeSpeechUseCase:
    """Orchestrates text-to-speech generation."""
    
    def __init__(self, synthesizer: Synthesizer):
        self.synthesizer = synthesizer
    
    async def execute(self, request: TTSRequest) -> SynthesisResult:
        import time
        
        if not request.validate():
            raise ValueError("Invalid TTS request")
        
        if not self.synthesizer.supports_language(request.language):
            raise ValueError(f"Language {request.language} not supported")
        
        start = time.time()
        audio = await self.synthesizer.synthesize(request)
        elapsed_ms = (time.time() - start) * 1000
        
        return SynthesisResult(
            audio=audio,
            text_length=len(request.text),
            processing_time_ms=elapsed_ms
        )
