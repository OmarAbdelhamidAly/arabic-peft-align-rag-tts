import torch
from TTS.api import TTS

from ...domain.entities.tts_request import TTSRequest
from ...domain.entities.audio_output import AudioOutput
from ...domain.entities.speaker import SpeakerEmbedding
from ...domain.interfaces.synthesizer import Synthesizer
from ...domain.interfaces.voice_cloner import VoiceCloner


class XTTSv2Adapter(Synthesizer, VoiceCloner):
    """
    XTTS-v2 adapter supporting:
    - Text-to-speech synthesis
    - Voice cloning (zero-shot)
    - Fine-tuning ready structure
    """
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = None,
        fine_tuned_path: str = None  # Path to fine-tuned model
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model (fine-tuned or pretrained)
        if fine_tuned_path:
            self.tts = TTS(model_path=fine_tuned_path).to(self.device)
        else:
            self.tts = TTS(model_name).to(self.device)
        
        self._speakers = {}  # In-memory speaker cache
    
    async def synthesize(self, request: TTSRequest) -> AudioOutput:
        """Generate speech using XTTS-v2."""
        import io
        import time
        
        start = time.time()
        
        # Generate
        wav = self.tts.tts(
            text=request.text,
            language=request.language,
            speaker_wav=request.speaker_wav
        )
        
        # Convert to bytes
        import scipy.io.wavfile as wavfile
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, wav)
        audio_bytes = buffer.getvalue()
        
        duration = len(wav) / 24000
        
        return AudioOutput(
            audio_bytes=audio_bytes,
            sample_rate=24000,
            duration_seconds=duration,
            format="wav"
        )
    
    def supports_language(self, lang: str) -> bool:
        """XTTS-v2 supports: ar, en, fr, de, etc."""
        supported = {"ar", "en", "fr", "de", "es", "it", "pt", "pl", "tr", "ru", "nl", "cs"}
        return lang in supported
    
    async def clone_from_sample(self, audio_path: str) -> SpeakerEmbedding:
        """Create speaker embedding from audio sample."""
        # XTTS-v2 uses the audio file directly as speaker reference
        # No explicit embedding extraction needed
        embedding = [0.0] * 512  # Placeholder
        
        return SpeakerEmbedding(
            embedding=embedding,
            speaker_id="temp",
            sample_audio_path=audio_path
        )
    
    def save_speaker(self, embedding: SpeakerEmbedding, speaker_id: str) -> None:
        """Cache speaker reference."""
        self._speakers[speaker_id] = embedding
    
    # ============== Fine-tuning Methods ==============
    
    def prepare_for_fine_tuning(self, output_path: str):
        """Export model structure for fine-tuning."""
        # This prepares the model for domain-specific training
        # on Arabic medical vocabulary
        pass
    
    def load_fine_tuned(self, checkpoint_path: str):
        """Load fine-tuned weights."""
        self.tts = TTS(model_path=checkpoint_path).to(self.device)
