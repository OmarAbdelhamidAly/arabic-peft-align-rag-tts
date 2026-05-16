from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioOutput:
    """Generated audio output entity."""
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"  # wav, mp3, etc.
    
    def save_to_file(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.audio_bytes)
