from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSRequest:
    """Text-to-speech request entity."""
    text: str
    language: str = "ar"  # Arabic default
    speaker_wav: Optional[str] = None  # For voice cloning
    speed: float = 1.0
    
    def validate(self) -> bool:
        return bool(self.text and len(self.text.strip()) > 0)
