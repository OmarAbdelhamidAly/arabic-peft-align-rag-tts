from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeakerEmbedding:
    """Speaker voice embedding for cloning."""
    embedding: list[float]
    speaker_id: str
    sample_audio_path: Optional[str] = None
    
    def to_tensor(self):
        import torch
        return torch.tensor(self.embedding)
