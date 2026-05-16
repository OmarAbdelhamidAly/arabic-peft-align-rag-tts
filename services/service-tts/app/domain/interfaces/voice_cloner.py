"""
Domain Interface: VoiceCloner
------------------------------
TODO: Abstract contract for voice cloning from a speaker audio sample.

Questions to answer during your research:
  - How does XTTS-v2 voice cloning work?
    (It uses a reference .wav file as a speaker embedding — no explicit extraction)
  - What is the minimum audio length for a good clone?
    (XTTS-v2 recommends 6–30 seconds of clean speech)
  - Should we store speaker embeddings? If yes, in what format?
"""

from abc import ABC, abstractmethod


class VoiceCloner(ABC):
    """Abstract base for voice cloning backends."""

    @abstractmethod
    async def clone_from_sample(self, audio_path: str) -> object:
        """Create a speaker reference from a WAV audio file."""
        ...

    @abstractmethod
    def save_speaker(self, embedding: object, speaker_id: str) -> None:
        """Persist a speaker reference for reuse."""
        ...
