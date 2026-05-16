"""
Domain Interface: Synthesizer
-------------------------------
TODO: Abstract contract for any TTS synthesis engine.

Questions to answer during your research:
  - What inputs does XTTS-v2 require?  (text, language, speaker_wav)
  - What does it return?               (numpy array of audio samples)
  - What sample rate does it use?      (24000 Hz for XTTS-v2)

Once you know, define the method signature here.
"""

from abc import ABC, abstractmethod


class Synthesizer(ABC):
    """Abstract base for TTS synthesis backends (XTTS-v2, Parler-TTS, etc.)."""

    @abstractmethod
    async def synthesize(self, request) -> object:
        """Generate audio from text. Define request/response types after research."""
        ...

    @abstractmethod
    def supports_language(self, lang: str) -> bool:
        """Return True if this synthesizer supports the given language code."""
        ...
