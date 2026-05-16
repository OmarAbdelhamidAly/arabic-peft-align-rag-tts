"""
fine_tuning/evaluate_tts.py
-----------------------------
TODO: Evaluate the fine-tuned TTS model quality.

Metrics to compute:
  1. WER (Word Error Rate)
     - Use Whisper (openai/whisper) to transcribe generated audio
     - Compare with ground-truth text using jiwer
     - Lower = better. Target: < 5% WER on test set

  2. MOS (Mean Opinion Score)  [Human evaluation]
     - Play generated audio to Arabic speakers
     - Rate naturalness 1–5
     - Target: > 4.0 MOS

  3. RTF (Real-Time Factor)
     - RTF = generation_time / audio_duration
     - RTF < 1.0 means faster than real-time (required for production)

  4. Arabic-specific checks
     - Does it pronounce emphatic consonants correctly? (ص ض ط ظ)
     - Does it handle diacritics (tashkeel) properly?
     - Medical terminology pronunciation accuracy

Useful libraries: jiwer, openai-whisper, soundfile
"""


def compute_wer(model_path: str, test_csv: str) -> float:
    """TODO: Compute Word Error Rate using Whisper transcription."""
    raise NotImplementedError


def compute_rtf(model_path: str, test_texts: list) -> float:
    """TODO: Compute Real-Time Factor for the TTS model."""
    raise NotImplementedError


if __name__ == "__main__":
    wer = compute_wer("./outputs/xtts_arabic_medical_v1", "./data/tts/test_metadata.csv")
    print(f"WER: {wer:.2%}")
