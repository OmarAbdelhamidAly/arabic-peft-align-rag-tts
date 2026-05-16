"""
fine_tuning/prepare_data.py
-----------------------------
TODO: Prepare the Arabic TTS dataset for XTTS-v2 fine-tuning.

Research questions to answer first:
  1. Which dataset will you use?
     - Common Voice 17 AR (~67 hrs) — crowd-sourced, good variety
     - Arabic Speech Corpus (3.7 hrs) — studio quality MSA
     - ClArTTS (~14 hrs) — classical Arabic, high quality
     OR combine them for maximum coverage.

  2. What format does XTTS-v2 fine-tuning require?
     - LJSpeech format: wavs/ folder + metadata.csv
     - metadata.csv columns: filename|transcription|normalized_transcription
     - WAV files: mono, 22050 Hz (XTTS-v2 will resample internally)

  3. What preprocessing steps are needed?
     - Resample audio to target Hz
     - Normalize volume (loudness normalization)
     - Remove silence at start/end
     - Filter by duration (typically 1–10 seconds per clip)
     - Validate Arabic text (remove diacritics? normalize hamza?)

Steps to implement:
  Step 1 — Download dataset (HuggingFace datasets library)
  Step 2 — Filter and clean audio clips
  Step 3 — Normalize and resample audio
  Step 4 — Write metadata.csv
  Step 5 — Validate output (run quick stats)

Useful libraries:
  datasets, librosa, soundfile, pydub, arabic_reshaper
"""


def download_dataset(dataset_name: str, output_dir: str) -> None:
    """TODO: Download and cache the chosen TTS dataset."""
    raise NotImplementedError("Research and implement this step first.")


def preprocess_audio(input_path: str, output_path: str, target_sr: int = 22050) -> None:
    """TODO: Resample, normalize, and trim silence from a WAV file."""
    raise NotImplementedError("Research and implement this step first.")


def build_metadata_csv(wavs_dir: str, output_csv: str) -> None:
    """TODO: Generate the LJSpeech-format metadata.csv."""
    raise NotImplementedError("Research and implement this step first.")


if __name__ == "__main__":
    # Data lives inside service-tts: data/tts/
    # Run from: services/service-tts/fine_tuning/
    download_dataset("mozilla-foundation/common_voice_17_0", "../data/tts/raw")
    build_metadata_csv("../data/tts/wavs", "../data/tts/metadata.csv")
