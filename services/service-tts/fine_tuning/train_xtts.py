"""
fine_tuning/train_xtts.py
---------------------------
TODO: Fine-tune XTTS-v2 on the prepared Arabic medical TTS dataset.

Research questions to answer first:
  1. How does XTTS-v2 fine-tuning work?
     - Uses Coqui TTS Trainer API
     - Config: XttsConfig + XttsAudioConfig
     - Training script: use TTS/bin/train_tts.py OR the Trainer class directly

  2. What are the recommended hyperparameters?
     - Learning rate: 1e-5 to 5e-5 (lower than text models)
     - Batch size: 4–8 (audio is memory-heavy)
     - Epochs: 5–30 depending on dataset size
     - Warm-up steps: ~500

  3. What does the fine-tuned model output?
     - A checkpoint directory with model weights + config.json
     - Load it via: TTS(model_path="./path/to/checkpoint")

  4. How to evaluate TTS quality?
     - WER (Word Error Rate) with a Whisper ASR model
     - MOS (Mean Opinion Score) — human listening test
     - RTF (Real-Time Factor) — how fast it generates vs real-time

Useful libraries:
  TTS (Coqui), torch, librosa, jiwer (for WER)
"""


def fine_tune_xtts(
    dataset_path: str,
    output_path:  str,
    base_model:   str = "tts_models/multilingual/multi-dataset/xtts_v2",
    epochs:       int = 10,
    batch_size:   int = 4,
    learning_rate: float = 5e-5,
) -> None:
    """
    TODO: Run the XTTS-v2 fine-tuning loop.

    Args:
        dataset_path:  Path to directory with wavs/ + metadata.csv
        output_path:   Where to save the fine-tuned checkpoint
        base_model:    Pretrained XTTS-v2 model name
        epochs:        Training epochs
        batch_size:    Batch size (reduce if OOM)
        learning_rate: Learning rate
    """
    raise NotImplementedError("Research XTTS-v2 Trainer API and implement this.")


if __name__ == "__main__":
    # Data lives inside service-tts: data/tts/
    # Run from: services/service-tts/fine_tuning/
    fine_tune_xtts(
        dataset_path = "../data/tts",
        output_path  = "../data/tts/outputs/xtts_arabic_medical_v1",
    )
