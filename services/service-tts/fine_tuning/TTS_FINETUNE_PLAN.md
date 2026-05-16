# Mental Health TTS Fine-Tuning Plan (Arabic XTTS-v2)

This document outlines the strategy for fine-tuning the TTS model to serve as an empathetic Mental Health Assistant.

## 1. Data Collection Strategy (The "Voice of Therapy")
For Mental Health, the model must sound **calm, slow, and supportive**.

- **Target Duration:** 1-2 hours of high-quality speech.
- **Vocabulary Focus:** 
    - Emotional support phrases (e.g., "أنا أتفهم ما تمر به", "لا بأس بأن تشعر بهذا").
    - Psychological terminology (e.g., "الاكتئاب", "اضطراب ما بعد الصدمة", "الناقلات العصبية").
- **Audio Quality:** 44.1kHz, Mono, WAV format, absolutely no background noise or music.

## 2. Preprocessing Pipeline
1. **Audio Segmentation:** Break long recordings into 3-10 second chunks.
2. **Transcription:** Use `OpenAI Whisper (Large-v3)` to generate precise Arabic transcripts.
3. **Normalization:** Convert numbers and abbreviations into full Arabic words (e.g., "5 دقائق" -> "خمسة دقائق").

## 3. Fine-Tuning Methodology (XTTS-v2)
- **Model:** Coqui TTS `xtts_v2`.
- **Technique:** GPT conditioning fine-tuning.
- **Hyperparameters:**
    - `batch_size`: 4-8 (depending on VRAM).
    - `learning_rate`: 1e-5 to 5e-6 (very low to preserve base quality).
    - `epochs`: 10-20.
- **Reference Voice:** A single 6-second high-quality clip of a professional counselor's voice.

## 4. Evaluation Metrics
- **MOS (Mean Opinion Score):** Subjective human rating of "Empathy" and "Naturalness".
- **WER (Word Error Rate):** Using Whisper to transcribe the output and check if it matches the input text.
- **Stress Test:** Testing the model with long medical explanations to check for "robotic" drift.

## 5. Directory Mapping
- Data: `services/service-tts/data/tts/wavs/`
- Metadata: `services/service-tts/data/tts/metadata.csv`
- Checkpoints: `services/service-tts/data/tts/outputs/`
