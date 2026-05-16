# 🔊 service-tts — TTS Fine-Tuning + Speech Synthesis API

> **Dual role:**
> 1. **Offline** — fine-tune XTTS-v2 on Arabic medical speech data
> 2. **Online** — serve a REST API that synthesizes Arabic speech from text

---

## 🏗️ Clean Architecture

```
service-tts/
├── app/                                    ← Online API (FastAPI)
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── tts_request.py              ← TTSRequest entity
│   │   │   ├── audio_output.py             ← AudioOutput entity
│   │   │   └── speaker.py                  ← SpeakerEmbedding entity
│   │   └── interfaces/
│   │       ├── synthesizer.py              ← Synthesizer ABC  ← TODO
│   │       └── voice_cloner.py             ← VoiceCloner ABC  ← TODO
│   ├── application/
│   │   └── use_cases/
│   │       ├── synthesize_speech.py        ← SynthesizeSpeechUseCase ✅
│   │       └── clone_voice.py              ← CloneVoiceUseCase ✅
│   ├── infrastructure/
│   │   ├── models/
│   │   │   └── xtts_v2_adapter.py          ← XTTS-v2 Synthesizer impl ✅
│   │   └── persistence/                    ← Speaker cache (TODO)
│   └── interfaces/
│       └── api/
│           └── main.py                     ← FastAPI app ✅
│
├── fine_tuning/                            ← Offline training scripts
│   ├── prepare_data.py                     ← Download + format TTS dataset  ← TODO
│   ├── train_xtts.py                       ← XTTS-v2 fine-tuning loop       ← TODO
│   └── evaluate_tts.py                     ← WER / RTF evaluation            ← TODO
│
├── data/tts/                               ← Audio dataset (fill in after research)
│   ├── wavs/                               ← WAV audio clips
│   └── metadata.csv                        ← LJSpeech format: filename|text|normalized
│
├── Dockerfile
└── requirements.txt
```

---

## 🔄 Two Flows

### Flow 1 — Offline Fine-Tuning (run once)
```
Choose dataset (Common Voice AR / ClArTTS / Arabic Speech Corpus)
        │
        ▼
fine_tuning/prepare_data.py
  → Download audio clips
  → Resample to 22050 Hz (mono)
  → Remove silence, normalize volume
  → Filter by duration (1–10 sec)
  → Write data/tts/wavs/ + metadata.csv
        │
        ▼
fine_tuning/train_xtts.py
  → Load XTTS-v2 pretrained weights
  → Configure Coqui TTS Trainer
  → Fine-tune on Arabic medical vocabulary
  → Save checkpoint → outputs/xtts_arabic_medical_v1/
        │
        ▼
fine_tuning/evaluate_tts.py
  → WER with Whisper (target: < 5%)
  → RTF measurement (target: < 1.0)
  → MOS subjective listening test
```

### Flow 2 — Online API (always running)
```
Client (service-rag or curl)
        │
        │  POST /tts/synthesize
        │  {"text": "جرعة الدواء 500 ملجم", "language": "ar"}
        ▼
   FastAPI (interfaces/api/main.py)
        │  validates request
        ▼
   SynthesizeSpeechUseCase (application)
        │  creates TTSRequest domain entity
        ▼
   XTTSv2Adapter (infrastructure)
        │  loads fine-tuned model
        │  tts.tts(text, language, speaker_wav)
        │  → numpy audio array @ 24000 Hz
        ▼
   StreamingResponse (WAV audio bytes)
```

---

## ⚙️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| TTS model | XTTS-v2 (Coqui TTS) | Supports Arabic, voice cloning, fine-tunable |
| Fine-tuning | Coqui TTS Trainer API | Built-in training loop for XTTS |
| Audio processing | `librosa`, `soundfile`, `pydub` | Resample, normalize, trim |
| Evaluation | `openai-whisper` + `jiwer` | WER computation |
| Web API | FastAPI + uvicorn | Async, streaming audio responses |
| Audio format | WAV @ 24000 Hz | XTTS-v2 native output sample rate |

---

## 📚 Research Roadmap (What to Learn Next)

### Step 1 — Understand the TTS datasets
```
Common Voice 17 (ar)       → 67 hrs, crowd-sourced, varied speakers
                               Good for: speaker diversity
Arabic Speech Corpus       → 3.7 hrs, studio MSA quality
                               Good for: clean pronunciation baseline
ClArTTS                    → 14 hrs, classical Arabic
                               Good for: formal medical register
```
**Recommendation:** Start with Arabic Speech Corpus (small, clean) to validate
your pipeline, then add Common Voice AR for diversity.

### Step 2 — Understand XTTS-v2 fine-tuning
Key resources to read:
- Coqui TTS Trainer docs: https://tts.readthedocs.io/en/latest/
- XTTS-v2 paper: https://arxiv.org/abs/2406.04904
- Fine-tuning guide: https://github.com/coqui-ai/TTS/discussions/3517

Key questions:
- What `XttsConfig` parameters matter? (`lr`, `batch_size`, `epochs`)
- How is the `speaker_encoder` used during fine-tuning?
- What is `GPT-2` doing inside XTTS-v2? (it's an autoregressive token predictor)

### Step 3 — Evaluate your fine-tuned model
```python
# WER evaluation with Whisper
import whisper, jiwer
model = whisper.load_model("large-v3")
result = model.transcribe("output.wav", language="ar")
wer = jiwer.wer(reference_text, result["text"])
```

### Step 4 — Build the voice system
After fine-tuning:
1. Load the checkpoint in `XTTSv2Adapter.__init__`
2. The online API at `POST /tts/synthesize` is already scaffolded
3. Test voice cloning with a real Arabic speaker WAV file

---

## 🚀 Usage

### Fine-tuning (once you implement the scripts):
```bash
cd services/service-tts
pip install TTS torch librosa soundfile jiwer openai-whisper

# Prepare data
python fine_tuning/prepare_data.py

# Train
python fine_tuning/train_xtts.py

# Evaluate
python fine_tuning/evaluate_tts.py
```

### Online API:
```bash
TTS_MODEL=./outputs/xtts_arabic_medical_v1 \
uvicorn app.interfaces.api.main:app --host 0.0.0.0 --port 8002
```

### Test synthesis:
```bash
curl -X POST http://localhost:8002/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "جرعة الدواء خمسمئة ملليغرام يومياً", "language": "ar"}' \
  --output response.wav
```

---

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tts/synthesize` | Generate Arabic speech → WAV stream |
| `POST` | `/tts/clone-voice` | Clone speaker from WAV sample |
| `POST` | `/admin/fine-tune/load` | Load a fine-tuned checkpoint |
| `GET`  | `/health` | Liveness probe |
