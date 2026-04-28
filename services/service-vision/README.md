# 👁️ service-vision (Placeholder)

> **Status:** 🚧 Planned — not yet implemented.

## Purpose

This service will handle Computer Vision tasks that complement the Arabic Medical LLM pipeline. Potential use cases:

- **Medical Image Analysis**: X-rays, MRI, CT scan interpretation using Vision-Language Models (VLMs)
- **Document OCR**: Extract Arabic text from scanned medical documents/prescriptions
- **Chart Understanding**: Parse lab results or medical charts visually

## Planned Models

| Model | Task | Link |
|-------|------|------|
| LLaVA-Med | Medical VQA | [HuggingFace](https://huggingface.co/microsoft/llava-med) |
| BioViL-T | Radiology report generation | [HuggingFace](https://huggingface.co/microsoft/BioViL-T) |
| Arabic-OCR | Arabic document OCR | TBD |

## Integration with Other Services

```
Medical Image Input
      ↓
service-vision (OCR / VQA)
      ↓
Arabic Text Output
      ↓
service-rag → service-inference → service-tts
```
