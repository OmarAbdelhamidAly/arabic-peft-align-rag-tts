"""
Test script: Test model via HuggingFace Inference API
=====================================================
Tests the model without local loading using HuggingFace Inference API.
"""

import os
from pathlib import Path

# Load .env
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
def load_dotenv(dotenv_path: Path):
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

load_dotenv(ROOT_DIR / ".env")

MODEL_ID = "OmarAbdelhamid/arabic-medical-qwen2-simpo"
token = os.environ.get("HF_TOKEN", "")

import os



print(f"Testing model: {MODEL_ID}")
print(f"Using HF Token: {HF_TOKEN[:20]}...")

# Try using InferenceClient (text generation instead of chat)
try:
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(token=HF_TOKEN)
    
    # Format prompt manually
    prompt = "أنت معالج نفسي عربي متخصص، تتعامل بتتعاطف واحترافية.\n\nالسؤال: أشعر بالحزن الشديد مؤخراً، ماذا أفعل؟\n\nالرد:"
    
    response = client.text_generation(
        model=MODEL_ID,
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.7,
    )
    
    print(f"\n✅ Response:\n{response}")
    
except ImportError:
    print("❌ huggingface_hub not installed")
    print("Install it: pip install huggingface_hub")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nNote: The model might not be available on Inference API")
