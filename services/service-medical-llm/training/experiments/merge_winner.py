"""
merge_winner.py
================
Merge the winning SimPO adapter into the base Qwen2.5-3B model.

Winner: SimPO (weighted_score=95.914, good_rate=100%, hard_fail=0%)

Run:
    conda activate unsloth_env
    python E:\\FineTuning\\services\\service-training\\experiments\\merge_winner.py

Output:
    experiments/merged_model_16bit/   (~6-7 GB)

Next step:
    huggingface-cli upload OmarAbdelhamid/arabic-medical-qwen2-simpo \
        experiments/merged_model_16bit --repo-type model
"""

import os
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
BASE_MODEL  = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"
ADAPTER_DIR = BASE_DIR / "02_post_training" / "qwen_medical_arabic_simpo"
OUTPUT_DIR  = BASE_DIR / "merged_model_16bit"

# ── Validation ────────────────────────────────────────────────────────────────
def validate():
    if not ADAPTER_DIR.exists():
        print(f"❌ Adapter not found: {ADAPTER_DIR}")
        sys.exit(1)

    adapter_weights = ADAPTER_DIR / "adapter_model.safetensors"
    if not adapter_weights.exists():
        print(f"❌ adapter_model.safetensors not found in {ADAPTER_DIR}")
        sys.exit(1)

    print(f"✅ Adapter found: {ADAPTER_DIR}")
    print(f"   Size: {adapter_weights.stat().st_size / 1e6:.1f} MB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output dir ready: {OUTPUT_DIR}")


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge():
    print("\n" + "="*60)
    print("  Merging SimPO adapter → Qwen2.5-3B (16-bit)")
    print("="*60)

    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in unsloth_env: conda activate unsloth_env")
        sys.exit(1)

    t0 = time.time()

    # 1. Load base model in 4-bit (for efficient loading)
    print("\n[1/4] Loading base model (4-bit for speed)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = 4096,
        dtype          = None,
        load_in_4bit   = True,
    )
    print(f"      ✅ Base model loaded ({time.time()-t0:.0f}s)")

    # 2. Load SimPO adapter on top
    print(f"\n[2/4] Loading SimPO adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    print(f"      ✅ Adapter loaded ({time.time()-t0:.0f}s)")

    # 3. Merge adapter weights into base model (dequantize + merge)
    print("\n[3/4] Merging adapter into base model (this takes a few minutes)...")
    model = model.merge_and_unload()
    print(f"      ✅ Merge complete ({time.time()-t0:.0f}s)")

    # 4. Save merged model in 16-bit
    print(f"\n[4/4] Saving merged 16-bit model to: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"      ✅ Saved ({time.time()-t0:.0f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    elapsed    = time.time() - t0

    print("\n" + "="*60)
    print("  ✅ MERGE COMPLETE!")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Size:   {total_size / 1e9:.2f} GB")
    print(f"  Time:   {elapsed:.0f}s")
    print("="*60)
    print("""
Next steps:
  1. Login to HuggingFace:
       huggingface-cli login

  2. Upload the merged model:
       huggingface-cli upload OmarAbdelhamid/arabic-medical-qwen2-simpo \\
           experiments/merged_model_16bit --repo-type model

  3. Test with vLLM:
       docker run --gpus all -p 8080:8080 vllm/vllm-openai:v0.6.3 \\
           --model OmarAbdelhamid/arabic-medical-qwen2-simpo \\
           --max-model-len 4096
""")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    validate()
    merge()
