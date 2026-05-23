"""
Test script: Test inference with HuggingFace model
==================================================
Tests the model directly from HuggingFace without local deployment.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "OmarAbdelhamid/arabic-medical-qwen2-simpo"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Test prompt
messages = [
    {"role": "system", "content": "أنت معالج نفسي عربي متخصص، تتعامل بتتعاطف واحترافية."},
    {"role": "user", "content": "أشعر بالحزن الشديد مؤخراً، ماذا أفعل؟"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"\nResponse:\n{response}")
