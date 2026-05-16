# Quick Reference Guide - Training Notebook

## 🚀 Quick Start (5 Steps)

### 1. Install Dependencies
```bash
pip install torch transformers datasets trl unsloth accelerate peft bitsandbytes
```

### 2. Verify GPU
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### 3. Configure Training
Edit Cell 2 in the notebook:
```python
class Config:
    MAX_STEPS = 100           # Start small, then increase to 500-1000
    LORA_R = 16              # 8=fast, 16=balanced, 32=quality
    LEARNING_RATE = 2e-4     # Try 1e-4 or 5e-4
    DATA_PATH = "../data_pipeline/processed_data/chatml_dataset.json"
```

### 4. Run All Cells
- Click "Run All" in Jupyter
- Or execute cells one by one (Shift+Enter)

### 5. Test Your Model
Cell 8 automatically tests with sample prompts. Add your own:
```python
test_prompts = [
    "Your test question in Arabic",
    "Another test question",
]
```

---

## 📊 Training Configuration Cheat Sheet

### For 8GB VRAM (RTX 3060)
```python
MAX_SEQ_LENGTH = 512
LORA_R = 16
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
```

### For 12GB VRAM (RTX 3060 Ti)
```python
MAX_SEQ_LENGTH = 768
LORA_R = 32
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
```

### For 6GB VRAM (GTX 1060)
```python
MAX_SEQ_LENGTH = 256
LORA_R = 8
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
```

---

## ⚡ Performance Tips

### Speed Up Training
```python
MAX_SEQ_LENGTH = 256        # Shorter sequences
GRADIENT_ACCUMULATION_STEPS = 4  # Fewer accumulation
```

### Improve Quality
```python
MAX_STEPS = 1000           # Train longer
LORA_R = 32               # Higher rank
LEARNING_RATE = 1e-4      # Lower learning rate
```

### Save VRAM
```python
MAX_SEQ_LENGTH = 256       # Shorter sequences
LORA_R = 8                # Lower rank
BATCH_SIZE = 1            # Keep at 1
```

---

## 🔍 Monitoring Training

### What to Watch
1. **VRAM Usage**: Should stay under your GPU limit
   ```
   Step 20: VRAM Allocated=5.2GB, Reserved=6.1GB
   ```

2. **Training Loss**: Should decrease over time
   ```
   Step 50: train_loss=1.234
   Step 100: train_loss=0.567  ← Good!
   ```

3. **Validation Loss**: Should track training loss
   ```
   eval_loss: 0.589
   ```

### Signs of Problems

❌ **Overfitting**: Validation loss increases while training loss decreases
```
Solution: 
- Reduce MAX_STEPS
- Increase dataset size
- Add more dropout
```

❌ **Underfitting**: Both losses remain high
```
Solution:
- Increase MAX_STEPS
- Increase LORA_R
- Lower LEARNING_RATE
```

❌ **OOM (Out of Memory)**
```
Solution:
- Reduce MAX_SEQ_LENGTH
- Reduce LORA_R
- Increase GRADIENT_ACCUMULATION_STEPS
```

---

## 💾 File Outputs

After training, you'll have:

### 1. Checkpoints (During Training)
```
checkpoints/
  ├── checkpoint-20/
  ├── checkpoint-40/
  └── checkpoint-60/
```

### 2. Final Model
```
qwen_medical_arabic_lora/
  ├── adapter_config.json
  ├── adapter_model.safetensors  ← LoRA weights (~30-50 MB)
  ├── tokenizer_config.json
  └── special_tokens_map.json
```

---

## 🧪 Testing Your Model

### Option 1: Use Cell 8 (Automatic)
The notebook automatically tests with sample prompts.

### Option 2: Manual Testing
```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen_medical_arabic_lora",
    max_seq_length=512,
    load_in_4bit=True,
)

# Inference mode
FastLanguageModel.for_inference(model)

# Test
messages = [{"role": "user", "content": "ما هي أعراض السكري؟"}]
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🐛 Troubleshooting

### Problem: "Dataset not found"
```python
# Check the path in Config class
DATA_PATH = "../data_pipeline/processed_data/chatml_dataset.json"

# Verify file exists
from pathlib import Path
print(f"File exists: {Path(DATA_PATH).exists()}")
```

### Problem: "CUDA out of memory"
```python
# Reduce these in Config class:
MAX_SEQ_LENGTH = 256  # Was 512
LORA_R = 8           # Was 16
```

### Problem: "Training too slow"
```python
# Check GPU usage:
# Should be near 100%
import subprocess
subprocess.run(["nvidia-smi"])

# If GPU usage is low, increase batch size:
BATCH_SIZE = 2  # If VRAM allows
```

### Problem: "Model quality is poor"
```python
# Train longer with better parameters:
MAX_STEPS = 500      # Was 100
LORA_R = 32         # Was 16
LEARNING_RATE = 1e-4  # Was 2e-4
```

---

## 📈 Expected Results

### After 100 Steps (~10 minutes)
- ✅ Model works and generates text
- ⚠️ Quality may be basic
- 💡 Good for testing pipeline

### After 500 Steps (~50 minutes)
- ✅ Better quality responses
- ✅ More coherent Arabic
- ✅ Better medical knowledge

### After 1000 Steps (~100 minutes)
- ✅ High-quality responses
- ✅ Consistent formatting
- ✅ Production-ready

---

## 🎯 Optimal Settings (Recommended)

For most cases on RTX 3060:
```python
class Config:
    # Model
    MAX_SEQ_LENGTH = 512
    LOAD_IN_4BIT = True
    
    # LoRA
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Training
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    MAX_STEPS = 500
    WARMUP_STEPS = 10
    
    # Saving
    SAVE_STEPS = 20
    LOGGING_STEPS = 5
```

---

## 🔄 After Training

### Deploy Locally
```python
# Use the model in your app
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "qwen_medical_arabic_lora",
    max_seq_length=512,
    load_in_4bit=True,
)
```

### Push to HuggingFace
```python
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

### Create Merged Model (Optional)
```python
# For standalone deployment (larger file)
model.save_pretrained_merged(
    "qwen_medical_arabic_merged",
    tokenizer,
    save_method="merged_16bit"
)
```

---

## 📞 Quick Help

**GPU Not Detected?**
```bash
nvidia-smi  # Check GPU
nvcc --version  # Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Need More Memory?**
- Close other applications
- Reduce MAX_SEQ_LENGTH
- Reduce LORA_R
- Increase GRADIENT_ACCUMULATION_STEPS

**Training Not Converging?**
- Increase MAX_STEPS
- Try different LEARNING_RATE
- Check your dataset quality
- Monitor validation loss

**Model Not Learning?**
- Verify dataset format is correct
- Check training loss is decreasing
- Ensure enough training steps
- Try increasing LORA_R

---

## ✅ Checklist Before Training

- [ ] GPU is available and detected
- [ ] CUDA toolkit installed
- [ ] All dependencies installed
- [ ] Dataset file exists and accessible
- [ ] Config parameters set appropriately
- [ ] Output directories writable
- [ ] Enough disk space (~5 GB free)
- [ ] No other GPU applications running

---

## 🎓 Remember

1. **Start Small**: Test with 100 steps first
2. **Monitor**: Watch VRAM and losses
3. **Save Often**: Checkpoints every 20 steps
4. **Test Early**: Run Cell 8 after training
5. **Iterate**: Adjust config based on results

**Good luck with your training! 🚀**
