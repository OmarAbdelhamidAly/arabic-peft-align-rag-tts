import os
from pipeline import run_sft

# Dataset is at experiments/data_pipeline/processed_data/chatml_dataset.json
DATASET_PATH = "E:/FineTuning/services/service-medical-llm/training/experiments/data_pipeline/processed_data/chatml_dataset.json"
OUTPUT_DIR   = "E:/FineTuning/services/service-medical-llm/training/outputs/test_sft"

os.makedirs(OUTPUT_DIR, exist_ok=True)

result = run_sft.python_func(
    dataset_path=DATASET_PATH,
    output_dir=OUTPUT_DIR,
    base_model='unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit'
)

print('SFT Result:', result)