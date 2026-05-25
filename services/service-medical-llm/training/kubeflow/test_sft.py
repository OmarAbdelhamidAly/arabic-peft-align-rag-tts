# import os
# from pathlib import Path
# from pipeline import run_sft

# BASE = Path(__file__).resolve().parent.parent

# # Dataset is at experiments/data_pipeline/processed_data/chatml_dataset.json
# DATASET_PATH = str(BASE / "experiments/data_pipeline/processed_data/chatml_dataset.json")
# OUTPUT_DIR   = str(BASE / "outputs/test_sft")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# result = run_sft.python_func(
#     dataset_path=DATASET_PATH,
#     output_dir=OUTPUT_DIR,
#     base_model='unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit'
# )

# print('SFT Result:', result)





import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_sft

os.environ["WORKSPACE_PATH"] = r"E:\FineTuning\services\service-medical-llm\training"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

result = run_sft.python_func(
    dataset_path=r"E:\FineTuning\services\service-medical-llm\training\experiments\data_pipeline\processed_data\chatml_dataset.json",
    output_dir=r"E:\FineTuning\services\service-medical-llm\training\outputs\test_sft",
    base_model="unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    max_steps=100,
)

print(f"[DONE] SFT output: {result}")