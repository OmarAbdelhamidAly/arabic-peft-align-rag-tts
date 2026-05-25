import os
from pathlib import Path
from pipeline import run_alignment

BASE = Path(__file__).resolve().parent.parent

# Set up local paths
SFT_ADAPTER_PATH = str(BASE / "outputs/test_sft")  # Uses the test SFT adapter output
DATASET_PATH     = str(BASE / "data/alignment")    # Directory where 05_simpo_dataset.json resides
OUTPUT_DIR       = str(BASE / "outputs/test_alignment")
METHOD           = "simpo"  # Change to dpo, ipo, kto, orpo, etc. to test others

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify inputs exist
if not os.path.exists(SFT_ADAPTER_PATH):
    print(f"Warning: SFT adapter path '{SFT_ADAPTER_PATH}' does not exist.")
    print("Please run SFT training test first using: python test_sft.py")
    
if not os.path.exists(DATASET_PATH):
    print(f"Error: Alignment dataset directory '{DATASET_PATH}' does not exist.")
    exit(1)

print(f"Testing Local Alignment for method: {METHOD}...")
result = run_alignment.python_func(
    sft_adapter_path=SFT_ADAPTER_PATH,
    dataset_path=DATASET_PATH,
    output_dir=OUTPUT_DIR,
    method=METHOD,
    num_epochs=1,          # Run for 1 epoch only for quick testing
    learning_rate=5e-5
)

print(f"Alignment ({METHOD}) Test Result: {result}")
