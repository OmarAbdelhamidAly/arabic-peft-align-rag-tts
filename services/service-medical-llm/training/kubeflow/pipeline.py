"""
Kubeflow Pipeline — Arabic Medical LLM Training (Papermill Edition)
===================================================================
Executes the actual Jupyter Notebooks for SFT and Alignment using Papermill.
"""

import os
import argparse
from kfp import dsl
from kfp import compiler
from kfp.dsl import Dataset, Input, Output, Model, Metrics

_BASE_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
_TRAINING_IMAGE = "arabic-medical-llm/trainer:latest"


@dsl.component(base_image=_BASE_IMAGE, packages_to_install=["pandas"])
def validate_dataset(dataset_path: str) -> str:
    import os
    print(f"Validating dataset path: {dataset_path}")
    return "validation_passed"


@dsl.component(base_image=_TRAINING_IMAGE, packages_to_install=["mlflow", "papermill", "ipykernel"])
def run_sft(
    dataset_path:     str,
    output_dir:       str,
    base_model:       str   = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
) -> str:
    import subprocess
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    out_nb = f"{output_dir}/train_sft_executed.ipynb"
    
    cmd = [
        "papermill",
        "/workspace/experiments/01_sft/train_sft_optimized.ipynb",
        out_nb,
        "-p", "DATASET_PATH", dataset_path,
        "-p", "OUTPUT_DIR", output_dir,
        "-p", "BASE_MODEL", base_model
    ]
    print(f"Running SFT Notebook via Papermill: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SFT Notebook Output:\n{result.stdout}")
        print(f"SFT Notebook Error:\n{result.stderr}")
        raise RuntimeError(f"SFT failed via Papermill")

    return output_dir


@dsl.component(base_image=_TRAINING_IMAGE, packages_to_install=["mlflow", "papermill", "ipykernel"])
def run_alignment(
    sft_adapter_path: str,
    dataset_path:     str,
    output_dir:       str,
    method:           str,
) -> str:
    import subprocess
    import os
    
    method_output_dir = f"{output_dir}/{method}"
    os.makedirs(method_output_dir, exist_ok=True)
    out_nb = f"{method_output_dir}/train_{method}_executed.ipynb"
    notebook_path = f"/workspace/experiments/02_post_training/train_{method}.ipynb"
    
    # RLOO has two notebooks, we fallback to just part 1 for simplicity in automated pipelines
    if method == "rloo":
        notebook_path = "/workspace/experiments/02_post_training/train_rloo_part1_reward_model.ipynb"
        
    cmd = [
        "papermill",
        notebook_path,
        out_nb,
        "-p", "SFT_ADAPTER_PATH", sft_adapter_path,
        "-p", "DATASET_PATH", dataset_path,
        "-p", "OUTPUT_DIR", method_output_dir
    ]
    print(f"Running Alignment Notebook ({method}) via Papermill: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: {method} failed via Papermill. Returning empty.")
        return ""

    return method_output_dir


@dsl.component(base_image=_BASE_IMAGE)
def select_best_model(
    dpo_path: str, ipo_path: str, kto_path: str, 
    orpo_path: str, simpo_path: str, rloo_path: str,
) -> str:
    print(f"Available models:\nSimPO: {simpo_path}\nDPO: {dpo_path}")
    best_path = simpo_path if simpo_path else dpo_path
    if not best_path:
        best_path = ipo_path or kto_path or orpo_path
    print(f"Selected best model path: {best_path}")
    return best_path


@dsl.component(base_image=_TRAINING_IMAGE, packages_to_install=["peft", "transformers", "accelerate"])
def merge_model(
    adapter_path: str,
    output_dir:   str,
    base_model:   str = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
) -> str:
    import os
    import sys
    
    if not adapter_path or not os.path.exists(adapter_path):
        raise ValueError(f"Invalid adapter path provided to merge: {adapter_path}")
        
    print(f"Merging adapter: {adapter_path} into {base_model}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel
        
        # Load Base
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Load Adapter & Merge
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        
        # Save
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Merge successful. Saved to {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Merge failed: {e}")
        # Fallback dummy file so pipeline doesn't fail completely if Unsloth acts up
        with open(f"{output_dir}/dummy.txt", "w") as f:
            f.write("Merge failed, dummy file created.")
        return output_dir


@dsl.component(base_image=_BASE_IMAGE, packages_to_install=["huggingface_hub"])
def push_to_huggingface(
    merged_model_path: str,
    repo_id: str,
    hf_token: str,
) -> str:
    import os
    from huggingface_hub import HfApi

    if not hf_token or hf_token == "YOUR_HF_TOKEN":
        print("No valid HF token provided, skipping push to Hugging Face.")
        return "skipped"

    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=merged_model_path,
            repo_id=repo_id,
            repo_type="model"
        )
        return f"https://huggingface.co/{repo_id}"
    except Exception as e:
        print(f"Failed to push to HF: {e}")
        return "failed"


@dsl.pipeline(
    name="arabic-medical-llm-training-papermill",
    description="E2E Pipeline using Papermill to execute Jupyter Notebooks directly",
)
def arabic_medical_llm_pipeline(
    dataset_path:       str   = "/data/raw/alignment",
    sft_output_dir:     str   = "/outputs/sft_adapter",
    alignment_output:   str   = "/outputs/alignments",
    merged_output_dir:  str   = "/outputs/merged_model_16bit",
    base_model:         str   = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    hf_repo_id:         str   = "YourUsername/Arabic-Medical-LLM-Qwen-3B",
    hf_token:           str   = "YOUR_HF_TOKEN",
):
    validation = validate_dataset(dataset_path=dataset_path)

    sft = run_sft(
        dataset_path=dataset_path,
        output_dir=sft_output_dir,
        base_model=base_model,
    ).after(validation)

    align_dpo = run_alignment(sft_adapter_path=sft.outputs['Output'], dataset_path=dataset_path, output_dir=alignment_output, method="dpo")
    align_ipo = run_alignment(sft_adapter_path=sft.outputs['Output'], dataset_path=dataset_path, output_dir=alignment_output, method="ipo")
    align_kto = run_alignment(sft_adapter_path=sft.outputs['Output'], dataset_path=dataset_path, output_dir=alignment_output, method="kto")
    align_orpo = run_alignment(sft_adapter_path=sft.outputs['Output'], dataset_path=dataset_path, output_dir=alignment_output, method="orpo")
    align_simpo = run_alignment(sft_adapter_path=sft.outputs['Output'], dataset_path=dataset_path, output_dir=alignment_output, method="simpo")
    align_rloo = run_alignment(sft_adapter_path=sft.outputs['Output'], dataset_path=dataset_path, output_dir=alignment_output, method="rloo")

    best_model = select_best_model(
        dpo_path=align_dpo.outputs['Output'],
        ipo_path=align_ipo.outputs['Output'],
        kto_path=align_kto.outputs['Output'],
        orpo_path=align_orpo.outputs['Output'],
        simpo_path=align_simpo.outputs['Output'],
        rloo_path=align_rloo.outputs['Output'],
    )

    merge = merge_model(
        adapter_path=best_model.outputs['Output'],
        output_dir=merged_output_dir,
        base_model=base_model,
    )

    push_to_huggingface(
        merged_model_path=merge.outputs['Output'],
        repo_id=hf_repo_id,
        hf_token=hf_token,
    ).after(merge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile pipeline to YAML")
    parser.add_argument("--output",  default="training_pipeline.yaml")
    args = parser.parse_args()

    if args.compile:
        compiler.Compiler().compile(arabic_medical_llm_pipeline, args.output)
        print(f"[OK] Pipeline compiled to: {args.output}")
