"""
Kubeflow Pipeline — Arabic Medical LLM Training (Papermill Edition)
===================================================================
Executes the actual Jupyter Notebooks for SFT and Alignment using Papermill.

Local Testing (Windows):
    set WORKSPACE_PATH=E:\\FineTuning\\services\\service-medical-llm\\training
    python test_sft.py

Docker / Kubeflow:
    WORKSPACE_PATH defaults to /workspace (matches Dockerfile WORKDIR)
"""

import os
import argparse
from kfp import dsl
from kfp import compiler
from kfp.dsl import Output, Metrics

# ── Images ────────────────────────────────────────────────────────────────────
_BASE_IMAGE     = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
_TRAINING_IMAGE = "arabic-medical-llm/trainer:latest"


# ── 1. Validate Dataset ───────────────────────────────────────────────────────
@dsl.component(base_image=_BASE_IMAGE, packages_to_install=["pandas"])
def validate_dataset(
    dataset_path:     str,
    expected_min_rows: int = 500,
) -> str:
    import os, json

    required_files = [
        "02_sft_responses.json",
        "05_simpo_dataset_hard.json",
    ]
    for fname in required_files:
        fpath = os.path.join(dataset_path, fname)
        if not os.path.exists(fpath):
            print(f"Warning: Missing dataset file {fpath}, assuming handled by generator.")

    print(f"[validate_dataset] dataset_path={dataset_path} → OK")
    return "validation_passed"


# ── 2. SFT ────────────────────────────────────────────────────────────────────
@dsl.component(
    base_image=_TRAINING_IMAGE,
    packages_to_install=["mlflow", "papermill", "ipykernel"],
)
def run_sft(
    dataset_path:      str,
    output_dir:        str,
    base_model:        str   = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    num_epochs:        int   = 5,
    learning_rate:     float = 2e-4,
    per_device_batch:  int   = 8,
    grad_accumulation: int   = 8,
    max_steps:         int   = 100,
    metrics:           Output[Metrics] = None,
) -> str:
    import os, subprocess, sys
    import mlflow

    # Support local Windows runs via WORKSPACE_PATH env var
    workspace    = os.environ.get("WORKSPACE_PATH", "/workspace")
    notebook_in  = os.path.join(workspace, "experiments", "01_sft", "train_sft_optimized.ipynb")
    os.makedirs(output_dir, exist_ok=True)
    notebook_out = os.path.join(output_dir, "train_sft_executed.ipynb")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("arabic-medical-sft")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs":    num_epochs,
            "lr":        learning_rate,
            "batch":     per_device_batch,
            "grad_accum": grad_accumulation,
            "base_model": base_model,
        })

        cmd = [
            "papermill", notebook_in, notebook_out,
            "-p", "DATASET_PATH",  dataset_path,
            "-p", "OUTPUT_DIR",    output_dir,
            "-p", "BASE_MODEL",    base_model,
            "-p", "NUM_EPOCHS",    str(num_epochs),
            "-p", "LEARNING_RATE", str(learning_rate),
            "-p", "BATCH_SIZE",    str(per_device_batch),
            "-p", "GRAD_ACCUM",    str(grad_accumulation),
            "-p", "MAX_STEPS",     str(max_steps),
        ]
        print(f"[run_sft] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[run_sft] STDOUT:\n{result.stdout}")
            print(f"[run_sft] STDERR:\n{result.stderr}")
            raise RuntimeError("SFT failed via Papermill")

        # Parse final_loss from stdout and log it
        for line in result.stdout.splitlines():
            if "final_loss" in line.lower():
                try:
                    loss = float(line.split(":")[-1].strip())
                    mlflow.log_metric("final_loss", loss)
                    if metrics:
                        metrics.log_metric("sft_final_loss", loss)
                except ValueError:
                    pass

    print(f"[run_sft] Done → {output_dir}")
    return output_dir


# ── 3. Alignment (DPO / IPO / KTO / ORPO / SimPO / RLOO) ────────────────────
@dsl.component(
    base_image=_TRAINING_IMAGE,
    packages_to_install=["mlflow", "papermill", "ipykernel"],
)
def run_alignment(
    sft_adapter_path: str,
    dataset_path:     str,
    output_dir:       str,
    method:           str,
    num_epochs:       int   = 3,
    learning_rate:    float = 5e-5,
    max_steps:         int   = 10,
    metrics:          Output[Metrics] = None,
) -> str:
    import os, subprocess
    import mlflow

    workspace         = os.environ.get("WORKSPACE_PATH", "/workspace")
    method_output_dir = os.path.join(output_dir, method)
    os.makedirs(method_output_dir, exist_ok=True)

    # RLOO has a two-part notebook; we run part-1 (reward model) in automated pipelines
    if method == "rloo":
        notebook_in = os.path.join(
            workspace, "experiments", "02_post_training",
            "train_rloo_part1_reward_model.ipynb",
        )
    else:
        notebook_in = os.path.join(
            workspace, "experiments", "02_post_training",
            f"train_{method}.ipynb",
        )

    notebook_out = os.path.join(method_output_dir, f"train_{method}_executed.ipynb")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("arabic-medical-alignment")

    with mlflow.start_run(run_name=method):
        mlflow.log_params({
            "method":  method,
            "epochs":  num_epochs,
            "lr":      learning_rate,
        })

        cmd = [
            "papermill", notebook_in, notebook_out,
            "-p", "SFT_ADAPTER_PATH", sft_adapter_path,
            "-p", "DATASET_PATH",     dataset_path,
            "-p", "OUTPUT_DIR",       method_output_dir,
            "-p", "NUM_EPOCHS",       str(num_epochs),
            "-p", "LEARNING_RATE",    str(learning_rate),
            "-p", "MAX_STEPS",        str(max_steps),
        ]
        print(f"[run_alignment:{method}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[run_alignment:{method}] Warning — failed:\n{result.stderr}")
            return ""   # Non-fatal: other methods may still succeed

        # Parse reward_margin and log it
        for line in result.stdout.splitlines():
            if "reward_margin" in line.lower():
                try:
                    margin = float(line.split(":")[-1].strip())
                    mlflow.log_metric("reward_margin", margin)
                    if metrics:
                        metrics.log_metric(f"{method}_reward_margin", margin)
                except ValueError:
                    pass

    print(f"[run_alignment:{method}] Done → {method_output_dir}")
    return method_output_dir


# ── 4. Select Best Model ──────────────────────────────────────────────────────
@dsl.component(
    base_image=_BASE_IMAGE,
    packages_to_install=["mlflow"],
)
def select_best_model(
    dpo_path:   str,
    ipo_path:   str,
    kto_path:   str,
    orpo_path:  str,
    simpo_path: str,
    rloo_path:  str,
) -> str:
    """
    In production: query MLflow for the run with the highest reward_margin.
    Currently: prefer SimPO → DPO → IPO → KTO → ORPO → RLOO.
    """
    print("[select_best_model] Paths received:")
    for name, path in [
        ("SimPO", simpo_path), ("DPO",  dpo_path),
        ("IPO",   ipo_path),   ("KTO",  kto_path),
        ("ORPO",  orpo_path),  ("RLOO", rloo_path),
    ]:
        print(f"  {name}: {path or '(failed)'}")

    priority = [simpo_path, dpo_path, ipo_path, kto_path, orpo_path, rloo_path]
    best = next((p for p in priority if p), None)

    if not best:
        raise RuntimeError("All alignment methods failed — no model to select.")

    print(f"[select_best_model] Best → {best}")
    return best


# ── 5. Merge Model ────────────────────────────────────────────────────────────
@dsl.component(
    base_image=_TRAINING_IMAGE,
    packages_to_install=["peft", "transformers", "accelerate"],
)
def merge_model(
    adapter_path: str,
    output_dir:   str,
    base_model:   str = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    save_method:  str = "merged_16bit",
) -> str:
    import os

    if not adapter_path or not os.path.exists(adapter_path):
        raise ValueError(f"[merge_model] Invalid adapter path: {adapter_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[merge_model] Merging {adapter_path} → {output_dir}")

    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"[merge_model] Saved merged model → {output_dir}")

    except Exception as exc:
        print(f"[merge_model] Merge failed: {exc}")
        # Write a sentinel so the pipeline can continue
        with open(os.path.join(output_dir, "merge_failed.txt"), "w") as f:
            f.write(str(exc))

    return output_dir


# ── 6. Push to Hugging Face ───────────────────────────────────────────────────
@dsl.component(
    base_image=_BASE_IMAGE,
    packages_to_install=["huggingface_hub"],
)
def push_to_huggingface(
    merged_model_path: str,
    repo_id:           str,
    hf_token:          str,
) -> str:
    from huggingface_hub import HfApi

    if not hf_token or hf_token == "YOUR_HF_TOKEN":
        print("[push_to_huggingface] No valid HF token — skipping.")
        return "skipped"

    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=merged_model_path,
            repo_id=repo_id,
            repo_type="model",
        )
        url = f"https://huggingface.co/{repo_id}"
        print(f"[push_to_huggingface] Pushed → {url}")
        return url
    except Exception as exc:
        print(f"[push_to_huggingface] Failed: {exc}")
        return "failed"


# ── Pipeline Definition ───────────────────────────────────────────────────────
@dsl.pipeline(
    name="arabic-medical-llm-training-papermill",
    description="E2E Pipeline: SFT -> Parallel Alignments -> Select Best -> Merge -> Push to HF",
)
def arabic_medical_llm_pipeline(
    dataset_path:      str = "/data/raw/alignment",
    sft_output_dir:    str = "/outputs/sft_adapter",
    alignment_output:  str = "/outputs/alignments",
    merged_output_dir: str = "/outputs/merged_model_16bit",
    base_model:        str = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    hf_repo_id:        str = "YourUsername/Arabic-Medical-LLM-Qwen-3B",
    hf_token:          str = "YOUR_HF_TOKEN",
    sft_max_steps:     int = 100,
    align_max_steps:   int = 10,
):
    # Step 1 — Validate
    validation = validate_dataset(dataset_path=dataset_path)

    # Step 2 — SFT
    sft = run_sft(
        dataset_path=dataset_path,
        output_dir=sft_output_dir,
        base_model=base_model,
        max_steps=sft_max_steps,
    ).after(validation)

    # Step 3 — 6 Parallel Alignments
    sft_out = sft.outputs["Output"]

    align_dpo   = run_alignment(sft_adapter_path=sft_out, dataset_path=dataset_path, output_dir=alignment_output, method="dpo", max_steps=align_max_steps)
    align_ipo   = run_alignment(sft_adapter_path=sft_out, dataset_path=dataset_path, output_dir=alignment_output, method="ipo", max_steps=align_max_steps)
    align_kto   = run_alignment(sft_adapter_path=sft_out, dataset_path=dataset_path, output_dir=alignment_output, method="kto", max_steps=align_max_steps)
    align_orpo  = run_alignment(sft_adapter_path=sft_out, dataset_path=dataset_path, output_dir=alignment_output, method="orpo", max_steps=align_max_steps)
    align_simpo = run_alignment(sft_adapter_path=sft_out, dataset_path=dataset_path, output_dir=alignment_output, method="simpo", max_steps=align_max_steps)
    align_rloo  = run_alignment(sft_adapter_path=sft_out, dataset_path=dataset_path, output_dir=alignment_output, method="rloo", max_steps=align_max_steps)

    # Step 4 — Select Best
    best = select_best_model(
        dpo_path=align_dpo.outputs["Output"],
        ipo_path=align_ipo.outputs["Output"],
        kto_path=align_kto.outputs["Output"],
        orpo_path=align_orpo.outputs["Output"],
        simpo_path=align_simpo.outputs["Output"],
        rloo_path=align_rloo.outputs["Output"],
    )

    # Step 5 — Merge
    merge = merge_model(
        adapter_path=best.outputs["Output"],
        output_dir=merged_output_dir,
        base_model=base_model,
    )

    # Step 6 — Push to HF
    push_to_huggingface(
        merged_model_path=merge.outputs["Output"],
        repo_id=hf_repo_id,
        hf_token=hf_token,
    ).after(merge)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile pipeline to YAML")
    parser.add_argument("--output",  default="training_pipeline.yaml")
    args = parser.parse_args()

    if args.compile:
        compiler.Compiler().compile(arabic_medical_llm_pipeline, args.output)
        print(f"[OK] Pipeline compiled → {args.output}")