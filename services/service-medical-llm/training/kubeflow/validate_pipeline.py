"""
validate_pipeline.py
====================
Validates the full SFT training pipeline WITHOUT running the actual training.

Checks:
  1.  Python environment & packages
  2.  Dataset file — exists, valid JSON, correct schema
  3.  Notebook — has 'parameters' cell, cells count, no syntax errors
  4.  Papermill — parameter injection only (--prepare-only)
  5.  Pipeline flow — kubeflow component function signature
  6.  Output directories — writable
  7.  MLflow — server reachable

Run:  python validate_pipeline.py
"""

import os, sys, json, subprocess, importlib
from pathlib import Path

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"

passed = []
failed = []
warned = []

def ok(msg):
    print(f"  {GREEN}✅ {msg}{RESET}")
    passed.append(msg)

def fail(msg):
    print(f"  {RED}❌ {msg}{RESET}")
    failed.append(msg)

def warn(msg):
    print(f"  {YELLOW}⚠️  {msg}{RESET}")
    warned.append(msg)

def section(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("E:/FineTuning/services/service-medical-llm/training")
DATASET_PATH = BASE / "experiments/data_pipeline/processed_data/chatml_dataset.json"
NOTEBOOK_IN  = BASE / "experiments/01_sft/train_sft_optimized.ipynb"
OUTPUT_DIR   = BASE / "outputs/test_sft"
NOTEBOOK_OUT = OUTPUT_DIR / "train_sft_dry_run.ipynb"
KUBEFLOW_DIR = BASE / "kubeflow"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Python environment & packages
# ══════════════════════════════════════════════════════════════════════════════
section("1. Python Environment & Packages")

print(f"  Python: {sys.version.split()[0]}  |  Env: {sys.prefix}")

required_packages = {
    "torch":        "PyTorch",
    "transformers": "Transformers",
    "datasets":     "Datasets",
    "trl":          "TRL",
    "peft":         "PEFT",
    "papermill":    "Papermill",
    "mlflow":       "MLflow",
    "ipykernel":    "ipykernel",
}

for pkg, label in required_packages.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        ok(f"{label} ({pkg}=={ver})")
    except ImportError:
        fail(f"{label} ({pkg}) NOT installed")

# tokenizers version check (unsloth needs <0.22)
try:
    import tokenizers
    tok_ver = tokenizers.__version__
    parts = [int(x) for x in tok_ver.split(".")[:2]]
    if parts[0] == 0 and parts[1] <= 21:
        ok(f"tokenizers=={tok_ver}  (compatible with unsloth)")
    else:
        fail(f"tokenizers=={tok_ver}  — unsloth needs tokenizers>=0.21,<0.22. "
             f"Run: pip install 'tokenizers==0.21.4' --no-deps")
except ImportError:
    fail("tokenizers NOT installed")

# unsloth check (optional but important)
try:
    import unsloth
    ok(f"unsloth=={unsloth.__version__}")
except ImportError:
    warn("unsloth not importable from this check script (may still work in notebook kernel)")
except Exception as e:
    warn(f"unsloth import warning: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════════════
section("2. Dataset Validation")

print(f"  Path: {DATASET_PATH}")

if not DATASET_PATH.exists():
    fail(f"Dataset file NOT found at: {DATASET_PATH}")
else:
    ok(f"Dataset file exists ({DATASET_PATH.stat().st_size / 1024:.1f} KB)")

    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            ok(f"Valid JSON — {len(data):,} records (list format)")
            sample = data[0] if data else {}
        elif isinstance(data, dict):
            # HuggingFace datasets format: {"train": [...]}
            first_key = next(iter(data))
            records = data[first_key]
            ok(f"Valid JSON — {len(records):,} records (dict format, key='{first_key}')")
            sample = records[0] if records else {}
        else:
            fail(f"Unexpected JSON root type: {type(data)}")
            sample = {}

        # Schema check
        if "messages" in sample:
            ok("Schema: 'messages' field found (ChatML format ✓)")
            msgs = sample["messages"]
            roles = {m.get("role") for m in msgs}
            if {"user", "assistant"}.issubset(roles):
                ok(f"Roles present: {roles}")
            else:
                warn(f"Expected 'user' and 'assistant' roles, found: {roles}")
        elif "text" in sample:
            ok("Schema: 'text' field found (plain text format)")
        elif "prompt" in sample and "completion" in sample:
            ok("Schema: 'prompt'+'completion' fields found")
        else:
            warn(f"Unknown schema — top-level keys: {list(sample.keys())[:8]}")

    except json.JSONDecodeError as e:
        fail(f"Invalid JSON: {e}")
    except Exception as e:
        fail(f"Error reading dataset: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Notebook Validation
# ══════════════════════════════════════════════════════════════════════════════
section("3. Notebook Validation")

print(f"  Path: {NOTEBOOK_IN}")

if not NOTEBOOK_IN.exists():
    fail(f"Notebook NOT found: {NOTEBOOK_IN}")
else:
    ok("Notebook file exists")

    try:
        with open(NOTEBOOK_IN, encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c["cell_type"] == "code"]
        ok(f"Valid JSON notebook — {len(cells)} cells ({len(code_cells)} code cells)")

        # Check for parameters cell
        param_cells = [
            c for c in cells
            if "parameters" in c.get("metadata", {}).get("tags", [])
        ]
        if param_cells:
            ok(f"'parameters' tagged cell found (Papermill ready ✓)")
            # Show what parameters are defined
            src = "".join(param_cells[0]["source"])
            param_vars = [
                line.split("=")[0].strip()
                for line in src.splitlines()
                if "=" in line and not line.strip().startswith("#")
            ]
            ok(f"Parameters defined: {param_vars}")
        else:
            fail("No 'parameters' tagged cell found — Papermill cannot inject parameters!")

        # Check Config class reads from parameters
        config_cells = [
            c for c in code_cells
            if "class Config:" in "".join(c["source"])
        ]
        if config_cells:
            src = "".join(config_cells[0]["source"])
            checks = {
                "DATA_PATH = DATASET_PATH": "Config.DATA_PATH reads from DATASET_PATH ✓",
                "OUTPUT_DIR":               "Config.OUTPUT_DIR reads from OUTPUT_DIR ✓",
                "BASE_MODEL":               "Config.MODEL_NAME reads from BASE_MODEL ✓",
            }
            for pattern, label in checks.items():
                if pattern in src:
                    ok(label)
                else:
                    warn(f"Config class might not read '{pattern}' from parameters")
        else:
            warn("Could not find 'class Config:' cell to validate")

    except json.JSONDecodeError as e:
        fail(f"Notebook is invalid JSON: {e}")
    except Exception as e:
        fail(f"Error reading notebook: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Papermill Dry-Run (--prepare-only)
# ══════════════════════════════════════════════════════════════════════════════
section("4. Papermill Dry-Run (Parameter Injection Only)")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "papermill",
    str(NOTEBOOK_IN), str(NOTEBOOK_OUT),
    "-p", "DATASET_PATH", str(DATASET_PATH),
    "-p", "OUTPUT_DIR",   str(OUTPUT_DIR),
    "-p", "BASE_MODEL",   "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
    "-p", "NUM_EPOCHS",   "5",
    "-p", "LEARNING_RATE","0.0002",
    "-p", "BATCH_SIZE",   "8",
    "-p", "GRAD_ACCUM",   "8",
    "--prepare-only",   # ← inject parameters but DON'T execute
]

print(f"  Running: papermill --prepare-only ...")
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        ok("Papermill --prepare-only succeeded (parameters injected)")
        # Verify injected notebook
        if NOTEBOOK_OUT.exists():
            with open(NOTEBOOK_OUT, encoding="utf-8") as f:
                nb_out = json.load(f)
            injected = [
                c for c in nb_out["cells"]
                if "injected-parameters" in c.get("metadata", {}).get("tags", [])
            ]
            if injected:
                ok("Injected parameters cell found in output notebook ✓")
                src = "".join(injected[0]["source"])
                print(f"\n  {CYAN}--- Injected Parameters ---{RESET}")
                for line in src.splitlines():
                    if line.strip() and not line.strip().startswith("#"):
                        print(f"    {line}")
            else:
                warn("No injected-parameters cell found in output — check notebook structure")
    else:
        stderr = result.stderr.strip()
        if "does not contain a cell with tag" in stderr:
            fail("Notebook still missing 'parameters' tagged cell")
        else:
            fail(f"Papermill failed:\n    {stderr[-500:]}")
except subprocess.TimeoutExpired:
    fail("Papermill timed out (>30s) even with --prepare-only")
except FileNotFoundError:
    fail("papermill not found — install with: pip install papermill")
except Exception as e:
    fail(f"Unexpected error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline Flow (Kubeflow component signature)
# ══════════════════════════════════════════════════════════════════════════════
section("5. Kubeflow Pipeline Flow Validation")

sys.path.insert(0, str(KUBEFLOW_DIR))
try:
    import pipeline as pl
    ok("pipeline.py imported successfully")

    import inspect
    sig = inspect.signature(pl.run_sft.python_func)
    ok(f"run_sft signature: {list(sig.parameters.keys())}")

    # Check WORKSPACE_PATH env
    ws = os.environ.get("WORKSPACE_PATH", "NOT SET")
    if ws == "NOT SET":
        warn(f"WORKSPACE_PATH env var not set — pipeline will use '/workspace' (OK for Docker, fails locally)")
        warn(f"  Fix: set WORKSPACE_PATH=E:\\FineTuning\\services\\service-medical-llm\\training")
    else:
        ok(f"WORKSPACE_PATH = {ws}")
        nb_path = Path(ws) / "experiments" / "01_sft" / "train_sft_optimized.ipynb"
        if nb_path.exists():
            ok(f"Notebook reachable via WORKSPACE_PATH ✓")
        else:
            fail(f"Notebook NOT found at WORKSPACE_PATH: {nb_path}")

except ImportError as e:
    fail(f"Cannot import pipeline.py: {e}")
except Exception as e:
    fail(f"Pipeline validation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Output Directories
# ══════════════════════════════════════════════════════════════════════════════
section("6. Output Directories Writable")

dirs_to_check = [
    BASE / "outputs/test_sft",
    BASE / "outputs/sft_adapter",
    BASE / "outputs/alignments",
]
for d in dirs_to_check:
    try:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        ok(f"Writable: {d}")
    except Exception as e:
        fail(f"Cannot write to {d}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MLflow Server
# ══════════════════════════════════════════════════════════════════════════════
section("7. MLflow Server")

try:
    import urllib.request
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    req = urllib.request.urlopen(mlflow_uri, timeout=3)
    ok(f"MLflow reachable at {mlflow_uri} (status {req.status})")
except Exception as e:
    warn(f"MLflow not reachable at localhost:5000 — {type(e).__name__}")
    warn("Training will still work but metrics won't be logged to MLflow UI")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")

total = len(passed) + len(failed) + len(warned)
print(f"\n  {GREEN}✅ Passed : {len(passed)}{RESET}")
print(f"  {YELLOW}⚠️  Warnings: {len(warned)}{RESET}")
print(f"  {RED}❌ Failed : {len(failed)}{RESET}")

if failed:
    print(f"\n{RED}{BOLD}  Pipeline NOT ready. Fix the errors above first.{RESET}")
    for f_ in failed:
        print(f"  {RED}  → {f_}{RESET}")
    sys.exit(1)
elif warned:
    print(f"\n{YELLOW}{BOLD}  Pipeline mostly ready. Review warnings above.{RESET}")
else:
    print(f"\n{GREEN}{BOLD}  🎉 Pipeline is fully validated and ready to run!{RESET}")
    print(f"\n  To run training:")
    print(f"  {CYAN}  set WORKSPACE_PATH=E:\\FineTuning\\services\\service-medical-llm\\training{RESET}")
    print(f"  {CYAN}  python test_sft.py{RESET}")
