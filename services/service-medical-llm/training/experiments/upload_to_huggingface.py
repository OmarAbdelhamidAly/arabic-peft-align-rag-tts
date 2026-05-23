"""
upload_to_huggingface.py
========================
رفع الموديل المدموج (merged_model_16bit) على HuggingFace Hub.

الاستخدام:
    python upload_to_huggingface.py --repo YourUsername/Arabic-Medical-LLM-Qwen-3B
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


def load_dotenv(dotenv_path: Path):
    """Load .env file manually."""
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


def main():
    parser = argparse.ArgumentParser(description="Upload merged model to HuggingFace")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., YourUsername/Arabic-Medical-LLM-Qwen-3B)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to merged model (default: experiments/merged_model_16bit)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (default: from HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    args = parser.parse_args()
    
    # Load .env
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    # Get token
    hf_token = args.token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found. Set it in .env or pass --token")
        return
    
    # Model path
    model_path = Path(args.model_path) if args.model_path else script_dir / "merged_model_16bit"
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return
    
    print(f"📦 Model path: {model_path}")
    print(f"🎯 Target repo: {args.repo}")
    
    # Login
    try:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # Upload
    api = HfApi(token=hf_token)
    
    try:
        # Create repo if not exists
        api.create_repo(
            repo_id=args.repo,
            exist_ok=True,
            private=args.private
        )
        print(f"✅ Repository ready: {args.repo}")
        
        # Upload folder
        print(f"📤 Uploading model to HuggingFace...")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo,
            repo_type="model"
        )
        
        print(f"✅ Upload successful!")
        print(f"🔗 Model URL: https://huggingface.co/{args.repo}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return


if __name__ == "__main__":
    main()
