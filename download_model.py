import argparse
from cache_config import HF_HUB_CACHE_PATH, apply_cache_config
from huggingface_hub import snapshot_download

apply_cache_config()

parser = argparse.ArgumentParser(description="Download model from HuggingFace Hub")
parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-8B", help="Model repo id (e.g., Qwen/Qwen3-8B)")
parser.add_argument("--local_dir", type=str, default=None, help="Local directory to save the model")
args = parser.parse_args()

repo_id = args.repo_id
local_dir = args.local_dir or f"./base_model/{repo_id.split('/')[-1]}"

snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", cache_dir=HF_HUB_CACHE_PATH)
print(f"Downloaded {repo_id} to {local_dir}")
