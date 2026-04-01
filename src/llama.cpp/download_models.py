import os
import sys
import subprocess
import shutil
from huggingface_hub import Repository, snapshot_download

MODELS = {
    # 名称: HF 仓库名
    # "llama3_2_1b": "meta-llama/Llama-3.2-1B",
    # "tinyllama_1_1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   
    "qwen2_5_1_7b": "Qwen/Qwen2.5-7B",                      
    # "phi_1_5": "microsoft/phi-1_5",
    # "opt_1_3b": "facebook/opt-1.3b",
    # "gptneo_1_3b": "EleutherAI/gpt-neo-1.3B",
    # "pythia_1_4b": "EleutherAI/pythia-1.4b",
    # 新增 GLM Edge 1.5B Chat 模型
    # "glm_edge_1_5b_chat": "zai-org/glm-edge-1.5b-chat",
}

def _ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"])
    # import after install
    return Repository, snapshot_download

Repository, snapshot_download = _ensure_hf_hub()

os.makedirs("models", exist_ok=True)

for name, repo_id in MODELS.items():
    dest = os.path.join("models", name)
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"Skipping {name}: destination already exists and is not empty -> {dest}")
        continue

    try:
        print(f"Cloning {repo_id} into {dest} ...")
        Repository(local_dir=dest, clone_from=repo_id)
        print(f"Cloned {name}")
        continue
    except Exception as e:
        print(f"Repository clone failed for {repo_id}: {e}. Falling back to snapshot_download.")

    try:
        cached = snapshot_download(repo_id=repo_id)
        # copy cached snapshot to destination
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(cached, dest)
        print(f"Downloaded {name} to {dest} (from cache)")
    except Exception as e2:
        print(f"Failed to download {repo_id}: {e2}")