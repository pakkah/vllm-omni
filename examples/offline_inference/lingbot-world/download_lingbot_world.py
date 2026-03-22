# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import fcntl
import json
import os
import site
import subprocess
import tempfile
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


DEFAULT_MODEL_ID = "robbyant/lingbot-world-base-cam"
DEFAULT_OUTPUT_DIR = "./lingbot-world-base-cam"
DEFAULT_CLASS_NAME = "LingbotWorldPipeline"
DEPENDENCY_REPO = "https://github.com/robbyant/lingbot-world.git"
DEPENDENCY_BRANCH = "main"
CACHE_DIR = Path(tempfile.gettempdir()) / "vllm-omni-dependency"
LOCK_FILE = CACHE_DIR / ".lingbot_world_install.lock"
DEPENDENCY_DIR = CACHE_DIR / "lingbot-world"
PTH_FILE_NAME = "vllm_omni_lingbot_world_dependency.pth"

REQUIRED_FILES = (
    "configuration.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "low_noise_model/config.json",
    "high_noise_model/config.json",
)


def infer_control_type(model_ref: str) -> str:
    model_ref = model_ref.lower()
    if "act" in model_ref:
        return "act"
    return "cam"


def ensure_model_index(
    output_dir: Path,
    *,
    class_name: str = DEFAULT_CLASS_NAME,
    control_type: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_index_path = output_dir / "model_index.json"
    payload = {
        "_class_name": class_name,
        "control_type": control_type or infer_control_type(str(output_dir)),
    }
    model_index_path.write_text(json.dumps(payload, indent=2) + "\n")
    return model_index_path


def validate_model_directory(output_dir: Path) -> None:
    missing = [rel_path for rel_path in REQUIRED_FILES if not (output_dir / rel_path).exists()]
    if missing:
        raise FileNotFoundError("LingBot-World download is incomplete. Missing files: " + ", ".join(sorted(missing)))


def download_dependency() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not DEPENDENCY_DIR.exists():
            print(f"Downloading LingBot-World to {DEPENDENCY_DIR} ...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    DEPENDENCY_REPO,
                    "--branch",
                    DEPENDENCY_BRANCH,
                    str(DEPENDENCY_DIR),
                ],
                check=True,
            )
            print("Download finished.")
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    site_packages = Path(site.getsitepackages()[0])
    pth_file = site_packages / PTH_FILE_NAME
    pth_file.write_text(f"{DEPENDENCY_DIR}\n", encoding="utf-8")
    print(f"Added {DEPENDENCY_DIR} to site-packages via {pth_file}")
    return pth_file


def timed_download(repo_id: str, local_dir: str) -> None:
    if os.path.exists(local_dir):
        print(f"Directory {local_dir} already exists. Skipping download.")
        return
    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is required to download LingBot-World. Install it before running this script."
        )
    print(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    elapsed = time.time() - start_time
    print(f"Finished downloading {repo_id} in {elapsed:.2f} seconds. Files saved at: {local_dir}")


def download_lingbot_world(model_id: str, output_dir: Path) -> Path:
    timed_download(repo_id=model_id, local_dir=str(output_dir))
    ensure_model_index(output_dir, control_type=infer_control_type(model_id))
    validate_model_directory(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LingBot-World from Hugging Face.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID to download.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for the prepared model.",
    )
    return parser.parse_args()


def main(output_dir: str, model_id: str = DEFAULT_MODEL_ID) -> None:
    model_dir = download_lingbot_world(model_id, Path(output_dir).expanduser().resolve())
    download_dependency()
    print(f"Prepared LingBot-World model at: {model_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir, args.model_id)
