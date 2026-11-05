"""Upload SPARCLING-DE dataset and Dataset Card to the Hugging Face Hub.

Usage:
    # 1. Authenticate (or export HF_TOKEN=hf_...)
    huggingface-cli login

    # 2. Upload to your Hugging Face dataset repository
    python -m scripts.publish_hf --repo-id Matt00n/SPARCLING-DE
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload SPARCLING-DE to Hugging Face Hub."
    )
    ap.add_argument(
        "--repo-id",
        default="Matt00n/SPARCLING-DE",
        help="Target Hugging Face dataset repo ID (default: %(default)s)",
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("experiments/datasets/sparcling-de"),
        help="Local directory containing the SPARCLING-DE files",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private initially",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with:\n"
            "  poetry run pip install huggingface_hub"
        ) from exc

    if not args.dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")

    api = HfApi()
    print(f"Creating/verifying dataset repo: {args.repo_id} (private={args.private})...")
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repository ready: {repo_url}")

    print(f"Uploading files from {args.dataset_dir} ...")
    api.upload_folder(
        folder_path=str(args.dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload SPARCLING-DE dataset and dataset card",
    )
    print(f"\nDone! View your dataset at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

