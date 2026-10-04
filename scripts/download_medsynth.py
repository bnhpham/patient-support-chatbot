"""
Download the MedSynth dataset from Hugging Face (https://huggingface.co/datasets/Ahmad0067/MedSynth) into data/raw/medsynth.jsonl.

Login using e.g. `huggingface-cli login` to access this dataset.

Skip re-downloading if the output file already exists, unless --force is passed.

Usage:
    python scripts/download_medsynth.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_app_config  # noqa: E402

DATASET_NAME = "Ahmad0067/MedSynth"


def main() -> None:
    
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if output exists")
    args = parser.parse_args()

    # Config
    app_config = get_app_config()
    out_path = app_config.data_dir / "raw" / "medsynth.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"{out_path} already exists, skipping (use --force to re-download).")
        return

    # Download
    print(f"Downloading {DATASET_NAME} from Hugging Face...")
    dataset = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    split = dataset[split_name]

    with open(out_path, "w", encoding="utf-8") as f:
        for row in split:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(split)} rows to {out_path}")


if __name__ == "__main__":
    main()
