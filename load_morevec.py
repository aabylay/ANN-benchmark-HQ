#!/usr/bin/env python3
"""Download the public MoReVec dataset folder into data/."""

from __future__ import annotations

import argparse
import subprocess
import sys
import gdown
from pathlib import Path


DATASET_URL = "https://drive.google.com/drive/folders/1AqAVI8ASROqrFCQdEMPB8RNzPwilijRp?usp=drive_link"
DEFAULT_OUTPUT_DIR = Path("data")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the MoReVec datasets folder into data/."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the Google Drive folder will be downloaded (default: data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = gdown.download_folder(
        url=DATASET_URL,
        output=str(output_dir),
        quiet=False,
        remaining_ok=True,
    )

    if not downloaded:
        print("Dataset download failed.", file=sys.stderr)
        return 1

    datasets_dir = output_dir / "datasets"
    if datasets_dir.exists():
        print(f"MoReVec datasets downloaded to: {datasets_dir}")
    else:
        print(f"Download finished under: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
