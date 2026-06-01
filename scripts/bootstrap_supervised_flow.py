#!/usr/bin/env python3
"""Bootstrap a reproducible supervised training flow.

This command creates data/labels/*.jsonl from the bundled sample templates if the
working files do not already exist, then trains both supervised artifacts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_LABELS_DIR = ROOT / "data" / "labels"
EXAMPLES_DIR = ROOT / "examples" / "labels"


def ensure_label_file(sample_name: str, target_name: str) -> Path:
    target = DATA_LABELS_DIR / target_name
    if not target.exists():
        source = EXAMPLES_DIR / sample_name
        if not source.exists():
            raise FileNotFoundError(f"Missing sample template: {source}")
        shutil.copyfile(source, target)
        print(f"Created {target} from {source}")
    else:
        print(f"Using existing {target}")
    return target


def main() -> int:
    DATA_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    staff_labels = ensure_label_file("staff_labels.sample.jsonl", "staff_labels.jsonl")
    reid_pairs = ensure_label_file("reid_pairs.sample.jsonl", "reid_pairs.jsonl")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_supervised_models.py"),
        "--staff-labels",
        str(staff_labels),
        "--reid-pairs",
        str(reid_pairs),
        "--output-dir",
        str(ROOT / "pipeline" / "model_state"),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
