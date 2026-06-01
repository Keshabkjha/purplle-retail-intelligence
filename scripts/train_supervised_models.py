#!/usr/bin/env python3
"""Train supervised staff and identity models from labeled JSONL files.

Expected JSONL schemas:

Staff samples:
{"torso_match_ratio": 0.92, "is_clothing_staff": true, "zone_id": "BILLING",
 "wx": 860.0, "billing_duration_sec": 150.0, "total_duration_sec": 260.0,
 "camera_count": 2, "label": 1}

Identity samples:
{"spatial_score": 0.96, "temporal_score": 0.92, "visual_score": 0.85,
 "camera_score": 0.95, "zone_score": 0.90, "dist_norm": 0.04,
 "time_norm": 0.08, "label": 1}

Both formats can also provide a precomputed "features" list instead of raw fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.adaptive_models import (
    train_identity_model_from_jsonl,
    train_staff_model_from_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised staff and re-ID models")
    parser.add_argument("--staff-labels", help="Path to labeled staff JSONL file", default=None)
    parser.add_argument("--reid-pairs", help="Path to labeled identity JSONL file", default=None)
    parser.add_argument(
        "--output-dir",
        help="Directory where trained model artifacts will be written",
        default="pipeline/model_state",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.staff_labels and not args.reid_pairs:
        raise SystemExit("Provide at least one of --staff-labels or --reid-pairs")

    if args.staff_labels:
        staff_out = output_dir / "staff_supervised.pkl"
        train_staff_model_from_jsonl(args.staff_labels, str(staff_out))
        print(f"Saved supervised staff model to {staff_out}")

    if args.reid_pairs:
        reid_out = output_dir / "identity_supervised.pkl"
        train_identity_model_from_jsonl(args.reid_pairs, str(reid_out))
        print(f"Saved supervised identity model to {reid_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
