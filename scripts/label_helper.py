#!/usr/bin/env python3
"""Tiny helper for creating staff and identity JSONL annotations.

The script appends one labeled record at a time and can also print sample
templates for the two supported annotation modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAFF_TEMPLATE = {
    "torso_match_ratio": 0.0,
    "is_clothing_staff": False,
    "zone_id": "ENTRY",
    "wx": 0.0,
    "billing_duration_sec": 0.0,
    "total_duration_sec": 0.0,
    "camera_count": 1,
    "label": 0,
}

REID_TEMPLATE = {
    "spatial_score": 0.0,
    "temporal_score": 0.0,
    "visual_score": 0.0,
    "camera_score": 0.0,
    "zone_score": 0.0,
    "dist_norm": 0.0,
    "time_norm": 0.0,
    "label": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny JSONL annotations for supervised training")
    parser.add_argument("--mode", choices=("staff", "identity"), required=True)
    parser.add_argument("--output", help="Path to the JSONL file to append to", default=None)
    parser.add_argument(
        "--sample-template",
        action="store_true",
        help="Print a sample template JSON object and exit",
    )
    return parser.parse_args()


def prompt_bool(label: str, default: bool) -> bool:
    raw = input(f"{label} [{str(default).lower()}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "t", "yes", "y"}


def prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else float(default)


def prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    return int(raw) if raw else int(default)


def prompt_str(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def build_record(mode: str) -> dict:
    if mode == "staff":
        return {
            "torso_match_ratio": prompt_float("torso_match_ratio", STAFF_TEMPLATE["torso_match_ratio"]),
            "is_clothing_staff": prompt_bool("is_clothing_staff", STAFF_TEMPLATE["is_clothing_staff"]),
            "zone_id": prompt_str("zone_id", STAFF_TEMPLATE["zone_id"]),
            "wx": prompt_float("wx", STAFF_TEMPLATE["wx"]),
            "billing_duration_sec": prompt_float("billing_duration_sec", STAFF_TEMPLATE["billing_duration_sec"]),
            "total_duration_sec": prompt_float("total_duration_sec", STAFF_TEMPLATE["total_duration_sec"]),
            "camera_count": prompt_int("camera_count", STAFF_TEMPLATE["camera_count"]),
            "label": prompt_int("label (1=staff, 0=customer)", STAFF_TEMPLATE["label"]),
        }

    return {
        "spatial_score": prompt_float("spatial_score", REID_TEMPLATE["spatial_score"]),
        "temporal_score": prompt_float("temporal_score", REID_TEMPLATE["temporal_score"]),
        "visual_score": prompt_float("visual_score", REID_TEMPLATE["visual_score"]),
        "camera_score": prompt_float("camera_score", REID_TEMPLATE["camera_score"]),
        "zone_score": prompt_float("zone_score", REID_TEMPLATE["zone_score"]),
        "dist_norm": prompt_float("dist_norm", REID_TEMPLATE["dist_norm"]),
        "time_norm": prompt_float("time_norm", REID_TEMPLATE["time_norm"]),
        "label": prompt_int("label (1=same person, 0=different people)", REID_TEMPLATE["label"]),
    }


def main() -> int:
    args = parse_args()

    if args.sample_template:
        template = STAFF_TEMPLATE if args.mode == "staff" else REID_TEMPLATE
        print(json.dumps(template, indent=2))
        return 0

    if not args.output:
        raise SystemExit("--output is required unless --sample-template is used")

    record = build_record(args.mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Appended 1 {args.mode} annotation to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
