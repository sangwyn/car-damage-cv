#!/usr/bin/env python
"""Predict car damage severity for a local image or URL."""

from __future__ import annotations

import argparse
import json

from car_damage_cv import DamageSeverityPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a local image.")
    source.add_argument("--url", help="URL of an image to download and analyze.")
    parser.add_argument("--damage-checkpoint", default="model_damage.pth")
    parser.add_argument("--parts-checkpoint", default="model_parts.pth")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--image-size", type=int, default=512, help="Square model input size.")
    parser.add_argument("--damage-threshold", type=float, default=0.5)
    parser.add_argument("--part-threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DamageSeverityPredictor(
        damage_checkpoint=args.damage_checkpoint,
        parts_checkpoint=args.parts_checkpoint,
        device=args.device,
        image_size=args.image_size,
        damage_threshold=args.damage_threshold,
        part_threshold=args.part_threshold,
    )

    result = predictor.predict_url(args.url) if args.url else predictor.predict_path(args.image)
    payload = {
        "grade": result.grade,
        "raw_score": result.raw_score,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Damage grade: {result.grade}")
        print(f"Raw score: {result.raw_score:.4f}")


if __name__ == "__main__":
    main()
