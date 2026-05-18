"""Backward-compatible severity grading helper.

New code should use ``car_damage_cv.DamageSeverityPredictor`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from car_damage_cv import DamageSeverityPredictor


@lru_cache(maxsize=4)
def _get_predictor(
    damage_checkpoint: str = "model_damage.pth",
    parts_checkpoint: str = "model_parts.pth",
    device: Optional[str] = None,
    image_size: int = 512,
) -> DamageSeverityPredictor:
    return DamageSeverityPredictor(
        damage_checkpoint=damage_checkpoint,
        parts_checkpoint=parts_checkpoint,
        device=device,
        image_size=image_size,
    )


def get_grade(
    img_url: str,
    damage_checkpoint: str = "model_damage.pth",
    parts_checkpoint: str = "model_parts.pth",
    device: Optional[str] = None,
    image_size: int = 512,
) -> Optional[int]:
    try:
        predictor = _get_predictor(damage_checkpoint, parts_checkpoint, device, image_size)
        return predictor.predict_url(img_url).grade
    except Exception as exc:
        print(f"Error processing image: {exc}")
        return None
