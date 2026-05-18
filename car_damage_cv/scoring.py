"""Damage severity scoring from part and damage probability maps."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from car_damage_cv.constants import DAMAGE_WEIGHTS


def overlap_matrix(
    damage_probs: torch.Tensor,
    part_probs: torch.Tensor,
    damage_threshold: float = 0.5,
    part_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute damage-class coverage for every car-part class.

    Args:
        damage_probs: Tensor shaped ``(num_damage_classes, height, width)``.
        part_probs: Tensor shaped ``(num_part_classes, height, width)``.
    """

    damage_binary = damage_probs >= damage_threshold
    part_binary = part_probs >= part_threshold

    intersections = torch.einsum(
        "dhw,phw->dp",
        damage_binary.float(),
        part_binary.float(),
    )
    part_areas = part_binary.float().sum(dim=(1, 2)).clamp_min(1.0)
    return intersections / part_areas.unsqueeze(0)


def raw_damage_score(matrix: torch.Tensor, weights: np.ndarray = DAMAGE_WEIGHTS) -> float:
    """Return the weighted raw score used before grade clamping."""

    weight_tensor = torch.as_tensor(weights, dtype=matrix.dtype, device=matrix.device)
    return float((matrix * weight_tensor).sum().item() / 6.0)


def grade_from_score(score: float) -> int:
    """Map a raw score to the public 1..5 grade range."""

    return max(1, min(5, int(score)))


def score_overlap_matrix(matrix: torch.Tensor) -> Tuple[int, float]:
    """Return ``(grade, raw_score)`` for a damage/part overlap matrix."""

    score = raw_damage_score(matrix)
    return grade_from_score(score), score
