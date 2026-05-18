"""Backward-compatible metric helpers.

New code should import from ``car_damage_cv.scoring``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from car_damage_cv.constants import DAMAGE_LABELS as dmgs
from car_damage_cv.constants import PART_LABELS as parts
from car_damage_cv.scoring import overlap_matrix, score_overlap_matrix


def _as_channel_first(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"Expected mask tensor with 3 or 4 dimensions, got {tuple(mask.shape)}")
    return mask


def get_metric(part_mask: torch.Tensor, damage_mask: torch.Tensor):
    """Return ``(raw_score, grade)`` for part and damage model outputs."""

    part_mask = _as_channel_first(part_mask)
    damage_mask = _as_channel_first(damage_mask)

    part_probs = F.softmax(part_mask.float(), dim=0)
    damage_probs = F.softmax(damage_mask.float(), dim=0)
    matrix = overlap_matrix(damage_probs=damage_probs, part_probs=part_probs)
    grade, raw_score = score_overlap_matrix(matrix)
    return raw_score, grade


def get_descr(part_mask: torch.Tensor, damage_mask: torch.Tensor, x: int, y: int) -> str:
    """Describe active part/damage labels at a pixel coordinate."""

    part_mask = _as_channel_first(part_mask)
    damage_mask = _as_channel_first(damage_mask)

    found_parts = [part for idx, part in enumerate(parts) if part_mask[idx, y, x] > 0]
    found_damages = [dmg for idx, dmg in enumerate(dmgs) if damage_mask[idx, y, x] > 0]
    return f"Defects: {found_damages} | Parts: {found_parts}"
