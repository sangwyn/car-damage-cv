"""Backward-compatible imports for old notebooks.

New code should import from ``car_damage_cv.io`` and ``car_damage_cv.torch_utils``.
"""

from car_damage_cv.io import download_from_yadisk
from car_damage_cv.torch_utils import (
    CosineAnnealingRestartLR,
    LayerNorm2d,
    LayerNormFunction,
    evaluate_mse,
    get_position_from_periods,
    get_scheduler,
)

test_model = evaluate_mse
