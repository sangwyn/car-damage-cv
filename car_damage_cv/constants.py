"""Project labels and scoring constants."""

from __future__ import annotations

import numpy as np

DAMAGE_LABELS = [
    "Missing part",
    "Cracked",
    "Scratch",
    "Broken part",
    "None",
    "Corrosion",
    "Dent",
    "Flaking",
    "Paint chip",
]

PART_LABELS = [
    "Trunk",
    "Back-windshield",
    "Mirror_Headlight",
    "None",
    "Rocker-panel_Back-wheel",
    "Grille",
    "Back-door",
    "Front-wheel",
    "Windshield_Fender",
    "Tail-light_Quarter-panel",
    "License-plate_Front-bumper",
    "Hood",
    "Front-door",
]

DAMAGE_WEIGHTS = np.array(
    [
        [60, 90, 80, 0, 80, 40, 60, 80, 100, 80, 60, 70, 60],
        [48, 72, 64, 0, 64, 32, 48, 64, 80, 64, 48, 56, 48],
        [18, 27, 24, 0, 24, 12, 18, 24, 30, 24, 18, 21, 18],
        [54, 81, 72, 0, 72, 36, 54, 72, 90, 72, 54, 63, 54],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [30, 45, 40, 0, 40, 20, 30, 40, 50, 40, 30, 35, 30],
        [42, 63, 56, 0, 56, 28, 42, 56, 70, 56, 42, 49, 42],
        [36, 54, 48, 0, 48, 24, 36, 48, 60, 48, 36, 42, 36],
        [24, 36, 32, 0, 32, 16, 24, 32, 40, 32, 24, 28, 24],
    ],
    dtype=np.float32,
)

NUM_DAMAGE_CLASSES = len(DAMAGE_LABELS)
NUM_PART_CLASSES = len(PART_LABELS)
