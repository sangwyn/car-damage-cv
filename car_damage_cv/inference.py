"""Inference pipeline for car damage severity prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from car_damage_cv.constants import NUM_DAMAGE_CLASSES, NUM_PART_CLASSES
from car_damage_cv.io import load_image, load_image_from_url
from car_damage_cv.models import load_segmentation_model
from car_damage_cv.preprocessing import preprocess_image
from car_damage_cv.scoring import overlap_matrix, score_overlap_matrix

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SeverityResult:
    grade: int
    raw_score: float
    overlap: np.ndarray


class DamageSeverityPredictor:
    """Load damage/parts segmentation models and predict a severity grade."""

    def __init__(
        self,
        damage_checkpoint: PathLike = "model_damage.pth",
        parts_checkpoint: PathLike = "model_parts.pth",
        device: Optional[str] = None,
        image_size: Optional[int] = 512,
        damage_threshold: float = 0.5,
        part_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.damage_threshold = damage_threshold
        self.part_threshold = part_threshold

        self.damage_model = load_segmentation_model(
            checkpoint_path=damage_checkpoint,
            num_classes=NUM_DAMAGE_CLASSES,
            device=self.device,
        )
        self.parts_model = load_segmentation_model(
            checkpoint_path=parts_checkpoint,
            num_classes=NUM_PART_CLASSES,
            device=self.device,
        )

    @torch.inference_mode()
    def predict_image(self, image: Image.Image) -> SeverityResult:
        tensor = preprocess_image(image, image_size=self.image_size).unsqueeze(0).to(self.device)

        damage_probs = F.softmax(self.damage_model(tensor), dim=1)[0]
        part_probs = F.softmax(self.parts_model(tensor), dim=1)[0]
        matrix = overlap_matrix(
            damage_probs=damage_probs,
            part_probs=part_probs,
            damage_threshold=self.damage_threshold,
            part_threshold=self.part_threshold,
        )
        grade, score = score_overlap_matrix(matrix)
        return SeverityResult(
            grade=grade,
            raw_score=score,
            overlap=matrix.detach().cpu().numpy(),
        )

    def predict_path(self, image_path: PathLike) -> SeverityResult:
        return self.predict_image(load_image(image_path))

    def predict_url(self, image_url: str) -> SeverityResult:
        return self.predict_image(load_image_from_url(image_url))
