"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import segmentation_models_pytorch as smp
import torch
from torch import nn

PathLike = Union[str, Path]


def build_unet(
    num_classes: int,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    """Build the U-Net architecture used by the project checkpoints."""

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        decoder_channels=(256, 128, 64, 32, 16),
    )


def _state_dict_from_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_segmentation_model(
    checkpoint_path: PathLike,
    num_classes: int,
    device: torch.device,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = None,
) -> nn.Module:
    """Load a segmentation checkpoint and return an eval-mode model."""

    model = build_unet(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
    model.to(device)
    model.eval()
    return model
