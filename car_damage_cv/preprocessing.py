"""Image preprocessing shared by training and inference."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


def resize_and_pad(image: Image.Image, image_size: int = 512) -> Image.Image:
    """Resize an image preserving aspect ratio and pad it to a square."""

    image = ImageOps.contain(image.convert("RGB"), (image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def preprocess_image(image: Image.Image, image_size: Optional[int] = 512) -> torch.Tensor:
    """Convert an RGB image to a CHW float tensor in the 0..1 range."""

    if image_size is not None:
        image = resize_and_pad(image, image_size=image_size)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resize_mask(mask: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize a class-index mask without interpolation artifacts."""

    return mask.resize(size, Image.Resampling.NEAREST)
