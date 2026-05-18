"""Image and checkpoint I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union
from io import BytesIO
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image

PathLike = Union[str, Path]


def download_from_yadisk(short_url: str, filename: str, target_dir: PathLike) -> Path:
    """Download a public Yandex Disk file and return its local path."""

    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    response = requests.get(base_url + urlencode({"public_key": short_url}), timeout=30)
    response.raise_for_status()

    download_url = response.json()["href"]
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / filename

    file_response = requests.get(download_url, timeout=120)
    file_response.raise_for_status()
    target_file.write_bytes(file_response.content)
    return target_file


def load_image(source: PathLike) -> Image.Image:
    """Load an RGB image from a local path."""

    return Image.open(source).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    """Load an RGB image from a URL."""

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB uint8 array."""

    return np.asarray(image.convert("RGB"))
