#!/usr/bin/env python
"""Train a U-Net segmentation model for either parts or damage masks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF
from tqdm import tqdm

from car_damage_cv.models import build_unet
from car_damage_cv.preprocessing import resize_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SegmentationDataset(Dataset):
    """Image/mask pairs where masks contain integer class ids."""

    def __init__(self, images_dir: Path, masks_dir: Path, image_size: int, mask_ext: Optional[str] = None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.mask_ext = mask_ext
        self.images = sorted(
            path for path in self.images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = self._mask_path_for(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = resize_mask(mask, (self.image_size, self.image_size))

        image_tensor = TF.to_tensor(image)
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.long)
        return image_tensor, mask_tensor

    def _mask_path_for(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.images_dir)
        if self.mask_ext:
            mask_path = (self.masks_dir / relative).with_suffix(self.mask_ext)
            if mask_path.exists():
                return mask_path
        direct_path = self.masks_dir / relative
        if direct_path.exists():
            return direct_path
        for suffix in MASK_EXTENSIONS:
            candidate = (self.masks_dir / relative).with_suffix(suffix)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No mask found for {image_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--train-masks", required=True, type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-masks", type=Path)
    parser.add_argument("--mask-ext", help="Optional mask extension override, for example .png.")
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--val-split", default=0.15, type=float)
    return parser.parse_args()


def make_loaders(args: argparse.Namespace):
    train_dataset = SegmentationDataset(args.train_images, args.train_masks, args.image_size, args.mask_ext)

    if bool(args.val_images) != bool(args.val_masks):
        raise ValueError("--val-images and --val-masks must be provided together")

    if args.val_images and args.val_masks:
        val_dataset = SegmentationDataset(args.val_images, args.val_masks, args.image_size, args.mask_ext)
    elif args.val_split > 0 and len(train_dataset) > 1:
        val_size = max(1, int(len(train_dataset) * args.val_split))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def save_checkpoint(
    output: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: Optional[float],
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": {
                "num_classes": args.num_classes,
                "encoder": args.encoder,
                "image_size": args.image_size,
            },
        },
        output,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = make_loaders(args)
    encoder_weights = None if args.encoder_weights.lower() in {"none", "null", ""} else args.encoder_weights

    model = build_unet(
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device) if val_loader else None

        if val_loss is None:
            save_checkpoint(args.output, model, optimizer, epoch, None, args)
        elif best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output, model, optimizer, epoch, val_loss, args)

        val_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_text}")


if __name__ == "__main__":
    main()
