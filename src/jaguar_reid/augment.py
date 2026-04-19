"""Augmentation pipelines for training and for background intervention.

Train: standard ReID augmentations (flip, jitter, small rotation, erasing).
Background intervention (Q26): replace non-jaguar pixels with gray. Mask comes
from the HF `segmented_body` split when available, otherwise from an alpha
channel carried by the training image itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD


def training_transforms(input_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def eval_transforms(input_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def apply_gray_background(pil_image: Image.Image, mask: np.ndarray, gray_value: int = 128) -> Image.Image:
    """Replace all pixels where mask==0 with a constant gray value.

    mask: np.uint8 array matching image HxW; >0 = jaguar, 0 = background.
    """
    arr = np.array(pil_image.convert("RGB"))
    if mask.shape[:2] != arr.shape[:2]:
        mask_img = Image.fromarray(mask).resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)
    fg = mask > 0
    out = arr.copy()
    out[~fg] = gray_value
    return Image.fromarray(out)
