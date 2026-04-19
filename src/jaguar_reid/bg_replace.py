"""Background-replacement image loaders.

The Kaggle PNGs are RGBA; alpha encodes a jaguar-vs-background mask. For R2
(and for Q0/Q26 interventions) we want to replace the background region with
something other than black. This module provides a pluggable loader.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class BgMode(str, Enum):
    AS_IS = "as_is"  # drop alpha, keep RGB unchanged (=default PIL convert)
    BLACK = "black"  # replace alpha==0 pixels with black 0
    GRAY = "gray"  # replace alpha==0 pixels with gray 128
    IMAGENET_MEAN = "imagenet_mean"  # replace with ImageNet mean color (~124, 116, 104)
    BLUR = "blur"  # replace with heavily-blurred copy of the image (Q1 candidate)
    RANDOM_NOISE = "random_noise"  # replace with U(0,255) noise


IMAGENET_MEAN_U8 = np.array([124, 116, 104], dtype=np.uint8)


def load_rgb(path: Path, mode: BgMode = BgMode.AS_IS, *, blur_radius: int = 25) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGBA":
        return img.convert("RGB")
    arr = np.asarray(img)
    rgb = arr[..., :3].copy()
    alpha = arr[..., 3]
    if mode == BgMode.AS_IS:
        return Image.fromarray(rgb)
    bg = alpha == 0
    if mode == BgMode.BLACK:
        rgb[bg] = 0
    elif mode == BgMode.GRAY:
        rgb[bg] = 128
    elif mode == BgMode.IMAGENET_MEAN:
        rgb[bg] = IMAGENET_MEAN_U8
    elif mode == BgMode.BLUR:
        from PIL import ImageFilter
        blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(blur_radius)))
        rgb[bg] = blurred[bg]
    elif mode == BgMode.RANDOM_NOISE:
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, size=rgb.shape, dtype=np.uint8)
        rgb[bg] = noise[bg]
    else:
        raise ValueError(f"Unknown BgMode: {mode}")
    return Image.fromarray(rgb)
