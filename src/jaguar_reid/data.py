from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .bg_replace import BgMode, load_rgb
from .paths import KAGGLE_R1, KAGGLE_R2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_csv_path(round_dir: Path = KAGGLE_R1) -> Path:
    return round_dir / "train.csv"


def train_images_dir(round_dir: Path = KAGGLE_R1) -> Path:
    # R1 extracted layout: train/train/<filename>.png
    direct = round_dir / "train" / "train"
    if direct.exists():
        return direct
    return round_dir / "train"


def test_csv_path(round_dir: Path) -> Path:
    return round_dir / "test.csv"


def test_images_dir(round_dir: Path) -> Path:
    nested = round_dir / "test" / "test"
    if nested.exists():
        return nested
    return round_dir / "test"


def sample_submission_path(round_dir: Path) -> Path:
    return round_dir / "sample_submission.csv"


def load_train_df(round_dir: Path = KAGGLE_R1) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path(round_dir))
    if "filename" not in df.columns or "ground_truth" not in df.columns:
        raise RuntimeError(f"train.csv missing expected columns: {df.columns.tolist()}")
    return df


@dataclass(frozen=True)
class IdentityDisjointSplit:
    train_filenames: list[str]
    val_filenames: list[str]
    train_identities: list[str]
    val_identities: list[str]
    seed: int
    val_frac: float
    version: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "version": self.version,
            "seed": self.seed,
            "val_frac": self.val_frac,
            "train_identities": self.train_identities,
            "val_identities": self.val_identities,
            "train_filenames": self.train_filenames,
            "val_filenames": self.val_filenames,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "IdentityDisjointSplit":
        d = json.loads(Path(path).read_text())
        return cls(
            train_filenames=d["train_filenames"],
            val_filenames=d["val_filenames"],
            train_identities=d["train_identities"],
            val_identities=d["val_identities"],
            seed=d["seed"],
            val_frac=d["val_frac"],
            version=d["version"],
        )


def build_identity_disjoint_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    val_frac: float = 0.2,
    min_train_images_per_identity: int = 2,
    min_val_images_per_identity: int = 2,
    version: str = "v1",
) -> IdentityDisjointSplit:
    """Split identities (not images) between train and val.

    Val identities never appear in train. This is the correct protocol for
    re-identification evaluation: the val set simulates the closed-set
    gallery of unseen individuals.
    """
    counts = df.groupby("ground_truth").size().sort_values(ascending=False)
    # Keep only identities with enough images to be useful on either side.
    eligible_train_ids = counts[counts >= min_train_images_per_identity].index.tolist()
    eligible_val_ids = counts[counts >= min_val_images_per_identity].index.tolist()

    rng = random.Random(seed)

    # Deterministic shuffle on sorted identity list so split is reproducible
    # independent of pandas ordering.
    candidate_val_ids = sorted(set(eligible_val_ids))
    rng.shuffle(candidate_val_ids)

    n_val = max(1, int(round(len(candidate_val_ids) * val_frac)))
    val_identities = sorted(candidate_val_ids[:n_val])
    val_set = set(val_identities)
    train_identities = sorted(i for i in eligible_train_ids if i not in val_set)

    train_mask = df["ground_truth"].isin(train_identities)
    val_mask = df["ground_truth"].isin(val_identities)

    return IdentityDisjointSplit(
        train_filenames=df.loc[train_mask, "filename"].astype(str).tolist(),
        val_filenames=df.loc[val_mask, "filename"].astype(str).tolist(),
        train_identities=list(train_identities),
        val_identities=list(val_identities),
        seed=seed,
        val_frac=val_frac,
        version=version,
    )


def assert_identity_disjoint(df: pd.DataFrame, split: IdentityDisjointSplit) -> None:
    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    tr_ids = {by_fn[f] for f in split.train_filenames}
    va_ids = {by_fn[f] for f in split.val_filenames}
    overlap = tr_ids & va_ids
    if overlap:
        raise AssertionError(
            f"Leak: {len(overlap)} identities appear in both train and val: {sorted(list(overlap))[:5]}"
        )


def default_preprocess(input_size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageEmbeddingDataset(Dataset):
    """Returns (tensor, filename) for embedding extraction over a list of paths."""

    def __init__(self, image_paths: Iterable[Path], preprocess, bg_mode: BgMode = BgMode.AS_IS):
        self.paths = [Path(p) for p in image_paths]
        self.preprocess = preprocess
        self.bg_mode = bg_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = load_rgb(path, self.bg_mode)
        return self.preprocess(img), path.name


def iter_image_paths(filenames: Iterable[str], images_dir: Path) -> list[Path]:
    return [images_dir / f for f in filenames]
