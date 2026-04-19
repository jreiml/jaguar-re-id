from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bg_replace import BgMode
from .data import ImageEmbeddingDataset, default_preprocess


@torch.no_grad()
def extract_embeddings(
    backbone: torch.nn.Module,
    image_paths: Iterable[Path],
    *,
    input_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    desc: str = "embed",
    bg_mode: BgMode = BgMode.AS_IS,
) -> tuple[np.ndarray, list[str]]:
    preprocess = default_preprocess(input_size)
    ds = ImageEmbeddingDataset(image_paths, preprocess, bg_mode=bg_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    backbone.eval()
    embs = []
    fns = []
    for imgs, names in tqdm(dl, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        out = backbone(imgs).cpu().numpy()
        embs.append(out)
        fns.extend(names)
    return np.vstack(embs), fns


def save_embeddings(path: Path, embeddings: np.ndarray, filenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, filenames=np.array(filenames, dtype=object))


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=True)
    return z["embeddings"], z["filenames"].tolist()


def reorder_embeddings(embs: np.ndarray, filenames: list[str], target: list[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(filenames)}
    return np.stack([embs[idx[f]] for f in target], axis=0)
