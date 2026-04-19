"""Generate a Kaggle submission CSV from a trained ArcFace checkpoint.

Pipeline:
  1. Compute backbone features for every unique test image (cached on disk).
  2. Project features through the fine-tuned head, L2-normalize.
  3. For each (query, gallery) pair in test.csv, output cosine similarity
     clipped to [0, 1].

Also exposes `validate_submission_format` which compares a produced CSV to
the downloaded sample_submission.csv row-by-row (columns + row count), used
as a Phase 0 dry-run probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    KAGGLE_R1,
    KAGGLE_R2,
    sample_submission_path,
    test_csv_path,
    test_images_dir,
)
from .embed import extract_embeddings, load_embeddings, reorder_embeddings, save_embeddings
from .model import ArcFaceModel, load_backbone
from .paths import EMB_CACHE, SUBMISSIONS


def validate_submission_format(submission_csv: Path, sample_csv: Path) -> None:
    sub = pd.read_csv(submission_csv)
    sample = pd.read_csv(sample_csv)
    if list(sub.columns) != list(sample.columns):
        raise AssertionError(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")
    if len(sub) != len(sample):
        raise AssertionError(f"Row count mismatch: {len(sub)} vs {len(sample)}")
    if (sub["row_id"].values != sample["row_id"].values).any():
        raise AssertionError("row_id ordering mismatch")
    if not np.isfinite(sub["similarity"].values).all():
        raise AssertionError("similarity contains NaN/Inf")
    sim = sub["similarity"].values
    if sim.min() < 0.0 or sim.max() > 1.0:
        raise AssertionError(f"similarity out of [0, 1]: min={sim.min()} max={sim.max()}")


def _get_or_cache_backbone_test_emb(
    backbone_name: str,
    round_dir: Path,
    *,
    input_size: int,
    batch_size: int,
    device: str,
    cache_key: str,
    num_workers: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Extract frozen backbone features for every unique image in test.csv."""
    pairs = pd.read_csv(test_csv_path(round_dir))
    unique = sorted(set(pairs["query_image"]).union(set(pairs["gallery_image"])))
    cache_path = EMB_CACHE / f"{cache_key}.npz"
    if cache_path.exists():
        emb, fns = load_embeddings(cache_path)
        if set(fns) == set(unique):
            return reorder_embeddings(emb, fns, unique), unique

    backbone, _ = load_backbone(backbone_name, device=device)
    paths = [test_images_dir(round_dir) / f for f in unique]
    emb, fns = extract_embeddings(
        backbone,
        paths,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        desc=f"embed:{cache_key}",
    )
    save_embeddings(cache_path, emb, fns)
    del backbone
    torch.cuda.empty_cache()
    return reorder_embeddings(emb, fns, unique), unique


def make_submission(
    checkpoint_path: Path,
    round_dir: Path,
    out_csv: Path,
    *,
    cache_suffix: str | None = None,
    input_size: int | None = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_name = cfg["backbone"]
    input_size = int(input_size or cfg["input_size"])
    round_tag = "r2" if "kaggle_r2" in str(round_dir) else "r1"
    backbone_slug = backbone_name.replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    cache_key = f"{backbone_slug}_test_{round_tag}" + (f"_{cache_suffix}" if cache_suffix else "")

    test_emb, unique = _get_or_cache_backbone_test_emb(
        backbone_name,
        round_dir,
        input_size=input_size,
        batch_size=int(cfg["batch_size"]),
        device=device,
        cache_key=cache_key,
    )

    model = ArcFaceModel(
        input_dim=int(ckpt["backbone_feature_dim"]),
        num_classes=int(ckpt["num_classes"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        margin=float(cfg["arcface_margin"]),
        scale=float(cfg["arcface_scale"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        proj = model.get_embeddings(torch.from_numpy(test_emb).float().to(device)).cpu().numpy()
    fn_to_emb = {fn: proj[i] for i, fn in enumerate(unique)}

    pairs = pd.read_csv(test_csv_path(round_dir))
    q = np.stack([fn_to_emb[f] for f in pairs["query_image"]], axis=0)
    g = np.stack([fn_to_emb[f] for f in pairs["gallery_image"]], axis=0)
    sim = (q * g).sum(axis=1)
    sim = np.clip(sim, 0.0, 1.0)

    out = pd.DataFrame({"row_id": pairs["row_id"].values, "similarity": sim})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    validate_submission_format(out_csv, sample_submission_path(round_dir))
    return out_csv


def make_sample_like_submission(round_dir: Path, out_csv: Path) -> Path:
    """Copy sample_submission.csv as an initial dry-run submission, used for
    format validation and to probe that the Kaggle submission pipeline works."""
    sample = pd.read_csv(sample_submission_path(round_dir))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_csv, index=False)
    validate_submission_format(out_csv, sample_submission_path(round_dir))
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--round", choices=["r1", "r2"], default="r2")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    round_dir = KAGGLE_R1 if args.round == "r1" else KAGGLE_R2
    path = make_submission(Path(args.checkpoint), round_dir, Path(args.out))
    print(json.dumps({"submission_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
