"""E11: Does deduplicating the training set affect re-ID performance?

Takes E3's near-duplicate detection and trains the Phase-2-winning recipe
(DINOv2 + ArcFace) on (a) the full training set and (b) a deduplicated
training set. Measures val mAP on val_v1. Complements E3 with the
downstream-effect answer Q14 asks for.

Deduplication rule: for each pHash-cluster of exact duplicates (Hamming=0),
keep the lowest-alphabetical filename and drop the rest. This is a
conservative rule — only truly identical camera-trap burst frames are
removed, and each identity keeps at least one representative per cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..data import IdentityDisjointSplit, load_train_df, train_images_dir
from ..paths import KAGGLE_R1, LOGS, SPLITS
from ..train_loss_comparison import LossRunConfig, train_one_loss


def _phash_bits(img: Image.Image) -> bytes:
    import imagehash
    return bytes(np.packbits(np.asarray(imagehash.phash(img).hash).flatten()).tolist())


def dedup_filenames(train_fns: list[str], imgs_dir: Path) -> tuple[list[str], int]:
    """Drop all-but-one of every exact-pHash cluster within the input list."""
    hashes: dict[bytes, list[str]] = {}
    for fn in tqdm(train_fns, desc="phash for dedup"):
        with Image.open(imgs_dir / fn) as img:
            h = _phash_bits(img)
        hashes.setdefault(h, []).append(fn)
    kept = sorted(min(v) for v in hashes.values())
    dropped = sum(len(v) - 1 for v in hashes.values())
    return kept, dropped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=30)
    args = p.parse_args()

    split = IdentityDisjointSplit.load(SPLITS / "val_v1.json")
    df = load_train_df(KAGGLE_R1)
    imgs_dir = train_images_dir(KAGGLE_R1)

    kept, dropped = dedup_filenames(split.train_filenames, imgs_dir)
    print(f"Original train size: {len(split.train_filenames)}")
    print(f"After dedup (exact pHash): {len(kept)} (dropped {dropped})")

    # Build a dedup split (same val, smaller train).
    dedup_split = IdentityDisjointSplit(
        train_filenames=kept,
        val_filenames=split.val_filenames,
        train_identities=split.train_identities,
        val_identities=split.val_identities,
        seed=split.seed,
        val_frac=split.val_frac,
        version="v1_dedup",
    )
    dedup_path = SPLITS / "val_v1_dedup.json"
    dedup_split.save(dedup_path)

    # Train: full vs dedup. "Full" checkpoint is re-used from E6-arcface.
    results = {"full_train_val_map": None, "dedup_train_val_map": None,
               "n_train_full": len(split.train_filenames), "n_train_dedup": len(kept),
               "dedup_dropped": dropped}

    # Re-train DINOv2 + ArcFace on the dedup split only.
    cfg = LossRunConfig(
        loss="arcface",
        backbone="vit_large_patch14_reg4_dinov2.lvd142m",
        input_size=518,
        num_epochs=args.num_epochs,
        run_name="E11-dedup-arcface",
        split_version="v1_dedup",
        wandb_group="exp_E11_dedup",
    )
    out = train_one_loss(cfg)
    results["dedup_train_val_map"] = out["best_val_map"]

    # Load pre-existing E6-arcface result for the full-train value.
    import torch
    ck = torch.load(Path("checkpoints/E6-arcface.pth"), map_location="cpu", weights_only=False)
    results["full_train_val_map"] = float(ck["val_map"])
    results["delta_vs_full"] = results["dedup_train_val_map"] - results["full_train_val_map"]

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E11_dedup.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
