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
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1337, 2024])
    args = p.parse_args()

    split = IdentityDisjointSplit.load(SPLITS / "val_v1.json")
    df = load_train_df(KAGGLE_R1)
    imgs_dir = train_images_dir(KAGGLE_R1)

    kept, dropped = dedup_filenames(split.train_filenames, imgs_dir)
    print(f"Original train size: {len(split.train_filenames)}")
    print(f"After dedup (exact pHash): {len(kept)} (dropped {dropped})")

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

    # Run multiple seeds of dedup training to get a std on the measured Δ.
    import torch
    import numpy as np
    dedup_maps: list[float] = []
    for seed in args.seeds:
        run_name = f"E11-dedup-arcface-seed{seed}"
        ckpt_path = Path(f"checkpoints/{run_name}.pth")
        if ckpt_path.exists():
            d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            dedup_maps.append(float(d["val_map"]))
            print(f"[reuse] seed={seed}: {d['val_map']:.4f}")
            continue
        # Seed 42 may be reusable from the original single-seed run.
        if seed == 42 and Path("checkpoints/E11-dedup-arcface.pth").exists():
            d = torch.load("checkpoints/E11-dedup-arcface.pth", map_location="cpu", weights_only=False)
            dedup_maps.append(float(d["val_map"]))
            print(f"[reuse] seed=42 from legacy: {d['val_map']:.4f}")
            continue
        cfg = LossRunConfig(
            loss="arcface",
            backbone="vit_large_patch14_reg4_dinov2.lvd142m",
            input_size=518,
            num_epochs=args.num_epochs,
            seed=seed,
            run_name=run_name,
            split_version="v1_dedup",
            wandb_group="exp_E11_dedup",
        )
        out = train_one_loss(cfg)
        dedup_maps.append(float(out["best_val_map"]))

    # Compare against E13 multi-seed baseline (full train).
    full_json = Path("logs/exp_E13_multiseed.json")
    if full_json.exists():
        e13 = json.loads(full_json.read_text())
        full_maps = [r["best_val_map"] for r in e13["per_seed"] if r["seed"] in args.seeds]
    else:
        # Fallback: pull from individual checkpoints.
        full_maps = []
        for seed in args.seeds:
            fp = Path(f"checkpoints/E13-arcface-seed{seed}.pth")
            if seed == 42: fp = Path("checkpoints/E6-arcface.pth")
            d = torch.load(fp, map_location="cpu", weights_only=False)
            full_maps.append(float(d["val_map"]))

    results = {
        "seeds": args.seeds,
        "dedup_val_maps": dedup_maps,
        "full_val_maps": full_maps,
        "n_train_full": len(split.train_filenames),
        "n_train_dedup": len(kept),
        "dedup_dropped": dropped,
        "dedup_mean": float(np.mean(dedup_maps)),
        "dedup_std": float(np.std(dedup_maps)),
        "full_mean": float(np.mean(full_maps)),
        "full_std": float(np.std(full_maps)),
        "delta_mean": float(np.mean(dedup_maps) - np.mean(full_maps)),
        "delta_std_pooled": float(np.sqrt(np.var(dedup_maps) + np.var(full_maps))),
    }

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E11_dedup.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
