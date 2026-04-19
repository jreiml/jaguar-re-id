"""E13: Multi-seed stability of the best single model (Q22).

Re-trains the best Phase-2 config (DINOv2 + ArcFace on identity-disjoint
val_v1) across 5 random seeds. Measures mean ± std of val mAP and of the
best epoch. Seeds: 42 (already run as E6-arcface), 7, 1337, 2024, 9001.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..train_loss_comparison import LossRunConfig, train_one_loss
from ..paths import CHECKPOINTS, LOGS


SEEDS = [42, 7, 1337, 2024, 9001]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=30)
    args = p.parse_args()

    results = []
    for seed in SEEDS:
        run_name = f"E13-arcface-seed{seed}"
        ckpt_path = CHECKPOINTS / f"{run_name}.pth"
        if ckpt_path.exists():
            d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            results.append({"seed": seed, "best_val_map": float(d["val_map"]), "best_epoch": int(d["epoch"]), "skipped": True})
            continue
        # Seed 42 is E6-arcface; reuse if its file exists and skip retrain.
        if seed == 42:
            legacy = CHECKPOINTS / "E6-arcface.pth"
            if legacy.exists():
                d = torch.load(legacy, map_location="cpu", weights_only=False)
                results.append({"seed": seed, "best_val_map": float(d["val_map"]), "best_epoch": int(d["epoch"]), "skipped": True, "source": "E6-arcface.pth"})
                continue

        cfg = LossRunConfig(
            loss="arcface",
            backbone="vit_large_patch14_reg4_dinov2.lvd142m",
            input_size=518,
            num_epochs=args.num_epochs,
            seed=seed,
            run_name=run_name,
            wandb_group="exp_E13_multiseed",
        )
        out = train_one_loss(cfg)
        results.append({"seed": seed, **out, "skipped": False})

    maps = [r["best_val_map"] for r in results]
    summary = {
        "seeds": SEEDS,
        "per_seed": results,
        "mean_val_map": float(np.mean(maps)),
        "std_val_map": float(np.std(maps)),
        "min_val_map": float(np.min(maps)),
        "max_val_map": float(np.max(maps)),
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E13_multiseed.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
