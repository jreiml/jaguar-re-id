"""E18: Can a smaller model beat MegaDescriptor? (Q18)

MegaDescriptor-L-384 = 195M params, val mAP 0.598 on val_v1. We train two
smaller backbones under the same projection+ArcFace recipe and measure:
  - val mAP on val_v1,
  - backbone parameter count,
  - total params (backbone + projection head).

Candidates:
  - DINOv2-ViT-B/14  (86M, 2.27× smaller than Mega)
  - ConvNeXtV2-Base   (87.7M, 2.22× smaller than Mega)

If either beats Mega, Q18 is satisfied. Cross-refs E2 for backbone-level
comparison at equivalent-or-larger sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..train import TrainConfig, train_baseline


CANDIDATES = [
    ("dinov2-vitb14", "vit_base_patch14_reg4_dinov2.lvd142m", 518),
    ("convnextv2-base", "convnextv2_base.fcmae_ft_in22k_in1k_384", 384),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--num-epochs", type=int, default=30)
    args = p.parse_args()

    results = {}
    for short, hub, size in CANDIDATES:
        if args.only and args.only != short:
            continue
        cfg = TrainConfig(
            backbone=hub,
            input_size=size,
            num_epochs=args.num_epochs,
            run_name=f"E18-{short}",
            wandb_group="exp_E18_efficient",
        )
        out = train_baseline(cfg)
        results[short] = out

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
