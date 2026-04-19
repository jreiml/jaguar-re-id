"""E2: Backbone comparison (Q5).

Controlled comparison of 4 frozen backbones projected to a 256-d embedding
via the same projection head + ArcFace training recipe. All runs share:
  - identity-disjoint val split v1
  - ArcFace margin/scale, hidden dim, dropout, optimizer, schedule,
    batch size, embedding dim, num_epochs, seed
Only the backbone (and therefore the backbone feature dim) changes.

Runs are logged to W&B group `exp_E2_backbone`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from ..train import TrainConfig, train_baseline


BACKBONES = [
    ("mega-l384", "hf-hub:BVRA/MegaDescriptor-L-384", 384),
    ("convnextv2-large", "convnextv2_large.fcmae_ft_in22k_in1k_384", 384),
    ("dinov2-vitl14", "vit_large_patch14_reg4_dinov2.lvd142m", 518),
    ("efficientnetv2-l", "tf_efficientnetv2_l.in21k_ft_in1k", 384),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="run only a single backbone (by short name)")
    p.add_argument("--num-epochs", type=int, default=30)
    args = p.parse_args()

    results = {}
    for short, hub, size in BACKBONES:
        if args.only and args.only != short:
            continue
        cfg = TrainConfig(
            backbone=hub,
            input_size=size,
            num_epochs=args.num_epochs,
            run_name=f"E2-{short}",
            wandb_group="exp_E2_backbone",
        )
        out = train_baseline(cfg)
        results[short] = out

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
