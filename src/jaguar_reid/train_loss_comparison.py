"""E6: train projection head with different losses on frozen backbone features.

Uses cached frozen-backbone features (from train.py's embedding cache) so that
each loss variant shares identical input features. Only the loss head differs.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    IdentityDisjointSplit,
    assert_identity_disjoint,
    build_identity_disjoint_split,
    load_train_df,
    train_images_dir,
)
from .embed import load_embeddings, reorder_embeddings, save_embeddings, extract_embeddings
from .eval import identity_balanced_map
from .losses import CircleLossWithClassPrototypes, CosFaceLayer, SubCenterArcFaceLayer, triplet_semi_hard_loss
from .model import ArcFaceLayer, EmbeddingProjection, count_parameters, load_backbone
from .paths import CHECKPOINTS, EMB_CACHE, KAGGLE_R1, SPLITS


LOSS_NAMES = ["arcface", "cosface", "subcenter_arcface", "triplet", "circle"]


@dataclass
class LossRunConfig:
    loss: str
    backbone: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    input_size: int = 518
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    patience: int = 10
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    cosface_margin: float = 0.35
    subcenters: int = 3
    triplet_margin: float = 0.3
    circle_gamma: float = 64.0
    circle_margin: float = 0.25
    seed: int = 42
    split_version: str = "v1"
    run_name: str = ""
    wandb_project: str = "jaguar-reid-jreiml"
    wandb_group: str = "exp_E6_loss"


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _get_or_build_split(df: pd.DataFrame, cfg: LossRunConfig) -> IdentityDisjointSplit:
    path = SPLITS / f"val_{cfg.split_version}.json"
    if path.exists():
        split = IdentityDisjointSplit.load(path)
        assert_identity_disjoint(df, split)
        return split
    split = build_identity_disjoint_split(df, seed=cfg.seed, version=cfg.split_version)
    assert_identity_disjoint(df, split)
    split.save(path)
    return split


def _cache_features(backbone_name: str, filenames: list[str], cache_key: str, cfg: LossRunConfig) -> np.ndarray:
    path = EMB_CACHE / f"{cache_key}.npz"
    if path.exists():
        emb, fns = load_embeddings(path)
        if set(fns) == set(filenames):
            return reorder_embeddings(emb, fns, filenames)
    backbone, _ = load_backbone(backbone_name, device="cuda")
    from .data import iter_image_paths
    paths = iter_image_paths(filenames, train_images_dir(KAGGLE_R1))
    emb, fns = extract_embeddings(backbone, paths, input_size=cfg.input_size, batch_size=cfg.batch_size, num_workers=4, desc=cache_key)
    save_embeddings(path, emb, fns)
    del backbone; torch.cuda.empty_cache()
    return reorder_embeddings(emb, fns, filenames)


def _build_head(loss: str, embedding_dim: int, num_classes: int, cfg: LossRunConfig) -> nn.Module | None:
    if loss == "arcface":
        return ArcFaceLayer(embedding_dim, num_classes, margin=cfg.arcface_margin, scale=cfg.arcface_scale)
    if loss == "cosface":
        return CosFaceLayer(embedding_dim, num_classes, margin=cfg.cosface_margin, scale=cfg.arcface_scale)
    if loss == "subcenter_arcface":
        return SubCenterArcFaceLayer(embedding_dim, num_classes, k_subcenters=cfg.subcenters, margin=cfg.arcface_margin, scale=cfg.arcface_scale)
    if loss == "circle":
        return CircleLossWithClassPrototypes(embedding_dim, num_classes, gamma=cfg.circle_gamma, margin=cfg.circle_margin)
    if loss == "triplet":
        return None  # no classifier head
    raise ValueError(f"Unknown loss {loss}")


def train_one_loss(cfg: LossRunConfig) -> dict:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    _set_seed(cfg.seed)
    device = "cuda"
    df = load_train_df(KAGGLE_R1)
    split = _get_or_build_split(df, cfg)

    backbone_slug = cfg.backbone.replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    train_emb = _cache_features(cfg.backbone, split.train_filenames, f"{backbone_slug}_train_{cfg.split_version}", cfg)
    val_emb = _cache_features(cfg.backbone, split.val_filenames, f"{backbone_slug}_val_{cfg.split_version}", cfg)

    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    train_labels_str = [by_fn[f] for f in split.train_filenames]
    val_labels = np.asarray([by_fn[f] for f in split.val_filenames])
    id_to_class = {i: c for c, i in enumerate(sorted(set(train_labels_str)))}
    train_labels = np.array([id_to_class[l] for l in train_labels_str], dtype=np.int64)
    num_classes = len(id_to_class)

    projection = EmbeddingProjection(train_emb.shape[1], cfg.hidden_dim, cfg.embedding_dim, cfg.dropout).to(device)
    head = _build_head(cfg.loss, cfg.embedding_dim, num_classes, cfg)
    if head is not None:
        head = head.to(device)
    params = list(projection.parameters()) + (list(head.parameters()) if head is not None else [])
    num_params = sum(p.numel() for p in params)

    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    ce = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(train_emb).float(), torch.from_numpy(train_labels).long())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_emb_t = torch.from_numpy(val_emb).float().to(device)

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name or f"E6-{cfg.loss}",
        group=cfg.wandb_group,
        config={**asdict(cfg), "num_parameters": num_params, "num_classes": num_classes, "backbone_feature_dim": int(train_emb.shape[1])},
    )
    wandb.summary["num_parameters"] = num_params

    best_map = -1.0
    best_epoch = 0
    patience_ctr = 0
    checkpoint_path = CHECKPOINTS / f"{cfg.run_name or f'E6-{cfg.loss}'}.pth"

    for epoch in range(1, cfg.num_epochs + 1):
        projection.train()
        if head is not None: head.train()
        total_loss = 0.0
        total = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            emb = projection(x)
            if cfg.loss == "triplet":
                emb_n = F.normalize(emb, p=2, dim=1)
                loss = triplet_semi_hard_loss(emb_n, y, margin=cfg.triplet_margin)
            elif cfg.loss == "circle":
                loss = head(emb, y)  # circle head already returns a scalar loss
            else:
                logits = head(emb, y)
                loss = ce(logits, y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0); total += x.size(0)
        train_loss = total_loss / total

        projection.eval()
        with torch.no_grad():
            val_ft = F.normalize(projection(val_emb_t), p=2, dim=1).cpu().numpy()
        val_map = identity_balanced_map(val_ft, val_labels)
        scheduler.step(val_map)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_map": val_map, "learning_rate": current_lr})

        if val_map > best_map:
            best_map = float(val_map); best_epoch = epoch; patience_ctr = 0
            state = {
                "epoch": epoch,
                "projection_state_dict": projection.state_dict(),
                "head_state_dict": head.state_dict() if head is not None else None,
                "config": asdict(cfg),
                "id_to_class": id_to_class,
                "num_classes": num_classes,
                "val_map": best_map,
                "backbone_feature_dim": int(train_emb.shape[1]),
                "num_parameters": num_params,
            }
            torch.save(state, checkpoint_path)
        else:
            patience_ctr += 1
        print(f"[{cfg.run_name or cfg.loss}] epoch {epoch:03d} loss {train_loss:.4f} val_mAP {val_map:.4f} lr {current_lr:.2e} best {best_map:.4f}@{best_epoch} patience {patience_ctr}")
        if patience_ctr >= cfg.patience:
            break

    wandb.summary["best_val_map"] = best_map
    wandb.summary["best_epoch"] = best_epoch
    art = wandb.Artifact(name=cfg.run_name or f"E6-{cfg.loss}", type="model")
    art.add_file(str(checkpoint_path))
    wandb.log_artifact(art)
    wandb.finish()
    return {"loss": cfg.loss, "best_val_map": best_map, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=LOSS_NAMES + ["all"], default="all")
    p.add_argument("--backbone", default="vit_large_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--num-epochs", type=int, default=30)
    args = p.parse_args()

    losses = LOSS_NAMES if args.loss == "all" else [args.loss]
    results = []
    for loss_name in losses:
        cfg = LossRunConfig(
            loss=loss_name,
            backbone=args.backbone,
            input_size=args.input_size,
            num_epochs=args.num_epochs,
            run_name=f"E6-{loss_name}",
        )
        results.append(train_one_loss(cfg))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
