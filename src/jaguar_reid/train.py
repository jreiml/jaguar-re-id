"""Baseline training: frozen MegaDescriptor backbone + ArcFace projection head.

Embeddings are pre-extracted from the frozen backbone once per image, then the
projection head + ArcFace layer is trained on the cached embeddings. This is
the same recipe as the Kaggle baseline notebook but restructured as a module
and trained against a fixed identity-disjoint validation split.

The key protocol changes from the Kaggle baseline are:
- Val split is identity-disjoint (val identities never appear in train). This
  is the correct closed-set re-identification protocol. It also means val mAP
  is a realistic proxy for Kaggle test mAP, rather than a near-perfect number.
- ArcFace classifier covers only train identities (val identities are
  unlabelled at evaluation, which matches the retrieval setting).
- num_parameters is logged to W&B on every run, per assessment requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    IdentityDisjointSplit,
    assert_identity_disjoint,
    build_identity_disjoint_split,
    iter_image_paths,
    load_train_df,
    train_images_dir,
)
from .embed import extract_embeddings, load_embeddings, reorder_embeddings, save_embeddings
from .eval import identity_balanced_map
from .model import ArcFaceModel, count_parameters, load_backbone
from .paths import CHECKPOINTS, EMB_CACHE, KAGGLE_R1, SPLITS


@dataclass
class TrainConfig:
    # Backbone / projection head
    backbone: str = "hf-hub:BVRA/MegaDescriptor-L-384"
    input_size: int = 384
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3

    # ArcFace
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0

    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 10

    # Misc
    seed: int = 42
    val_frac: float = 0.2
    split_version: str = "v1"
    run_name: str = "baseline-megadescriptor-arcface"
    wandb_project: str = "jaguar-reid-jreiml"
    wandb_group: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_or_build_split(df: pd.DataFrame, cfg: TrainConfig) -> IdentityDisjointSplit:
    path = SPLITS / f"val_{cfg.split_version}.json"
    if path.exists():
        split = IdentityDisjointSplit.load(path)
        if split.version != cfg.split_version:
            raise RuntimeError(f"Split version mismatch at {path}: {split.version} vs {cfg.split_version}")
        assert_identity_disjoint(df, split)
        return split
    split = build_identity_disjoint_split(
        df, seed=cfg.seed, val_frac=cfg.val_frac, version=cfg.split_version
    )
    assert_identity_disjoint(df, split)
    split.save(path)
    return split


def get_or_cache_backbone_embeddings(
    backbone_name: str,
    filenames: list[str],
    images_dir: Path,
    *,
    input_size: int,
    batch_size: int,
    device: str,
    cache_key: str,
    num_workers: int = 4,
) -> np.ndarray:
    cache_path = EMB_CACHE / f"{cache_key}.npz"
    if cache_path.exists():
        emb, cached = load_embeddings(cache_path)
        if set(cached) == set(filenames):
            return reorder_embeddings(emb, cached, filenames)

    backbone, _ = load_backbone(backbone_name, device=device)
    paths = iter_image_paths(filenames, images_dir)
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
    return reorder_embeddings(emb, fns, filenames)


def train_baseline(cfg: TrainConfig) -> dict:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    df = load_train_df(KAGGLE_R1)
    split = get_or_build_split(df, cfg)

    # Backbone features (frozen) cached per-split + per-backbone.
    backbone_slug = cfg.backbone.replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    cache_key_train = f"{backbone_slug}_train_{cfg.split_version}"
    cache_key_val = f"{backbone_slug}_val_{cfg.split_version}"
    images_dir = train_images_dir(KAGGLE_R1)

    train_emb = get_or_cache_backbone_embeddings(
        cfg.backbone,
        split.train_filenames,
        images_dir,
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        device=device,
        cache_key=cache_key_train,
    )
    val_emb = get_or_cache_backbone_embeddings(
        cfg.backbone,
        split.val_filenames,
        images_dir,
        input_size=cfg.input_size,
        batch_size=cfg.batch_size,
        device=device,
        cache_key=cache_key_val,
    )

    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    train_labels_str = [by_fn[f] for f in split.train_filenames]
    val_labels_str = [by_fn[f] for f in split.val_filenames]

    id_to_class = {i: c for c, i in enumerate(sorted(set(train_labels_str)))}
    train_labels = np.array([id_to_class[l] for l in train_labels_str], dtype=np.int64)
    num_classes = len(id_to_class)

    # val labels retain string identities (they are not in id_to_class).
    val_labels = np.asarray(val_labels_str)

    model = ArcFaceModel(
        input_dim=train_emb.shape[1],
        num_classes=num_classes,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        margin=cfg.arcface_margin,
        scale=cfg.arcface_scale,
        dropout=cfg.dropout,
    ).to(device)

    num_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    train_ds = TensorDataset(torch.from_numpy(train_emb).float(), torch.from_numpy(train_labels).long())
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_emb_t = torch.from_numpy(val_emb).float().to(device)

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        group=cfg.wandb_group,
        config={
            **cfg.__dict__,
            "num_parameters": num_params,
            "num_train_identities": len(split.train_identities),
            "num_val_identities": len(split.val_identities),
            "num_train_images": len(split.train_filenames),
            "num_val_images": len(split.val_filenames),
            "backbone_feature_dim": int(train_emb.shape[1]),
        },
    )
    wandb.summary["num_parameters"] = num_params

    best_map = -1.0
    best_epoch = 0
    patience_ctr = 0
    checkpoint_path = CHECKPOINTS / f"{cfg.run_name}.pth"

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _ = model(x, y)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss = running / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_ft = model.get_embeddings(val_emb_t).cpu().numpy()
        val_map = identity_balanced_map(val_ft, val_labels)

        scheduler.step(val_map)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_map": val_map,
            "learning_rate": current_lr,
        })

        improved = val_map > best_map
        if improved:
            best_map = float(val_map)
            best_epoch = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "id_to_class": id_to_class,
                "num_classes": num_classes,
                "val_map": best_map,
                "num_parameters": num_params,
                "backbone_feature_dim": int(train_emb.shape[1]),
            }, checkpoint_path)
        else:
            patience_ctr += 1
        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} acc {train_acc:.3f} | val_mAP {val_map:.4f} | lr {current_lr:.2e} | best {best_map:.4f}@{best_epoch} patience {patience_ctr}")

        if patience_ctr >= cfg.patience:
            print(f"Early stop @ epoch {epoch}, best mAP {best_map:.4f} @ {best_epoch}")
            break

    wandb.summary["best_val_map"] = best_map
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["checkpoint_path"] = str(checkpoint_path)

    artifact = wandb.Artifact(name=cfg.run_name, type="model")
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    wandb.finish()

    return {"best_val_map": best_map, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="baseline-megadescriptor-arcface")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-group", default=None)
    args = p.parse_args()
    cfg = TrainConfig(run_name=args.run_name, num_epochs=args.num_epochs, seed=args.seed, wandb_group=args.wandb_group)
    out = train_baseline(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
