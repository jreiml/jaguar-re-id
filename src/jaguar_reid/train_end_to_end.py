"""End-to-end training: fine-tune backbone through the ArcFace head.

For experiments where frozen-backbone features are insufficient (e.g., the Q5
backbone comparison at smaller model scales), this module trains the backbone
with its projection head + ArcFace layer on raw images.

Kept separate from `train.py` so the baseline path stays simple and fast.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augment import eval_transforms, training_transforms
from .data import (
    IdentityDisjointSplit,
    assert_identity_disjoint,
    build_identity_disjoint_split,
    load_train_df,
    train_images_dir,
)
from .eval import identity_balanced_map
from .losses import CosFaceLayer, SubCenterArcFaceLayer, triplet_semi_hard_loss
from .model import ArcFaceLayer, EmbeddingProjection, count_parameters
from .paths import CHECKPOINTS, KAGGLE_R1, SPLITS


@dataclass
class E2EConfig:
    backbone: str = "hf-hub:BVRA/MegaDescriptor-L-384"
    input_size: int = 384
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3
    freeze_backbone: bool = False

    loss: str = "arcface"  # one of: arcface, cosface, subcenter_arcface, triplet
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    cosface_margin: float = 0.35
    subcenters: int = 3
    triplet_margin: float = 0.3

    batch_size: int = 16
    learning_rate: float = 3e-5
    backbone_lr: float = 3e-6
    weight_decay: float = 1e-4
    num_epochs: int = 20
    warmup_epochs: int = 1
    patience: int = 6

    seed: int = 42
    num_workers: int = 4
    split_version: str = "v1"
    run_name: str = "e2e-mega-arcface"
    wandb_project: str = "jaguar-reid-jreiml"
    wandb_group: str | None = None
    extra_config: dict = field(default_factory=dict)


class JaguarTrainDataset(Dataset):
    def __init__(self, filenames: list[str], labels: list[int], images_dir: Path, transform):
        self.filenames = filenames
        self.labels = labels
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        p = self.images_dir / self.filenames[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), int(self.labels[idx])


class JaguarEvalDataset(Dataset):
    def __init__(self, filenames: list[str], images_dir: Path, transform):
        self.filenames = filenames
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        p = self.images_dir / self.filenames[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


class E2EModel(nn.Module):
    def __init__(self, cfg: E2EConfig, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy_size = int(self.backbone.default_cfg.get("input_size", (3, cfg.input_size, cfg.input_size))[-1])
            dummy = torch.zeros(1, 3, dummy_size, dummy_size)
            feat = self.backbone(dummy)
        self.feature_dim = int(feat.shape[1])
        self.input_size = dummy_size
        self.projection = EmbeddingProjection(self.feature_dim, cfg.hidden_dim, cfg.embedding_dim, cfg.dropout)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        if cfg.loss == "arcface":
            self.head = ArcFaceLayer(cfg.embedding_dim, num_classes, margin=cfg.arcface_margin, scale=cfg.arcface_scale)
        elif cfg.loss == "cosface":
            self.head = CosFaceLayer(cfg.embedding_dim, num_classes, margin=cfg.cosface_margin, scale=cfg.arcface_scale)
        elif cfg.loss == "subcenter_arcface":
            self.head = SubCenterArcFaceLayer(cfg.embedding_dim, num_classes, k_subcenters=cfg.subcenters, margin=cfg.arcface_margin, scale=cfg.arcface_scale)
        elif cfg.loss == "triplet":
            self.head = None
        else:
            raise ValueError(f"Unknown loss: {cfg.loss}")

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feat = self.backbone(x)
        emb = self.projection(feat)
        if self.head is None or labels is None:
            return None, F.normalize(emb, p=2, dim=1)
        logits = self.head(emb, labels)
        return logits, F.normalize(emb, p=2, dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_or_build_split(df: pd.DataFrame, cfg: E2EConfig) -> IdentityDisjointSplit:
    path = SPLITS / f"val_{cfg.split_version}.json"
    if path.exists():
        split = IdentityDisjointSplit.load(path)
        assert_identity_disjoint(df, split)
        return split
    split = build_identity_disjoint_split(df, seed=cfg.seed, version=cfg.split_version)
    assert_identity_disjoint(df, split)
    split.save(path)
    return split


@torch.no_grad()
def _embed_eval(model: E2EModel, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _, emb = model(imgs, None)
        out.append(emb.cpu().numpy())
    return np.vstack(out)


def train(cfg: E2EConfig) -> dict:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_train_df(KAGGLE_R1)
    split = _get_or_build_split(df, cfg)
    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))

    tr_labels_str = [by_fn[f] for f in split.train_filenames]
    va_labels_str = [by_fn[f] for f in split.val_filenames]
    id_to_class = {i: c for c, i in enumerate(sorted(set(tr_labels_str)))}
    tr_labels = [id_to_class[l] for l in tr_labels_str]
    num_classes = len(id_to_class)

    model = E2EModel(cfg, num_classes=num_classes).to(device)
    num_params = count_parameters(model)

    input_size = model.input_size
    train_tfm = training_transforms(input_size)
    eval_tfm = eval_transforms(input_size)

    imgs_dir = train_images_dir(KAGGLE_R1)
    tr_ds = JaguarTrainDataset(split.train_filenames, tr_labels, imgs_dir, train_tfm)
    va_ds = JaguarEvalDataset(split.val_filenames, imgs_dir, eval_tfm)

    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    params = [
        {"params": model.projection.parameters(), "lr": cfg.learning_rate},
        {"params": model.backbone.parameters(), "lr": cfg.backbone_lr if not cfg.freeze_backbone else 0.0},
    ]
    if model.head is not None:
        params.append({"params": model.head.parameters(), "lr": cfg.learning_rate})

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.num_epochs))
    ce = nn.CrossEntropyLoss()

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        group=cfg.wandb_group,
        config={**asdict(cfg), "num_parameters": num_params, "num_classes": num_classes, "backbone_feature_dim": model.feature_dim},
    )
    wandb.summary["num_parameters"] = num_params

    va_labels = np.asarray(va_labels_str)
    best_map = -1.0
    best_epoch = 0
    patience_ctr = 0
    checkpoint_path = CHECKPOINTS / f"{cfg.run_name}.pth"

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for imgs, labels in tr_dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if cfg.loss == "triplet":
                _, emb = model(imgs, None)
                loss = triplet_semi_hard_loss(emb, labels, margin=cfg.triplet_margin)
            else:
                logits, _ = model(imgs, labels)
                loss = ce(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
        scheduler.step()
        train_loss = total_loss / max(1, total)

        val_emb = _embed_eval(model, va_dl, device)
        val_map = identity_balanced_map(val_emb, va_labels)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_map": val_map,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        improved = val_map > best_map
        if improved:
            best_map = float(val_map)
            best_epoch = epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "id_to_class": id_to_class,
                "num_classes": num_classes,
                "input_size": input_size,
                "feature_dim": model.feature_dim,
                "val_map": best_map,
                "num_parameters": num_params,
            }, checkpoint_path)
        else:
            patience_ctr += 1
        print(f"[{cfg.run_name}] epoch {epoch:03d} loss {train_loss:.4f} val_mAP {val_map:.4f} best {best_map:.4f}@{best_epoch} patience {patience_ctr}")

        if patience_ctr >= cfg.patience:
            break

    wandb.summary["best_val_map"] = best_map
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["checkpoint_path"] = str(checkpoint_path)

    art = wandb.Artifact(name=cfg.run_name, type="model")
    art.add_file(str(checkpoint_path))
    wandb.log_artifact(art)
    wandb.finish()
    return {"best_val_map": best_map, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="hf-hub:BVRA/MegaDescriptor-L-384")
    p.add_argument("--input-size", type=int, default=384)
    p.add_argument("--loss", default="arcface", choices=["arcface", "cosface", "subcenter_arcface", "triplet"])
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--run-name", default="e2e-run")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--backbone-lr", type=float, default=3e-6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = E2EConfig(
        backbone=args.backbone,
        input_size=args.input_size,
        loss=args.loss,
        freeze_backbone=args.freeze_backbone,
        num_epochs=args.num_epochs,
        run_name=args.run_name,
        wandb_group=args.wandb_group,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr=args.backbone_lr,
        seed=args.seed,
    )
    print(json.dumps(train(cfg), indent=2))
