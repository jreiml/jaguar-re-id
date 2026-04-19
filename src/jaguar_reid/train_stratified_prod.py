"""Production baseline: MegaDescriptor-L-384 + ArcFace on all 31 identities.

Matches the *published* baseline recipe (stratified 80/20 val split where every
identity appears in both train and val). Exists specifically to (a) satisfy
the `docs/kaggle.md` "MegaDescriptor+ArcFace must beat 0.741" validity gate
on its native split protocol, and (b) maximise Kaggle leaderboard potential
by training on the full 31-identity pool.

Separate from `train.py` (which enforces identity-disjoint val_v1 for the
Phase-2 experimental comparisons) and `train_loss_comparison.py` (Phase-2
loss studies). No experiment entries live here — this is a production train.
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
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .embed import extract_embeddings, load_embeddings, reorder_embeddings, save_embeddings
from .eval import identity_balanced_map
from .model import ArcFaceModel, count_parameters, load_backbone
from .data import load_train_df, train_images_dir, iter_image_paths
from .paths import CHECKPOINTS, EMB_CACHE, KAGGLE_R1, SPLITS


@dataclass
class ProdConfig:
    backbone: str = "hf-hub:BVRA/MegaDescriptor-L-384"
    input_size: int = 384
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 10
    seed: int = 42
    val_frac: float = 0.2
    run_name: str = "prod-mega-arcface-stratified"
    wandb_project: str = "jaguar-reid-jreiml"
    wandb_group: str = "phase0_production"


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _cache_emb(backbone_name: str, fns: list[str], cache_key: str, cfg: ProdConfig) -> np.ndarray:
    path = EMB_CACHE / f"{cache_key}.npz"
    if path.exists():
        emb, cached = load_embeddings(path)
        if set(cached) == set(fns):
            return reorder_embeddings(emb, cached, fns)
    backbone, _ = load_backbone(backbone_name, device="cuda")
    paths = iter_image_paths(fns, train_images_dir(KAGGLE_R1))
    emb, cached = extract_embeddings(backbone, paths, input_size=cfg.input_size, batch_size=cfg.batch_size, num_workers=4, desc=cache_key)
    save_embeddings(path, emb, cached)
    del backbone
    torch.cuda.empty_cache()
    return reorder_embeddings(emb, cached, fns)


def train(cfg: ProdConfig) -> dict:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    _seed(cfg.seed)
    device = "cuda"

    df = load_train_df(KAGGLE_R1)
    # Stratified split — every identity in both train and val. This is the
    # published-baseline protocol that defines the 0.741 anchor.
    tr_df, va_df = train_test_split(
        df, test_size=cfg.val_frac, random_state=cfg.seed, stratify=df["ground_truth"],
    )
    tr_fns = tr_df["filename"].astype(str).tolist()
    va_fns = va_df["filename"].astype(str).tolist()

    backbone_slug = cfg.backbone.replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    train_emb = _cache_emb(cfg.backbone, tr_fns, f"{backbone_slug}_train_stratified", cfg)
    val_emb = _cache_emb(cfg.backbone, va_fns, f"{backbone_slug}_val_stratified", cfg)

    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    tr_ids = sorted(set(df["ground_truth"].astype(str)))
    id_to_class = {i: c for c, i in enumerate(tr_ids)}
    tr_labels = np.array([id_to_class[by_fn[f]] for f in tr_fns], dtype=np.int64)
    va_labels_str = np.array([by_fn[f] for f in va_fns])
    num_classes = len(id_to_class)

    model = ArcFaceModel(
        input_dim=train_emb.shape[1], num_classes=num_classes,
        embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
        margin=cfg.arcface_margin, scale=cfg.arcface_scale, dropout=cfg.dropout,
    ).to(device)
    num_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    ds = TensorDataset(torch.from_numpy(train_emb).float(), torch.from_numpy(tr_labels).long())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    val_t = torch.from_numpy(val_emb).float().to(device)

    wandb.init(
        project=cfg.wandb_project, name=cfg.run_name, group=cfg.wandb_group,
        config={**asdict(cfg), "num_parameters": num_params, "num_train_identities": num_classes,
                "num_train_images": len(tr_fns), "num_val_images": len(va_fns),
                "backbone_feature_dim": int(train_emb.shape[1]), "split_protocol": "stratified"},
    )
    wandb.summary["num_parameters"] = num_params

    best_map = -1.0; best_epoch = 0; patience_ctr = 0
    checkpoint_path = CHECKPOINTS / f"{cfg.run_name}.pth"
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0; total = 0
        for x, y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits, _ = model(x, y)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0); total += x.size(0)
        model.eval()
        with torch.no_grad():
            val_ft = model.get_embeddings(val_t).cpu().numpy()
        val_map = identity_balanced_map(val_ft, va_labels_str)
        scheduler.step(val_map)
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"epoch": epoch, "train_loss": total_loss / total, "val_map": val_map, "learning_rate": current_lr})

        if val_map > best_map:
            best_map = float(val_map); best_epoch = epoch; patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "config": asdict(cfg), "id_to_class": id_to_class,
                        "num_classes": num_classes, "val_map": best_map,
                        "num_parameters": num_params,
                        "backbone_feature_dim": int(train_emb.shape[1]),
                        "split_protocol": "stratified"}, checkpoint_path)
        else:
            patience_ctr += 1
        print(f"[{cfg.run_name}] epoch {epoch:03d} loss {total_loss/total:.4f} val_mAP {val_map:.4f} lr {current_lr:.2e} best {best_map:.4f}@{best_epoch} patience {patience_ctr}")
        if patience_ctr >= cfg.patience:
            break

    wandb.summary["best_val_map"] = best_map; wandb.summary["best_epoch"] = best_epoch
    art = wandb.Artifact(name=cfg.run_name, type="model"); art.add_file(str(checkpoint_path)); wandb.log_artifact(art)
    wandb.finish()
    return {"best_val_map": best_map, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--run-name", default="prod-mega-arcface-stratified")
    args = p.parse_args()
    cfg = ProdConfig(run_name=args.run_name, num_epochs=args.num_epochs)
    print(json.dumps(train(cfg), indent=2))
