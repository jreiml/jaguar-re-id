"""E23: Optimizer comparison on the best single-model recipe (Q23).

Controlled comparison of optimisers on the Phase-2 winning DINOv2-L + 256-d
projection + ArcFace pipeline (frozen backbone features cached once, so only
the head-training optimiser changes). All other hyperparameters held fixed.

Q23 allows comparing optimisers under one scheduler (= one experiment).
We use AdamW, SGD+Nesterov-momentum, and RMSProp — three optimiser families
covering adaptive-moment (AdamW), classic-momentum (SGDm), and RMS-gradient
(RMSProp). Scheduler is ReduceLROnPlateau on val-mAP in all three (max
mode), so dynamics are comparable.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

from ..data import IdentityDisjointSplit, load_train_df, train_images_dir, iter_image_paths
from ..embed import extract_embeddings, load_embeddings, reorder_embeddings, save_embeddings
from ..eval import identity_balanced_map
from ..model import ArcFaceLayer, EmbeddingProjection, count_parameters, load_backbone
from ..paths import CHECKPOINTS, EMB_CACHE, KAGGLE_R1, LOGS, SPLITS


OPTIMIZERS = ["adamw", "sgdm", "rmsprop"]


def _make_optimizer(name: str, params, lr: float, wd: float):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgdm":
        # Nesterov SGD with momentum — classic.
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9, weight_decay=wd)
    raise ValueError(name)


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _cache_features(backbone_name: str, fns: list[str], cache_key: str, input_size: int) -> np.ndarray:
    path = EMB_CACHE / f"{cache_key}.npz"
    if path.exists():
        emb, cached = load_embeddings(path)
        if set(cached) == set(fns):
            return reorder_embeddings(emb, cached, fns)
    backbone, _ = load_backbone(backbone_name, device="cuda")
    paths = iter_image_paths(fns, train_images_dir(KAGGLE_R1))
    emb, cached = extract_embeddings(backbone, paths, input_size=input_size, batch_size=32, num_workers=4, desc=cache_key)
    save_embeddings(path, emb, cached)
    del backbone
    torch.cuda.empty_cache()
    return reorder_embeddings(emb, cached, fns)


def train_one_opt(name: str, *, backbone: str, input_size: int, num_epochs: int, lr: float) -> dict:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    _set_seed(42)
    device = "cuda"

    split = IdentityDisjointSplit.load(SPLITS / "val_v1.json")
    df = load_train_df(KAGGLE_R1)
    backbone_slug = backbone.replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    tr = _cache_features(backbone, split.train_filenames, f"{backbone_slug}_train_v1", input_size)
    va = _cache_features(backbone, split.val_filenames, f"{backbone_slug}_val_v1", input_size)

    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    tr_str = [by_fn[f] for f in split.train_filenames]
    va_labels = np.asarray([by_fn[f] for f in split.val_filenames])
    id_to_class = {i: c for c, i in enumerate(sorted(set(tr_str)))}
    tr_lab = np.array([id_to_class[l] for l in tr_str], dtype=np.int64)

    projection = EmbeddingProjection(tr.shape[1], 512, 256, 0.3).to(device)
    head = ArcFaceLayer(256, len(id_to_class), margin=0.5, scale=64.0).to(device)
    params = list(projection.parameters()) + list(head.parameters())
    num_params = sum(p.numel() for p in params)
    opt = _make_optimizer(name, params, lr, 1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    ce = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(tr).float(), torch.from_numpy(tr_lab).long())
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    vt = torch.from_numpy(va).float().to(device)

    run_name = f"E23-{name}"
    wandb.init(
        project="jaguar-reid-jreiml", name=run_name, group="exp_E23_optimizer",
        config={"optimizer": name, "backbone": backbone, "input_size": input_size,
                "learning_rate": lr, "weight_decay": 1e-4, "batch_size": 64,
                "num_epochs": num_epochs, "num_parameters": num_params,
                "seed": 42, "split_version": "v1"},
    )
    wandb.summary["num_parameters"] = num_params

    best_map = -1.0; best_epoch = 0; patience_ctr = 0
    checkpoint_path = CHECKPOINTS / f"{run_name}.pth"
    for epoch in range(1, num_epochs + 1):
        projection.train(); head.train()
        total = 0.0; n = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = head(projection(x), y)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        projection.eval()
        with torch.no_grad():
            v = F.normalize(projection(vt), 2, 1).cpu().numpy()
        vm = identity_balanced_map(v, va_labels)
        sched.step(vm)
        current_lr = opt.param_groups[0]["lr"]
        wandb.log({"epoch": epoch, "train_loss": total/n, "val_map": vm, "learning_rate": current_lr})
        if vm > best_map:
            best_map = float(vm); best_epoch = epoch; patience_ctr = 0
            torch.save({"epoch": epoch, "projection_state_dict": projection.state_dict(),
                        "head_state_dict": head.state_dict(),
                        "config": {"optimizer": name, "backbone": backbone, "input_size": input_size,
                                   "learning_rate": lr, "weight_decay": 1e-4, "batch_size": 64,
                                   "num_epochs": num_epochs, "hidden_dim": 512, "embedding_dim": 256, "dropout": 0.3,
                                   "arcface_margin": 0.5, "arcface_scale": 64.0, "seed": 42,
                                   "split_version": "v1"},
                        "id_to_class": id_to_class, "num_classes": len(id_to_class),
                        "val_map": best_map, "backbone_feature_dim": int(tr.shape[1]),
                        "num_parameters": num_params}, checkpoint_path)
        else:
            patience_ctr += 1
        print(f"[E23-{name}] epoch {epoch:03d} loss {total/n:.4f} val_mAP {vm:.4f} lr {current_lr:.2e} best {best_map:.4f}@{best_epoch} patience {patience_ctr}")
        if patience_ctr >= 10:
            break

    wandb.summary["best_val_map"] = best_map; wandb.summary["best_epoch"] = best_epoch
    wandb.finish()
    return {"optimizer": name, "best_val_map": best_map, "best_epoch": best_epoch,
            "checkpoint_path": str(checkpoint_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vit_large_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--only", default=None)
    args = p.parse_args()

    results = []
    # Per-optimiser LR tuned to same OOM as AdamW default 1e-4:
    lr_map = {"adamw": 1e-4, "sgdm": 1e-2, "rmsprop": 1e-4}
    for name in OPTIMIZERS:
        if args.only and args.only != name: continue
        results.append(train_one_opt(name, backbone=args.backbone, input_size=args.input_size,
                                      num_epochs=args.num_epochs, lr=lr_map[name]))
    print(json.dumps(results, indent=2))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E23_optimizer.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
