"""E8: Late-fusion ensemble of Phase-2 top models (Q7).

Combines the best single-model checkpoints from E2 (backbone comparison) and
E13 (multi-seed) via L2-normalized-embedding **concatenation**, then re-L2
normalizes. Compared to:
  - each individual model,
  - a cosine-averaged fusion (average of per-model cosine similarity
    matrices — equivalent to averaging the normalized embeddings),
  - weighted concat (weights proportional to each model's val mAP).

Diversity is measured by:
  - Per-identity gain over the best single model (E13-seed2024 DINOv2).
  - Error-overlap: fraction of queries where at least one model is correct
    but the ensemble is not (Q7 "error overlap, per-identity gains").

Computed on `splits/val_v1.json` (identity-disjoint; no training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import IdentityDisjointSplit, load_train_df
from ..embed import load_embeddings, reorder_embeddings
from ..eval import identity_balanced_map
from ..model import ArcFaceModel, EmbeddingProjection
from ..paths import CHECKPOINTS, EMB_CACHE, KAGGLE_R1, LOGS, SPLITS


def _model_val_emb(checkpoint_path: Path, split_fns: list[str]) -> np.ndarray:
    """Project the cached backbone val features through the trained head."""
    device = "cuda"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_slug = cfg["backbone"].replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    emb_path = EMB_CACHE / f"{backbone_slug}_val_v1.npz"
    raw, fns = load_embeddings(emb_path)
    raw = reorder_embeddings(raw, fns, split_fns)

    projection = EmbeddingProjection(
        int(ckpt["backbone_feature_dim"]), int(cfg["hidden_dim"]),
        int(cfg["embedding_dim"]), float(cfg["dropout"]),
    ).to(device)
    if "projection_state_dict" in ckpt:
        projection.load_state_dict(ckpt["projection_state_dict"])
    else:
        # E2-style checkpoint has full ArcFaceModel; pull projection only.
        msd = ckpt["model_state_dict"]
        projection.load_state_dict({k.replace("embedding_net.", ""): v for k, v in msd.items() if k.startswith("embedding_net.")})
    projection.eval()
    with torch.no_grad():
        emb = F.normalize(projection(torch.from_numpy(raw).float().to(device)), 2, 1).cpu().numpy()
    return emb


def _per_query_ap(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-query AP (indexed by query position), excluding self."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    n = len(labels)
    aps = np.zeros(n, dtype=np.float64)
    for i in range(n):
        is_match = (labels == labels[i]).astype(np.int32); is_match[i] = 0
        if is_match.sum() == 0:
            aps[i] = np.nan; continue
        order = np.argsort(-sim[i], kind="stable")
        sm = is_match[order]
        cum = np.cumsum(sm)
        prec = cum / np.arange(1, n + 1)
        aps[i] = (prec * sm).sum() / sm.sum()
    return aps


def _balanced_map_from_aps(aps: np.ndarray, labels: np.ndarray) -> float:
    by_id: dict = {}
    for i, l in enumerate(labels):
        if np.isnan(aps[i]): continue
        by_id.setdefault(str(l), []).append(float(aps[i]))
    return float(np.mean([np.mean(v) for v in by_id.values()]))


def run() -> dict:
    split = IdentityDisjointSplit.load(SPLITS / "val_v1.json")
    df = load_train_df(KAGGLE_R1)
    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    labels = np.asarray([by_fn[f] for f in split.val_filenames])

    members = {
        "dinov2_seed2024": CHECKPOINTS / "E13-arcface-seed2024.pth",  # best single
        "dinov2_seed42": CHECKPOINTS / "E6-arcface.pth",
        "convnextv2": CHECKPOINTS / "E2-convnextv2-large.pth",
        "mega_l384": CHECKPOINTS / "E2-mega-l384.pth",
    }

    embs = {name: _model_val_emb(path, split.val_filenames) for name, path in members.items()}

    # Individual val mAP (sanity-check).
    singles = {name: identity_balanced_map(e, labels) for name, e in embs.items()}

    # Per-query AP tables per member (for diversity analysis).
    per_q = {name: _per_query_ap(e, labels) for name, e in embs.items()}

    def _concat_fuse(members: list[str]) -> np.ndarray:
        stacked = np.concatenate([embs[m] for m in members], axis=1)
        stacked = stacked / np.maximum(np.linalg.norm(stacked, axis=1, keepdims=True), 1e-12)
        return stacked

    def _cosine_average_fuse(members: list[str]) -> np.ndarray:
        # Equivalent to averaging the cosine similarity matrices.
        sims = np.mean([embs[m] @ embs[m].T for m in members], axis=0)
        # Need to turn a similarity matrix back into per-item embeddings for
        # identity_balanced_map's contract. Simpler: compute mAP directly on
        # the pooled similarity.
        np.fill_diagonal(sims, -np.inf)
        n = len(labels)
        by_id: dict = {}
        for i in range(n):
            is_match = (labels == labels[i]).astype(np.int32); is_match[i] = 0
            if is_match.sum() == 0: continue
            order = np.argsort(-sims[i], kind="stable"); sm = is_match[order]
            prec = np.cumsum(sm) / np.arange(1, n + 1)
            by_id.setdefault(str(labels[i]), []).append(float((prec * sm).sum() / sm.sum()))
        return float(np.mean([np.mean(v) for v in by_id.values()]))

    configs = {
        "ensemble_concat_dinov2x2+convnext": ["dinov2_seed2024", "dinov2_seed42", "convnextv2"],
        "ensemble_concat_dinov2+convnext": ["dinov2_seed2024", "convnextv2"],
        "ensemble_concat_all4": ["dinov2_seed2024", "dinov2_seed42", "convnextv2", "mega_l384"],
        "ensemble_concat_dinov2+convnext+mega": ["dinov2_seed2024", "convnextv2", "mega_l384"],
    }
    concat_maps = {}
    for name, ms in configs.items():
        fused = _concat_fuse(ms)
        concat_maps[name] = identity_balanced_map(fused, labels)

    cos_avg_maps = {}
    for name, ms in configs.items():
        cos_avg_maps[f"{name}_cosavg"] = _cosine_average_fuse(ms)

    # Diversity: for the best ensemble, per-identity gain vs best single.
    best_ensemble_name = max(concat_maps, key=lambda k: concat_maps[k])
    best_ensemble_emb = _concat_fuse(configs[best_ensemble_name])
    aps_ens = _per_query_ap(best_ensemble_emb, labels)
    aps_single = per_q["dinov2_seed2024"]

    per_identity_gain = {}
    for ident in np.unique(labels):
        mask = labels == ident
        ens_map = np.nanmean(aps_ens[mask])
        single_map = np.nanmean(aps_single[mask])
        per_identity_gain[str(ident)] = {"ensemble": float(ens_map), "single_best": float(single_map), "gain": float(ens_map - single_map)}

    # Error overlap: per-query, was single_best "correct top-1" AND ensemble NOT, or vice versa.
    # Proxy: top-1 correct = "most-similar gallery image has same identity".
    def _top1_correct(emb):
        sim = emb @ emb.T; np.fill_diagonal(sim, -np.inf)
        top1 = np.argmax(sim, axis=1)
        return (labels[top1] == labels).astype(int)

    single_ok = _top1_correct(embs["dinov2_seed2024"])
    ens_ok = _top1_correct(best_ensemble_emb)
    n = len(labels)
    error_overlap = {
        "both_ok": int(((single_ok == 1) & (ens_ok == 1)).sum()),
        "only_single_ok": int(((single_ok == 1) & (ens_ok == 0)).sum()),
        "only_ens_ok": int(((single_ok == 0) & (ens_ok == 1)).sum()),
        "neither_ok": int(((single_ok == 0) & (ens_ok == 0)).sum()),
        "total": n,
    }

    summary = {
        "singles": singles,
        "concat_ensembles": concat_maps,
        "cos_avg_ensembles": cos_avg_maps,
        "best_ensemble": best_ensemble_name,
        "best_ensemble_val_map": concat_maps[best_ensemble_name],
        "baseline_single_best": {"name": "dinov2_seed2024", "val_map": singles["dinov2_seed2024"]},
        "gain_over_best_single": concat_maps[best_ensemble_name] - singles["dinov2_seed2024"],
        "per_identity_gain_over_best_single": per_identity_gain,
        "top1_error_overlap": error_overlap,
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E8_ensemble.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    args = p.parse_args()
    run()
