"""E7: k-reciprocal re-ranking (Q28) + Bayesian vs random-search (Q27).

Uses the best Phase-2 single model (DINOv2 + ArcFace from E6) to compute val
embeddings once, then applies k-reciprocal re-ranking on the val-vs-val
closed-set retrieval for many (k1, k2, λ) triples. Because the re-ranking
itself is O(n²) in embedding count, val is the cheap proxy (n=479) — no
model re-training.

Two search strategies are compared:
  1. Coarse **grid** over k1 ∈ {10,15,20,25,30}, λ ∈ {0.1,0.2,0.3,0.4,0.5},
     k2 fixed at 6.
  2. **Random search** over a wider continuous space (k1 in [5,40], λ in
     [0.05,0.7], k2 in {3,4,5,6,8,10}) with 40 samples, seed 0.

Both are cheap. The comparison tests whether random-search beats grid inside
the same compute budget (Q27).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data import IdentityDisjointSplit, load_train_df
from ..embed import load_embeddings, reorder_embeddings
from ..eval import identity_balanced_map
from ..model import EmbeddingProjection
from ..paths import CHECKPOINTS, EMB_CACHE, KAGGLE_R1, LOGS, SPLITS
from ..rerank import k_reciprocal_rerank


def _load_val_embeddings(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    split = IdentityDisjointSplit.load(SPLITS / "val_v1.json")
    df = load_train_df(KAGGLE_R1)
    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    labels = np.asarray([by_fn[f] for f in split.val_filenames])

    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    backbone_slug = cfg["backbone"].replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    emb_path = EMB_CACHE / f"{backbone_slug}_val_v1.npz"
    raw, fns = load_embeddings(emb_path)
    raw = reorder_embeddings(raw, fns, split.val_filenames)

    projection = EmbeddingProjection(int(ckpt["backbone_feature_dim"]), int(cfg["hidden_dim"]),
                                     int(cfg["embedding_dim"]), float(cfg["dropout"])).to("cuda")
    projection.load_state_dict(ckpt["projection_state_dict"])
    projection.eval()
    with torch.no_grad():
        emb = F.normalize(projection(torch.from_numpy(raw).float().to("cuda")), p=2, dim=1).cpu().numpy()
    return emb, labels


def _map_from_rerank(emb: np.ndarray, labels: np.ndarray, *, k1: int, k2: int, lam: float) -> float:
    dist = k_reciprocal_rerank(emb, None, k1=k1, k2=k2, lam=lam)
    # identity_balanced_map expects similarities; we'll flip dist to sim.
    sim = -dist
    n = len(emb)
    np.fill_diagonal(sim, -np.inf)
    aps_by_id: dict = {}
    for i in range(n):
        q = labels[i]
        is_match = (labels == q).astype(np.int32); is_match[i] = 0
        if is_match.sum() == 0: continue
        order = np.argsort(-sim[i], kind="stable")
        sm = is_match[order]
        cum = np.cumsum(sm)
        prec = cum / np.arange(1, n + 1)
        ap = float((prec * sm).sum() / sm.sum())
        aps_by_id.setdefault(q, []).append(ap)
    return float(np.mean([np.mean(v) for v in aps_by_id.values()]))


def run(checkpoint: str) -> None:
    emb, labels = _load_val_embeddings(Path(checkpoint))
    baseline_map = identity_balanced_map(emb, labels)
    print(f"Baseline (no rerank) val mAP: {baseline_map:.4f}")

    # --- 1. Grid over (k1, λ) with k2=6 ---
    grid_results = []
    k1_grid = [10, 15, 20, 25, 30]
    lam_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    t0 = time.time()
    for k1 in k1_grid:
        for lam in lam_grid:
            m = _map_from_rerank(emb, labels, k1=k1, k2=6, lam=lam)
            grid_results.append({"method": "grid", "k1": k1, "k2": 6, "lam": lam, "val_map": m})
            print(f"grid k1={k1:3d} k2=6 lam={lam:.2f} -> {m:.4f}")
    grid_time = time.time() - t0
    grid_best = max(grid_results, key=lambda d: d["val_map"])

    # --- 2. Random search over wider space ---
    rng = np.random.default_rng(0)
    rand_results = []
    t0 = time.time()
    for t in range(40):
        k1 = int(rng.integers(5, 41))
        k2 = int(rng.choice([3, 4, 5, 6, 8, 10]))
        lam = float(rng.uniform(0.05, 0.7))
        m = _map_from_rerank(emb, labels, k1=k1, k2=k2, lam=lam)
        rand_results.append({"method": "random", "trial": t, "k1": k1, "k2": k2, "lam": lam, "val_map": m})
    rand_time = time.time() - t0
    rand_best = max(rand_results, key=lambda d: d["val_map"])

    # --- 3. Bayesian optimization (TPE) over the same wider space ---
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bayes_results = []
    t0 = time.time()

    def objective(trial: "optuna.Trial") -> float:
        k1 = trial.suggest_int("k1", 5, 40)
        k2 = trial.suggest_categorical("k2", [3, 4, 5, 6, 8, 10])
        lam = trial.suggest_float("lam", 0.05, 0.7)
        m = _map_from_rerank(emb, labels, k1=k1, k2=k2, lam=lam)
        bayes_results.append({"method": "bayesian", "trial": trial.number, "k1": int(k1), "k2": int(k2), "lam": float(lam), "val_map": float(m)})
        return m

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    bayes_time = time.time() - t0
    bayes_best = max(bayes_results, key=lambda d: d["val_map"])

    # --- 4. Grid refinement around the Bayesian optimum ---
    refine_k1 = [max(5, bayes_best["k1"] - 3), max(5, bayes_best["k1"] - 1), bayes_best["k1"], bayes_best["k1"] + 1, bayes_best["k1"] + 3]
    refine_lam = [max(0.0, bayes_best["lam"] - 0.05), max(0.0, bayes_best["lam"] - 0.02), bayes_best["lam"], bayes_best["lam"] + 0.02, bayes_best["lam"] + 0.05]
    refine_results = []
    t0 = time.time()
    for k1 in refine_k1:
        for lam in refine_lam:
            m = _map_from_rerank(emb, labels, k1=int(k1), k2=bayes_best["k2"], lam=float(lam))
            refine_results.append({"method": "grid_refine", "k1": int(k1), "k2": bayes_best["k2"], "lam": float(lam), "val_map": float(m)})
    refine_time = time.time() - t0
    refine_best = max(refine_results, key=lambda d: d["val_map"])

    summary = {
        "baseline_val_map": baseline_map,
        "grid": {"n_trials": len(grid_results), "seconds": grid_time, "best": grid_best,
                 "top5": sorted(grid_results, key=lambda d: -d["val_map"])[:5]},
        "random": {"n_trials": len(rand_results), "seconds": rand_time, "best": rand_best,
                   "top5": sorted(rand_results, key=lambda d: -d["val_map"])[:5]},
        "bayesian": {"n_trials": len(bayes_results), "seconds": bayes_time, "best": bayes_best,
                     "top5": sorted(bayes_results, key=lambda d: -d["val_map"])[:5]},
        "grid_refine": {"n_trials": len(refine_results), "seconds": refine_time, "best": refine_best,
                        "top5": sorted(refine_results, key=lambda d: -d["val_map"])[:5]},
        "checkpoint": checkpoint,
    }
    print(json.dumps(summary, indent=2))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E7_rerank.json").write_text(json.dumps(summary, indent=2))

    all_rows = grid_results + rand_results + bayes_results + refine_results
    pd.DataFrame(all_rows).to_csv(LOGS / "exp_E7_rerank.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/E6-arcface.pth")
    args = p.parse_args()
    run(args.checkpoint)
