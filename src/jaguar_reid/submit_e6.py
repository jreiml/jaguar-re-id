"""Submission generator for E6-style checkpoints (projection + optional head),
which are stored with `projection_state_dict` rather than a full ArcFaceModel
state. Supports optional bg-replacement at embedding time, and optional
k-reciprocal re-ranking at scoring time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .bg_replace import BgMode
from .data import KAGGLE_R1, KAGGLE_R2, sample_submission_path, test_csv_path, test_images_dir
from .embed import extract_embeddings, load_embeddings, reorder_embeddings, save_embeddings
from .model import EmbeddingProjection, load_backbone
from .paths import EMB_CACHE, SUBMISSIONS
from .rerank import k_reciprocal_rerank
from .submit import validate_submission_format


def make_submission(
    checkpoint_path: Path,
    round_dir: Path,
    out_csv: Path,
    *,
    bg_mode: BgMode = BgMode.AS_IS,
    rerank: bool = False,
    rerank_k1: int = 35,
    rerank_k2: int = 6,
    rerank_lam: float = 0.2,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_name = cfg["backbone"]
    input_size = int(cfg["input_size"])

    backbone_slug = backbone_name.replace("hf-hub:", "").replace("/", "_").replace(":", "_")
    round_tag = "r2" if "kaggle_r2" in str(round_dir) else "r1"
    bg_tag = "" if bg_mode == BgMode.AS_IS else f"_{bg_mode.value}"
    cache_key = f"{backbone_slug}_test_{round_tag}{bg_tag}"

    pairs = pd.read_csv(test_csv_path(round_dir))
    unique = sorted(set(pairs["query_image"]).union(set(pairs["gallery_image"])))
    cache_path = EMB_CACHE / f"{cache_key}.npz"
    if cache_path.exists():
        raw, fns = load_embeddings(cache_path)
        if set(fns) == set(unique):
            raw = reorder_embeddings(raw, fns, unique)
        else:
            cache_path.unlink()
    if not cache_path.exists():
        backbone, _ = load_backbone(backbone_name, device=device)
        paths = [test_images_dir(round_dir) / f for f in unique]
        raw, fns = extract_embeddings(
            backbone, paths, input_size=input_size, batch_size=int(cfg["batch_size"]),
            num_workers=4, device=device, desc=f"embed:{cache_key}", bg_mode=bg_mode,
        )
        save_embeddings(cache_path, raw, fns)
        raw = reorder_embeddings(raw, fns, unique)
        del backbone
        torch.cuda.empty_cache()

    projection = EmbeddingProjection(
        int(ckpt["backbone_feature_dim"]), int(cfg["hidden_dim"]),
        int(cfg["embedding_dim"]), float(cfg["dropout"]),
    ).to(device)
    projection.load_state_dict(ckpt["projection_state_dict"])
    projection.eval()
    with torch.no_grad():
        emb = F.normalize(projection(torch.from_numpy(raw).float().to(device)), p=2, dim=1).cpu().numpy()
    idx = {fn: i for i, fn in enumerate(unique)}

    if rerank:
        dist = k_reciprocal_rerank(emb, None, k1=rerank_k1, k2=rerank_k2, lam=rerank_lam)
        # Convert rerank distance matrix to a [0,1] similarity by 1 - normalized_dist.
        sim_mat = 1.0 - (dist - dist.min()) / max(dist.max() - dist.min(), 1e-12)
        sim_pairs = np.array([
            sim_mat[idx[q], idx[g]] for q, g in zip(pairs["query_image"], pairs["gallery_image"])
        ])
    else:
        q = np.stack([emb[idx[f]] for f in pairs["query_image"]], axis=0)
        g = np.stack([emb[idx[f]] for f in pairs["gallery_image"]], axis=0)
        sim_pairs = (q * g).sum(axis=1)
    sim_pairs = np.clip(sim_pairs, 0.0, 1.0)

    out = pd.DataFrame({"row_id": pairs["row_id"].values, "similarity": sim_pairs})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    validate_submission_format(out_csv, sample_submission_path(round_dir))
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--round", choices=["r1", "r2"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bg-mode", default="as_is", choices=[m.value for m in BgMode])
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-k1", type=int, default=35)
    p.add_argument("--rerank-k2", type=int, default=6)
    p.add_argument("--rerank-lam", type=float, default=0.2)
    args = p.parse_args()
    round_dir = KAGGLE_R1 if args.round == "r1" else KAGGLE_R2
    path = make_submission(
        Path(args.checkpoint), round_dir, Path(args.out),
        bg_mode=BgMode(args.bg_mode), rerank=args.rerank,
        rerank_k1=args.rerank_k1, rerank_k2=args.rerank_k2, rerank_lam=args.rerank_lam,
    )
    print(json.dumps({"submission_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
