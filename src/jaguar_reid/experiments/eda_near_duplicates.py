"""EDA — Q14 near-duplicate analysis.

Runs two detectors over the train set:
  1. Perceptual hash (pHash via imagehash) — cheap, good for exact / near-
     pixel duplicates.
  2. MegaDescriptor cosine similarity — semantic near-duplicates.

For each detector we:
  - sweep a threshold grid and count duplicate pairs,
  - separate within-identity vs across-identity pairs,
  - surface the top-N most-duplicated candidates for visual inspection,
  - write a JSON summary + CSV of pair candidates to logs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..data import load_train_df, train_images_dir
from ..embed import load_embeddings
from ..paths import EMB_CACHE, KAGGLE_R1, LOGS


def _phash(img: Image.Image) -> np.ndarray:
    import imagehash
    return np.asarray(imagehash.phash(img).hash, dtype=np.uint8).flatten()


def run(
    *,
    thresholds_cos: tuple[float, ...] = (0.995, 0.99, 0.98, 0.95, 0.9),
    thresholds_phash: tuple[int, ...] = (0, 2, 4, 6, 8),
    top_n: int = 30,
    emb_cache_name: str = "mega_l384_train_v1.npz",
) -> dict:
    df = load_train_df(KAGGLE_R1)
    by_fn = dict(zip(df["filename"].astype(str), df["ground_truth"].astype(str)))
    imgs_dir = train_images_dir(KAGGLE_R1)

    # --- 1. perceptual hash ---------------------------------------------------
    hashes = {}
    for fn in tqdm(df["filename"].astype(str), desc="phash"):
        with Image.open(imgs_dir / fn) as img:
            hashes[fn] = _phash(img)

    fns = list(hashes.keys())
    arr = np.stack([hashes[f] for f in fns], axis=0)

    phash_summary = {}
    all_phash_pairs = []
    for thr in thresholds_phash:
        n_within = 0
        n_cross = 0
        example_pairs = []
        # Hamming distance in O(n^2) is fine for 3k images.
        for i in range(len(fns)):
            xor = np.bitwise_xor(arr[i], arr[i + 1 :])
            ham = xor.sum(axis=1)
            match_idx = np.where(ham <= thr)[0]
            for j in match_idx:
                other = fns[i + 1 + int(j)]
                same = by_fn[fns[i]] == by_fn[other]
                if same:
                    n_within += 1
                else:
                    n_cross += 1
                if len(example_pairs) < top_n:
                    example_pairs.append((fns[i], other, int(ham[j]), same))
        phash_summary[thr] = {"within": n_within, "cross": n_cross, "examples": example_pairs}
        all_phash_pairs.extend([
            {"thr": thr, "a": a, "b": b, "hamming": h, "same_identity": same}
            for a, b, h, same in example_pairs
        ])

    # --- 2. MegaDescriptor embedding cosine ----------------------------------
    emb_path = EMB_CACHE / emb_cache_name
    cos_summary = {}
    cos_pairs = []
    if emb_path.exists():
        embs, emb_fns = load_embeddings(emb_path)
        emb_fns = list(emb_fns)
        order = [emb_fns.index(f) for f in fns if f in emb_fns]
        emb_mat = embs[order]
        emb_used = [emb_fns[i] for i in order]
        emb_mat = emb_mat / np.maximum(np.linalg.norm(emb_mat, axis=1, keepdims=True), 1e-12)
        sim = emb_mat @ emb_mat.T
        np.fill_diagonal(sim, -np.inf)
        for thr in thresholds_cos:
            mask = sim >= thr
            triu = np.triu_indices_from(mask, k=1)
            n_within = 0
            n_cross = 0
            example_pairs = []
            for i, j in zip(*triu):
                if not mask[i, j]:
                    continue
                a = emb_used[int(i)]; b = emb_used[int(j)]
                same = by_fn[a] == by_fn[b]
                if same: n_within += 1
                else: n_cross += 1
                if len(example_pairs) < top_n:
                    example_pairs.append((a, b, float(sim[i, j]), same))
            cos_summary[thr] = {"within": n_within, "cross": n_cross, "examples": example_pairs}
            cos_pairs.extend([
                {"thr": thr, "a": a, "b": b, "cos": c, "same_identity": same}
                for a, b, c, same in example_pairs
            ])
    else:
        cos_summary = {"note": f"no cached embeddings at {emb_path}; run training first"}

    summary = {
        "n_images": len(fns),
        "phash": {str(k): {"within": v["within"], "cross": v["cross"]} for k, v in phash_summary.items()},
        "cosine": {str(k): {"within": v["within"], "cross": v["cross"]} if isinstance(v, dict) and "within" in v else v for k, v in cos_summary.items()},
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "eda_near_duplicates.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(all_phash_pairs).to_csv(LOGS / "eda_near_duplicates_phash.csv", index=False)
    pd.DataFrame(cos_pairs).to_csv(LOGS / "eda_near_duplicates_cosine.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb-cache", default="mega_l384_train_v1.npz")
    args = p.parse_args()
    run(emb_cache_name=args.emb_cache)
