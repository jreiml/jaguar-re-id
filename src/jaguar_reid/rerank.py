"""k-reciprocal re-ranking (Zhong et al., CVPR 2017).

Minimal vectorized implementation for closed-set retrieval. Uses cosine
similarity as the base distance (1 - cos). Returns a re-ranked distance
matrix of the same shape as the input.
"""

from __future__ import annotations

import numpy as np


def _k_reciprocal_neighbors(init_rank: np.ndarray, i: int, k: int) -> np.ndarray:
    forward = init_rank[i, : k + 1]
    backward = init_rank[forward, : k + 1]
    fi = np.where(backward == i)[0]
    return forward[fi]


def k_reciprocal_rerank(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray | None,
    *,
    k1: int = 20,
    k2: int = 6,
    lam: float = 0.3,
) -> np.ndarray:
    """Return re-ranked cosine distances between query and gallery.

    If gallery_emb is None, performs all-vs-all re-ranking using query_emb as
    both query and gallery (the closed-set setting in this assessment).
    """
    if gallery_emb is None:
        all_emb = query_emb
    else:
        all_emb = np.vstack([query_emb, gallery_emb])
    n = len(all_emb)

    all_emb = all_emb / np.maximum(np.linalg.norm(all_emb, axis=1, keepdims=True), 1e-12)
    original_dist = 1.0 - all_emb @ all_emb.T
    original_dist = np.maximum(original_dist, 0.0)
    # Normalize so max column = 1 (common in the reference implementation).
    original_dist = original_dist / np.maximum(original_dist.max(axis=0, keepdims=True), 1e-12)

    init_rank = np.argsort(original_dist, axis=1, kind="stable")

    V = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(n):
        k_rec = _k_reciprocal_neighbors(init_rank, i, k1)
        k_expand = k_rec.copy()
        for cand in k_rec:
            small = _k_reciprocal_neighbors(init_rank, cand, int(round(k1 / 2)))
            if len(np.intersect1d(small, k_rec)) > 2.0 / 3.0 * len(small):
                k_expand = np.union1d(k_expand, small)
        weight = np.exp(-original_dist[i, k_expand]).astype(np.float32)
        V[i, k_expand] = weight / weight.sum()

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n):
            V_qe[i, :] = np.mean(V[init_rank[i, :k2], :], axis=0)
        V = V_qe

    invIndex = [np.where(V[:, c] != 0)[0] for c in range(n)]

    jaccard_dist = np.zeros_like(original_dist)
    for i in range(n):
        temp_min = np.zeros(n, dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[j] for j in indNonZero]
        for jj, idx_set in enumerate(indImages):
            c = indNonZero[jj]
            temp_min[idx_set] += np.minimum(V[i, c], V[idx_set, c])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1.0 - lam) + original_dist * lam
    np.fill_diagonal(final_dist, 0.0)
    if gallery_emb is None:
        return final_dist
    q = len(query_emb)
    return final_dist[:q, q:]
