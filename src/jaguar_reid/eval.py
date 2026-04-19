from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def identity_balanced_map(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Identity-balanced mean Average Precision over a set of embeddings.

    Closed-set retrieval: every image is in turn a query, all other images are
    the gallery. Per-query AP is computed against the set of same-identity
    matches (excluding self). We then average APs within each identity (so
    identities with many images do not dominate), then average across
    identities.

    Returns a single float in [0, 1].

    Assumptions:
      - `labels` is a 1-D array of identity strings or ints.
      - Every identity appears at least twice (otherwise APs are undefined
        and those queries are dropped).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels length mismatch")

    labels = np.asarray(labels)
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -np.inf)

    aps_by_identity: dict = {}
    n = len(labels)
    for i in range(n):
        q_label = labels[i]
        is_match = (labels == q_label).astype(np.int32)
        is_match[i] = 0
        n_pos = int(is_match.sum())
        if n_pos == 0:
            continue
        order = np.argsort(-sim[i], kind="stable")
        sorted_matches = is_match[order]
        cum = np.cumsum(sorted_matches)
        precision_at_k = cum / np.arange(1, n + 1)
        ap = float((precision_at_k * sorted_matches).sum() / n_pos)
        aps_by_identity.setdefault(q_label, []).append(ap)

    if not aps_by_identity:
        return 0.0
    per_identity = [float(np.mean(v)) for v in aps_by_identity.values()]
    return float(np.mean(per_identity))
