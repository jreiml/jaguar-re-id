import numpy as np

from jaguar_reid.rerank import k_reciprocal_rerank


def test_rerank_preserves_perfect_clusters() -> None:
    rng = np.random.default_rng(0)
    d = 16
    protos = rng.standard_normal((5, d))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    embs = []
    for p in protos:
        for _ in range(4):
            e = p + rng.standard_normal(d) * 1e-3
            e /= np.linalg.norm(e)
            embs.append(e)
    embs = np.stack(embs)

    dist = k_reciprocal_rerank(embs, None, k1=5, k2=3, lam=0.3)
    assert dist.shape == (20, 20)
    # For well-clustered data, same-cluster distance should be < cross-cluster.
    same = []
    cross = []
    for i in range(20):
        for j in range(20):
            if i == j:
                continue
            if i // 4 == j // 4:
                same.append(dist[i, j])
            else:
                cross.append(dist[i, j])
    assert np.mean(same) < np.mean(cross)
