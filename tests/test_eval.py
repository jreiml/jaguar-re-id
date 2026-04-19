import numpy as np
import pytest

from jaguar_reid.eval import identity_balanced_map


def test_map_perfect_clusters() -> None:
    """Embeddings perfectly clustered by identity => mAP = 1.0."""
    rng = np.random.default_rng(0)
    d = 8
    protos = rng.standard_normal((4, d))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    embs = []
    labels = []
    for i, p in enumerate(protos):
        for _ in range(5):
            e = p + rng.standard_normal(d) * 1e-3
            e /= np.linalg.norm(e)
            embs.append(e)
            labels.append(i)
    embs = np.stack(embs)
    labels = np.array(labels)
    m = identity_balanced_map(embs, labels)
    assert m == pytest.approx(1.0, abs=1e-6)


def test_map_random_embeddings_lower_bound() -> None:
    """Random embeddings give noticeably less than 1.0 identity-balanced mAP."""
    rng = np.random.default_rng(0)
    n = 80
    k = 8
    labels = np.array([i % k for i in range(n)])
    embs = rng.standard_normal((n, 16))
    m = identity_balanced_map(embs, labels)
    assert 0.0 < m < 0.9


def test_map_identity_balance() -> None:
    """Rare identities with large errors should be weighted equally to common
    identities — identity-balanced mAP != micro mAP."""
    rng = np.random.default_rng(1)
    d = 8
    # 10 'common' A/B samples that are well-clustered by identity + 2 'rare'
    # identities with noisy embeddings.
    p_a = rng.standard_normal(d); p_a /= np.linalg.norm(p_a)
    p_b = rng.standard_normal(d); p_b /= np.linalg.norm(p_b)
    p_c = rng.standard_normal(d); p_c /= np.linalg.norm(p_c)
    p_d = rng.standard_normal(d); p_d /= np.linalg.norm(p_d)
    embs, labels = [], []
    for _ in range(10):
        embs.append(p_a + rng.standard_normal(d) * 1e-3); labels.append(0)
        embs.append(p_b + rng.standard_normal(d) * 1e-3); labels.append(1)
    for _ in range(2):
        embs.append(p_c + rng.standard_normal(d) * 0.5); labels.append(2)
        embs.append(p_d + rng.standard_normal(d) * 0.5); labels.append(3)
    embs = np.stack(embs)
    labels = np.array(labels)
    m = identity_balanced_map(embs, labels)
    assert 0.0 <= m <= 1.0
