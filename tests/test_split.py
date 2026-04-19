import pandas as pd

from jaguar_reid.data import assert_identity_disjoint, build_identity_disjoint_split


def _make_df(n_per: dict) -> pd.DataFrame:
    rows = []
    for ident, n in n_per.items():
        for i in range(n):
            rows.append({"filename": f"{ident}_{i}.png", "ground_truth": ident})
    return pd.DataFrame(rows)


def test_identity_disjoint_split_no_leakage() -> None:
    df = _make_df({f"id_{i}": 5 for i in range(20)})
    split = build_identity_disjoint_split(df, seed=0, val_frac=0.2)
    assert_identity_disjoint(df, split)
    assert set(split.train_identities).isdisjoint(set(split.val_identities))
    assert len(split.train_filenames) + len(split.val_filenames) == len(df)


def test_split_is_deterministic() -> None:
    df = _make_df({f"id_{i}": 3 for i in range(10)})
    a = build_identity_disjoint_split(df, seed=42, val_frac=0.3)
    b = build_identity_disjoint_split(df, seed=42, val_frac=0.3)
    assert a.val_identities == b.val_identities
    assert a.train_filenames == b.train_filenames


def test_split_respects_val_frac() -> None:
    df = _make_df({f"id_{i}": 2 for i in range(50)})
    split = build_identity_disjoint_split(df, seed=1, val_frac=0.2)
    assert 8 <= len(split.val_identities) <= 12
