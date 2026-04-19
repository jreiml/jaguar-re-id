"""EDA: identity distribution and image-quality summary.

Writes `logs/eda_identity_distribution.json` with per-identity stats and a
histogram figure `logs/eda_identity_distribution.png`. Used as evidence for
the EDA entry on identity and quality distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..data import load_train_df, train_images_dir
from ..paths import KAGGLE_R1, LOGS


def _sharpness(img: Image.Image) -> float:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    return float(np.mean(gx * gx + gy * gy))


def _brightness(img: Image.Image) -> float:
    return float(np.mean(np.asarray(img.convert("L"), dtype=np.float32)))


def main(sample_per_identity: int = 3, out_dir: Path | None = None) -> None:
    out_dir = out_dir or LOGS
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_train_df(KAGGLE_R1)
    imgs_dir = train_images_dir(KAGGLE_R1)

    counts = df["ground_truth"].value_counts().sort_values(ascending=False)
    rng = np.random.default_rng(42)

    sharp_rows = []
    for ident, sub in df.groupby("ground_truth"):
        files = sub["filename"].astype(str).tolist()
        pick = rng.choice(files, size=min(sample_per_identity, len(files)), replace=False)
        for fn in pick:
            p = imgs_dir / fn
            try:
                with Image.open(p) as img:
                    sharp_rows.append({
                        "identity": ident,
                        "filename": fn,
                        "width": img.width,
                        "height": img.height,
                        "sharpness": _sharpness(img),
                        "brightness": _brightness(img),
                    })
            except Exception as exc:  # noqa: BLE001
                sharp_rows.append({"identity": ident, "filename": fn, "error": str(exc)})

    sharp_df = pd.DataFrame(sharp_rows)
    sharp_df.to_csv(out_dir / "eda_identity_quality.csv", index=False)

    summary = {
        "num_identities": int(df["ground_truth"].nunique()),
        "num_images": int(len(df)),
        "images_per_identity": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "q10": float(np.quantile(counts, 0.1)),
            "q90": float(np.quantile(counts, 0.9)),
        },
        "quality_sample_size": int(len(sharp_df)),
        "sharpness": {
            "min": float(sharp_df["sharpness"].min()),
            "max": float(sharp_df["sharpness"].max()),
            "mean": float(sharp_df["sharpness"].mean()),
            "p10": float(np.quantile(sharp_df["sharpness"].dropna(), 0.1)) if len(sharp_df) else None,
            "p90": float(np.quantile(sharp_df["sharpness"].dropna(), 0.9)) if len(sharp_df) else None,
        },
        "brightness": {
            "min": float(sharp_df["brightness"].min()),
            "max": float(sharp_df["brightness"].max()),
            "mean": float(sharp_df["brightness"].mean()),
        },
        "image_size": {
            "width_median": float(sharp_df["width"].median()) if len(sharp_df) else None,
            "height_median": float(sharp_df["height"].median()) if len(sharp_df) else None,
        },
    }
    (out_dir / "eda_identity_distribution.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=3)
    args = p.parse_args()
    main(sample_per_identity=args.samples)
