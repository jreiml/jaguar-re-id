from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "cache"
KAGGLE_R1 = CACHE / "kaggle"
KAGGLE_R2 = CACHE / "kaggle_r2"
CHECKPOINTS = REPO / "checkpoints"
LOGS = REPO / "logs"
SPLITS = REPO / "splits"
SUBMISSIONS = REPO / "submissions"
EMB_CACHE = CACHE / "embeddings"

for d in [CHECKPOINTS, LOGS, SPLITS, SUBMISSIONS, EMB_CACHE]:
    d.mkdir(parents=True, exist_ok=True)
