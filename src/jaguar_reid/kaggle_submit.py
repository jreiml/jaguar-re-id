"""Thin wrapper around the Kaggle CLI for late submissions + submissions.log bookkeeping."""

from __future__ import annotations

import csv
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Iterable

from .paths import REPO

SUBMISSIONS_LOG = REPO / "submissions.log"
COMPETITIONS = {
    "r1": "jaguar-re-id",
    "r2": "round-2-jaguar-reidentification-challenge",
}
MAX_PER_DAY = {"r1": 1, "r2": 3}


def _env_with_kaggle_token() -> dict:
    env = os.environ.copy()
    # CLAUDE.md spec: use env vars for auth, PAT token lives at KAGGLE_API_TOKEN in .env.
    # Caller is expected to have already `source`-d .env; this is a defensive mirror.
    if "KAGGLE_USERNAME" not in env:
        env["KAGGLE_USERNAME"] = "jreiml"
    if "KAGGLE_KEY" not in env and "KAGGLE_API_TOKEN" in env:
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
    return env


def _append_log(row: Iterable[str]) -> None:
    SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not SUBMISSIONS_LOG.exists()
    with SUBMISSIONS_LOG.open("a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if header_needed:
            w.writerow([
                "date", "time", "round", "run_name", "submission_file",
                "public_score", "private_score", "wandb_url", "message",
            ])
        w.writerow(list(row))


def _count_today(round_key: str) -> int:
    if not SUBMISSIONS_LOG.exists():
        return 0
    today = dt.date.today().isoformat()
    n = 0
    with SUBMISSIONS_LOG.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("date") == today and row.get("round") == round_key:
                n += 1
    return n


def submit(
    round_key: str,
    submission_file: Path,
    message: str,
    *,
    run_name: str = "",
    wandb_url: str = "",
    dry_run: bool = False,
) -> dict:
    if round_key not in COMPETITIONS:
        raise ValueError(f"round_key must be one of {list(COMPETITIONS)}")
    used = _count_today(round_key)
    if used >= MAX_PER_DAY[round_key]:
        raise RuntimeError(f"Daily budget exhausted for {round_key}: {used}/{MAX_PER_DAY[round_key]}")
    submission_file = Path(submission_file)
    if not submission_file.exists():
        raise FileNotFoundError(submission_file)

    env = _env_with_kaggle_token()
    cmd = [
        "/virtualenv/bin/kaggle", "competitions", "submit",
        "-c", COMPETITIONS[round_key],
        "-f", str(submission_file),
        "-m", message,
    ]
    if dry_run:
        return {"status": "dry_run", "cmd": " ".join(cmd)}

    res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    stdout = (res.stdout or "") + (res.stderr or "")

    now = dt.datetime.now()
    _append_log([
        now.date().isoformat(), now.strftime("%H:%M:%S"), round_key, run_name,
        str(submission_file), "", "", wandb_url, message,
    ])
    if res.returncode != 0:
        raise RuntimeError(f"Kaggle submit failed ({res.returncode}): {stdout}")
    return {"status": "ok", "output": stdout}


def fetch_latest_score(round_key: str) -> dict:
    env = _env_with_kaggle_token()
    res = subprocess.run(
        ["/virtualenv/bin/kaggle", "competitions", "submissions", "-c", COMPETITIONS[round_key], "-v"],
        capture_output=True, text=True, env=env, check=False,
    )
    if res.returncode != 0:
        return {"status": "error", "output": (res.stdout or "") + (res.stderr or "")}
    return {"status": "ok", "output": res.stdout}
