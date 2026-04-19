# Jaguar Re-ID Autonomous Agent — Spec

You are an autonomous agent operating end-to-end on the Jaguar Re-Identification course assignment. You own the full pipeline from empty repo to graded submission. No human approval gates exist between phases — your safety net is self-review via subagents and the rules in this file.

## Mission (one sentence)

Maximize the total assignment grade by simultaneously (a) producing ≥6 valid documented experiments per `docs/assessment.md`, and (b) reaching the **top 10% of the public Kaggle leaderboard** for the [Round 2 Jaguar Re-ID Challenge](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/).

## Success criteria

You are done when ALL of the following are true:

1. `EDA_EXPERIMENTS.md` and `LEADERBOARD_EXPERIMENTS.md` exist in repo root, each entry conforming to the template in `docs/assessment.md` ("Definition of a Valid Experiment").
2. Total of **≥8 valid experiments** (target buffer above the 6-solo minimum) with at least one entry in each of these high-leverage categories: backbone comparison (Q5), loss comparison (Q12), background reliance (Q26), Round1-vs-Round2 delta (Q30), k-reciprocal re-ranking (Q28).
3. A 1-page PDF report at `report.pdf` covering EDA + Model Training & Evaluation, summarizing the MD docs.
4. W&B project `jaguar-reid-jreiml` is public and contains every training run linked from the MD docs. Every run logs `num_parameters`, all hyperparameters, train/val identity-balanced mAP curves, and saves checkpoint artifacts.
5. Best Kaggle public LB submission ranks in the **top 10%** of the Round 2 leaderboard at submission time.
6. Subagent self-review (see `Self-review rules` below) confirms that EVERY experiment entry would pass the rubric and that no submission contains test-set leakage.

## Operating mode

- **Fully autonomous.** Do not stop to ask for approval. Decide, execute, document.
- **Self-correct via subagents**, not via the user. Spawn `general-purpose`, `Explore`, or `Plan` agents (and the `review` / `security-review` skills where relevant) at every checkpoint listed in `Self-review rules`.
- **Only halt** for: (a) success criteria met, (b) hardware/credential failure you cannot resolve, (c) you discover the rubric/competition rules have changed in a way that invalidates your plan, (d) two consecutive identical failure modes after a real attempt to fix.

## Dataset & competition

- Dataset: <https://huggingface.co/datasets/jaguaridentification/jaguars>
- Round 2 competition (test set with backgrounds removed) is the **primary** target for the leaderboard bonus — fewer participants → easier top-10%.
- Round 1 (backgrounds intact) is used for the Q30 R1-vs-R2 delta experiment.
- Baseline notebook to fork: [MegaDescriptor + ArcFace by @andandand](https://www.kaggle.com/code/andandand/jaguarreidentification-megadescriptor-arcfaceloss). Baseline mAP ≈ 0.741. Single-run MegaDescriptor+ArcFace experiments that fail to beat 0.741 are *invalid* per the rubric (structured studies are exempt).

## Phased plan

Execute phases in order. Within a phase, parallelize independent steps. Each phase has explicit "done" gates — do not advance until the gate passes.

### Phase 0 — Bootstrap (single GPU run total)

**Goal:** A reproducible baseline + eval harness + submission script, no experiments yet.

Tasks:
1. Port the Kaggle baseline notebook into a Python module structure (`src/jaguar_reid/{data,model,train,eval,submit}.py`). Do NOT keep notebook code as a notebook for execution — notebooks are for final delivery only (see Phase 5).
2. Download the HF dataset to local writable storage (NEVER write to NFS/Ceph). Verify checksum / row count against the HF dataset card.
3. Establish a **fixed identity-disjoint validation split** (no identity appears in both train and val). Persist the split to `splits/val_v1.json`. Every experiment uses this exact split unless the experiment explicitly varies it.
4. Implement `eval/identity_balanced_map.py` returning a single float. Cross-check against the official Kaggle metric description.
5. Wire W&B integration. Project: `jaguar-reid-jreiml`. Log `num_parameters`, all config keys, train/val mAP per epoch.
6. Implement `submit.py` that produces a Kaggle-format submission TSV/CSV from a checkpoint. Validate format on a 10-row dry run before any real submission.
7. Train the baseline (MegaDescriptor-L-384 + ArcFace, config exactly per `docs/kaggle.md`). Confirm val mAP ≥ 0.741 ± noise. If lower, debug the harness — do not advance.

**Done gate (Phase 0):**
- `pytest -q` (or equivalent) passes for `data`, `eval`, `submit` modules.
- Baseline run logged to W&B with mAP ≥ 0.741 on the fixed val split.
- One real Kaggle submission of the baseline accepted, public score recorded.
- Subagent code review (spawn `general-purpose` with prompt: "Audit `src/jaguar_reid/*` for test-set leakage, eval-metric correctness, and reproducibility. Look at split construction, augmentation locations, ArcFace head usage at inference time. Report under 300 words.") returns clean.

### Phase 1 — EDA experiments (cheap, no full training)

**Goal:** Build EDA tooling + bank 2-3 EDA experiment credits before spending GPU on modeling.

Run in parallel (CPU-only or short GPU jobs):
- **Q14: Near-duplicate analysis** of the training set (perceptual hash + embedding-similarity hybrid). Document scope (within-identity vs across), threshold sensitivity, mAP delta after deduplication. → `EDA_EXPERIMENTS.md`
- **Q0/Q26: Background intervention definition** — pick one (gray, blur, noise, segmentation+gray) with explicit rationale. Build the intervention as a reusable transform; defer measurement to Phase 2.
- **Identity & quality EDA**: distribution of images per identity, image quality histogram (sharpness, exposure), pose distribution (if a fast pose proxy is feasible). → `EDA_EXPERIMENTS.md`

**Done gate (Phase 1):** ≥2 EDA entries committed and self-reviewed.

### Phase 2 — Headline comparisons (highest credit-per-GPU-hour)

**Goal:** Bank the bonus-eligible comparisons. Each is **one** experiment with bonus multipliers.

- **Q5 backbone comparison** — train ≥4 backbones (MegaDescriptor-L-384, DINOv3, ConvNeXt-v2, EfficientNet-V2) under identical loss/schedule/aug/embedding-dim. Target credit: 1.60.
- **Q12 loss comparison** — fix the best backbone from Q5, train ≥4 losses (ArcFace, CosFace/ArcCos, Triplet w/ semi-hard mining, sub-center ArcFace OR Focal). Target credit: 2.00.
- **Q26 background reliance** on the best (backbone, loss) — eval same checkpoint with and without the Phase 1 background intervention. Bonus +0.5.
- **Q30 R1 vs R2 delta** — submit the best checkpoint to BOTH Round 1 and Round 2. One experiment, +1.0.

**Done gate (Phase 2):** All four entries committed to `LEADERBOARD_EXPERIMENTS.md` with W&B group links and Kaggle submission IDs. Subagent rubric review confirms each meets the Valid (1.0) bar.

### Phase 3 — Pipeline boosters (autoresearch may be invoked here)

**Goal:** Push leaderboard score with narrow, well-bounded sub-tasks. This is the ONLY phase where `/autoresearch:autoresearch` is appropriate, and only for sweeps with a pre-stated search plan that can be documented as ONE experiment per the rubric.

Candidates (run in priority order):
- **Q28 k-reciprocal re-ranking** — tune (k1, λ) with k2=6 fixed. Bayesian → grid refinement (Q27 makes this two-experiments-in-one if you also document the Bayesian-vs-grid comparison).
- **Q24 view-type filtering** — train a lightweight pose/view classifier, restrict retrieval to same-view candidates, measure mAP delta overall and per view.
- **Q9 augmentation study** — hypothesis-driven component ablations (not "on/off"). Aim for the 1.0 tier, not 0.5.
- **Q23 optimizer/scheduler comparison** — only if compute budget allows.

If you invoke autoresearch in this phase, the inline config MUST set:
- `Iterations: N` (bounded mode, never unbounded — assessment requires a documented plan).
- `Verify:` returning val identity-balanced mAP from a **proxy training run** (subset, ≤15 min). The proxy MUST be validated once against full training to confirm correlation before relying on it.
- `Guard:` against test-set leakage (e.g., a script that asserts no val identity appears in train).
- A pre-written experiment description in the autoresearch invocation — the autoresearch run becomes ONE documented experiment, not many.

### Phase 4 — Ensemble & final submissions

- **Q7 ensemble** of the top 2-3 single models from Phase 2 (different backbones for diversity). Document fusion method, per-identity gain, latency tradeoff.
- Optional: **Q19 model soup** if checkpoints are weight-compatible.
- Final Kaggle submission of best ensemble or single model to Round 2.

**Daily Kaggle submission budget:** Maximum 3 submissions per day to Round 2. Maximum 1 to Round 1 (only used for Q30 + a sanity check). Track usage in `submissions.log`.

### Phase 5 — Deliverables

1. Convert `src/jaguar_reid/` driver scripts into TWO clean Kaggle notebooks (top-1 and top-2 solutions). Each must execute on Kaggle as a public notebook with W&B run links.
2. Generate `report.pdf` (1 page) — two sections (EDA, Model Training & Evaluation) summarizing the MD docs.
3. Final subagent rubric audit (see Self-review rules).

## Experiment documentation template

Every entry in `EDA_EXPERIMENTS.md` and `LEADERBOARD_EXPERIMENTS.md` MUST include:

```markdown
### E<N>: <one-sentence title>

- **Research question / hypothesis:** ...
- **Intervention:** what changes vs the baseline; what is held fixed (backbone, loss, schedule, aug, embedding dim, val split).
- **Evaluation protocol:** identity-balanced mAP on `splits/val_v1.json`; for interpretability also list sanity + faithfulness tests; for efficiency also list params/FLOPs/latency.
- **Results table:** every variant on a row, columns = (config, val mAP, Kaggle public mAP if submitted, num_params, notes).
- **Interpretation:** what changed, why, what you would do next.
- **Artifacts:**
  - GitHub: `src/.../experiment_E<N>.py` (and notebook if applicable)
  - W&B group: <link>
  - Kaggle submission ID + score (LEADERBOARD only)
```

If an entry would not pass the Q&A rubric in `docs/assessment.md`, do not commit it — re-run with proper controls or move it to a "rejected" section.

## Tooling and environment rules

- Always use `/virtualenv/bin/python3`. Never `python` or `python3` without a path.
- **Never reinstall, upgrade, or `pip install torch`.** The system torch at `/usr/local/lib/python3.12/dist-packages/torch` is a custom NVIDIA build with CUDA. Use `--no-deps` when installing packages that depend on torch.
- Network drives (NFS, Ceph) are read-only. Write all checkpoints, logs, dataset cache, and submissions to local disk under `/code/jaguar-re-id/{checkpoints,logs,cache,submissions}`.
- GPU: `CUDA_VISIBLE_DEVICES=0` (single H100). One training job at a time on this GPU.
- W&B: project `jaguar-reid-jreiml`. Log `num_parameters` on every run.
- Kaggle CLI: configure `~/.kaggle/kaggle.json` once; reuse for all submissions.
- Hugging Face: use the `hf` CLI (skill `hf-cli`) for dataset download and any artifact upload.

## Job orchestration

- Long-running training (any job > 5 min) MUST be started via `Bash` with `run_in_background: true`. Capture both stdout and stderr to `logs/<run_name>.log` so you can inspect later.
- Pattern:
  ```
  Bash(run_in_background=true,
       command="cd /code/jaguar-re-id && /virtualenv/bin/python3 -m jaguar_reid.train --config configs/<...>.yaml > logs/<run>.log 2>&1")
  ```
- Use `Monitor` (deferred tool) to stream the log so you get notified on completion. Do not poll with `sleep` loops.
- If a run fails, read the log, fix the root cause, and re-launch. Do not retry the same broken command.
- Only one training job on the GPU at a time. CPU-bound work (EDA, dedup, report writing) can run in parallel with a training job.

## Memory and persistence

- Update `MEMORY.md` and the `memory/` files whenever you learn something durable about the project (a constraint, a non-obvious dataset quirk, a successful pattern). Do NOT use memory as a task tracker — that's what `TaskCreate` is for.
- Maintain a `submissions.log` (TSV) in repo root: `date\ttime\tround\tnotebook\tsubmission_id\tpublic_score\trun_w&b_link`. Read this before every Kaggle submission to check the daily budget.

## Self-review rules (mandatory checkpoints)

You operate without human approval, so subagent review is the only safety net. You MUST spawn a review subagent at each of these checkpoints. Do not skip even if you are confident.

| Checkpoint | Subagent | Prompt focus |
|---|---|---|
| Phase 0 done | `general-purpose` | Test-set leakage, metric correctness, reproducibility |
| Before EVERY Kaggle submission | `general-purpose` | Submission file format, no leakage, reasonable score range |
| Before committing an experiment MD entry | `general-purpose` | Does this entry meet the Valid (1.0) bar in `docs/assessment.md`? Cite specific Q&A items. |
| Phase 2 done | `Plan` | Has every bonus-eligible comparison been claimed? Any easier credits left on the table? |
| Phase 5 (final audit) | `general-purpose` | Read every MD entry, every notebook header, the report. Predict the grade against the rubric. List any risks of failing. |

If a subagent finds a problem, fix it BEFORE moving on. Re-run the review until clean.

## Where `/autoresearch:autoresearch` fits

Autoresearch is the WRONG tool for the assessment as a whole — its modify-and-revert loop produces ad-hoc tweaking that the rubric (Q4, Q20) marks invalid. Use it only:

- In Phase 3, for narrow numerical sweeps (k-reciprocal hyperparameters, augmentation strength) where you can pre-state the search strategy and document the entire run as ONE experiment.
- Always with `Iterations: N` (bounded), a fast proxy `Verify`, and a leakage `Guard`.
- Never for "improve mAP" as a goal — only for a specific hypothesis like "find optimal (k1, λ) via Bayesian then grid refinement."

## What you MUST do

- Self-review at every checkpoint above.
- Document each experiment using the template, in the right MD file.
- Cap Kaggle submissions at the daily budget; record every submission in `submissions.log`.
- Beat the 0.741 baseline before claiming any single-run MegaDescriptor+ArcFace experiment.
- Use the fixed val split for every comparison.

## What you MUST NOT do

- Do NOT modify torch or change `/virtualenv/bin/python3`.
- Do NOT write to NFS/Ceph mounts.
- Do NOT submit to Kaggle without a subagent review.
- Do NOT use `git add -A` (per `~/.claude/CLAUDE.md`).
- Do NOT pause for human approval — decide and act, then self-review.
- Do NOT modify `docs/assessment.md` or `docs/kaggle.md` — they are the contract.
- Do NOT use `--no-verify` or skip pre-commit hooks.
- Do NOT change the val split mid-stream; if you need a different split, version it (`splits/val_v2.json`) and document why.
