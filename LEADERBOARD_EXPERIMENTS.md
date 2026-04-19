# Leaderboard Experiments — Jaguar Re-ID

*Solo track — jreiml (Kaggle) / zyna (W&B). Round 2 submissions are **late** (official deadline passed 2026-03-17, instructor approved); they are scored via the Kaggle API but do not appear on the ranked public leaderboard. Round 1 submissions are used only for the Q30 R1-vs-R2 delta.*

*Fixed val split: `splits/val_v1.json` (identity-disjoint, 25 train / 6 val identities). Baseline calibration: see E0 below (not the published 0.741 — that number was measured with a stratified split, which is incompatible with the identity-disjoint protocol mandated here).*

---

## Summary table

| ID | Title | Rubric Q | Base credit | Status |
| -- | ----- | -------- | ----------- | ------ |
| E0 | Baseline calibration: MegaDescriptor-L-384 + ArcFace, identity-disjoint v1 | — | 0 (calibration, not an experiment) | committed |
| E2 | Backbone comparison (4 backbones) | Q5 | 1.60 | planned |
| E6 | Loss comparison (4 losses) | Q12 | 2.00 | planned |
| E7 | k-reciprocal re-ranking (k1, λ sweep) | Q28 + Q27 | 2.00 | planned |
| E8 | Ensemble of top 2-3 single models | Q7 | 1.0 | planned |
| E9 | Round 1 vs Round 2 delta (same model → both rounds) | Q30 | 1.0 | planned |

---

### E0: Baseline calibration (not a graded experiment)

- **Purpose:** Establish the identity-disjoint-val calibration point that Phase 2+ experiments must beat. Not a graded experiment — a calibration anchor for all subsequent rows in this document.
- **Setup:** Exact `docs/kaggle.md` recipe (MegaDescriptor-L-384 frozen + 1536→512→256 projection + ArcFace margin 0.5 scale 64, AdamW lr=1e-4 wd=1e-4, batch 32, 50 epochs with patience 10) on identity-disjoint val_v1 (25 train, 6 val identities).
- **Results:**
  - Val mAP (identity-balanced, val_v1): **0.5842** @ epoch 11 (early-stopped epoch 21).
  - Frozen-backbone reference (no projection, no training): **0.6188** val mAP — so training hurt generalization to the 6 held-out identities. *Expected* under ArcFace with only 25 training identities (the head specialises to train classes at the expense of general retrieval capacity).
  - Kaggle R1 public: **0.478**, private **0.453** (submission ID `baseline_r1.csv` @ 2026-04-19 14:12).
  - Kaggle R2 public: **0.243**, private **0.253** (submission ID `baseline_r2.csv` @ 2026-04-19 14:09).
- **Why so far below the published 0.741?**
  - The 0.741 was measured by the baseline author using a **stratified** val split (identities in both train and val) with **all 31 identities in training**. Our identity-disjoint protocol trains on only 25 identities and the test set (R1) also has identities the model has not seen — a strictly harder setting.
  - R2 (0.243) additionally suffers a large domain shift: R2 test images have RGB pre-zeroed in the background region, and MegaDescriptor-L-384 is OOD on large flat black regions (cf. E5 in `EDA_EXPERIMENTS.md`: frozen mAP drops from 0.62 to 0.49 under bg=black on val).
- **Decision:** Use this as the anchor for Phase 2 comparisons. A "production" model trained on all 31 identities for leaderboard submissions is a separate concern, tracked under `submissions.log` but not as a graded experiment.
- **Artifacts:**
  - Code: `src/jaguar_reid/train.py` (at the commit this entry was added).
  - Checkpoint: `checkpoints/baseline-megadescriptor-arcface.pth`.
  - W&B run: `zyna/jaguar-reid-jreiml` group `phase0_baseline`, run `baseline-megadescriptor-arcface`.
  - Kaggle: `baseline_r1.csv` (R1, 0.478), `baseline_r2.csv` (R2, 0.243); see `submissions.log`.

---
