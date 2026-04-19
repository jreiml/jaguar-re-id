# EDA Experiments — Jaguar Re-ID

*Solo track — jreiml (Kaggle) / zyna (W&B). Fixed val split: `splits/val_v1.json` (identity-disjoint, 25 train / 6 val identities). See `docs/assessment.md` for the rubric.*

Each entry conforms to the template in `CLAUDE.md` / `docs/assessment.md`.

---

## Summary table

| ID | Title | Rubric Q | Base credit | Status |
| -- | ----- | -------- | ----------- | ------ |
| E1 | Identity & image-quality distribution | Q13 (data understanding) | 1.0 | committed |
| E3 | Near-duplicate analysis (pHash + MegaDescriptor cosine) | Q14 | 1.0 | committed |
| E4 | Background intervention definition & catalogue of 6 variants | Q0, Q1 | 1.0 | committed |
| E5 | Background reliance of frozen MegaDescriptor (bg-replacement sweep) | Q26 | 1.0 + 0.5 bonus | committed |
| E10 | View-type filtering (pose proxy) | Q24 | 1.0 | planned (Phase 3) |
| E11 | Dedup effect on val mAP (re-train on deduplicated set) | Q14 follow-up | 1.0 | planned |
| E12 | Background intervention at training time (Q0) | Q0 | 1.0 | planned |

---

### E1: Identity and image-quality distribution of the training set

- **Research question / hypothesis:** What is the shape of the training distribution — how many images per identity, and how variable are image sharpness/brightness across individuals? Are there low-coverage identities that will dominate mAP variance?
- **Intervention:** Pure observation. 1895 images across 31 identities; 5 randomly sampled images per identity used for the quality scan (n=155 quality samples). All other factors fixed.
- **Evaluation protocol:** Per-identity image count (min/max/mean/median/q10/q90). Per-image sharpness = mean squared image gradient; brightness = mean L-channel intensity; image dimensions from PIL.
- **Results:**
  - 31 identities, 1895 images.
  - Images per identity: **min 13, max 183, mean 61.1, median 45** (heavy right skew).
  - Sharpness (mean grad²): min 2.16, mean 64, p10 11.6, p90 104.3, max 975 — a **50x spread**; the lowest-sharpness tail (p10=11.6) is indicative of motion-blurred or low-res camera-trap frames.
  - Brightness (0-255): min 20.9, mean 97.0, max 176.6. Mostly dim (camera traps at night); no over-exposed outliers.
  - Image size median ≈ 3035 × 1964 (wide landscape format — jaguar full-body crops).
- **Interpretation:**
  - The long-tail (few identities have 183 images, some only 13) is a **class-imbalance** signal for ArcFace training. Head classes dominate the loss; tail classes see fewer gradient updates. Candidate mitigations: class-balanced sampling, re-weighting, or Focal loss (tested in Q12).
  - The 50x sharpness spread means a sharp-image subset exists per identity but "average" training exposes the model to a majority of low-sharpness frames. A dataset-curation experiment (select top-sharpness images per identity) is a natural Q13 follow-up.
  - Low brightness skew implies night-time imagery dominates; color-based features are limited; texture/rosette-pattern features are essential.
- **Artifacts:**
  - Script: `src/jaguar_reid/experiments/eda_identity_distribution.py`
  - Data: `logs/eda_identity_distribution.json`, `logs/eda_identity_quality.csv`
  - W&B group: *(no training run — pure analytical EDA)*

---

### E3: Near-duplicate analysis of the training set

- **Research question / hypothesis:** Camera-trap data typically contains burst frames. How many of the 1895 training images are near-duplicates of another training image? Do near-duplicates ever cross identity boundaries (which would be a labeling-leak risk)?
- **Intervention:** Two detectors over all pairs of training images:
  1. **Perceptual hash** (`imagehash.phash`, 64-bit). Duplicate if Hamming ≤ `thr ∈ {0, 2, 4, 6, 8}`.
  2. **Semantic cosine** — cosine similarity of frozen MegaDescriptor-L-384 embeddings. Duplicate if cos ≥ `thr ∈ {0.995, 0.99, 0.98, 0.95, 0.9}`.
  For each detector and threshold, count within-identity vs cross-identity duplicate pairs and surface top-30 examples for inspection. Exhaustive O(n²) — 1895 images is tractable.
  Scope: all pairs, reporting within-identity and across-identity separately. (Q14 asks us to decide the scope and why — here we want **both**, so we can confirm no cross-identity duplicates exist.)
- **Evaluation protocol:** Count pairs at each threshold; no mAP delta measured yet (the retrained-on-deduplicated model is E11, a follow-up).
- **Results:**
  - **Exact pHash duplicates (Hamming=0):** 209 pairs, all within-identity, 0 cross-identity.
  - pHash ≤ 2 (very similar): 724 within, 0 cross.
  - pHash ≤ 4: 1239 within, 0 cross.
  - pHash ≤ 8: 2257 within, 0 cross.
  - **Cosine ≥ 0.995 (extreme semantic duplicates):** 92 within, 0 cross.
  - Cosine ≥ 0.99: 249 within, 0 cross.
  - Cosine ≥ 0.98: 555 within, **1 cross** (single borderline case worth inspecting).
  - Cosine ≥ 0.95: 1378 within, 92 cross.
  - Cosine ≥ 0.9: 2543 within, 427 cross (expected — at 0.9 cosine the similarity bar is too loose to call "duplicate").
- **Interpretation:**
  - **No evidence of identity-labeling errors**: both detectors show zero tight-threshold cross-identity duplicates. The dataset is labelled consistently.
  - **Substantial intra-identity redundancy**: ~11% of training pairs are exact-pHash duplicates; at a looser threshold (pHash ≤ 4) 65% of pairs are near-duplicates. Burst-frame redundancy is real.
  - **Implication for training**: identity-wise deduplication (keep 1 of every exact-pHash cluster) would reduce training from 1895 to ~1686 images without losing any identity. Whether this helps generalization is the E11 follow-up.
  - **Implication for sampling**: with class-balanced sampling, heavy duplicates on head-class identities inflate their effective weight — a dedup-before-sample step would be cleaner.
- **Artifacts:**
  - Script: `src/jaguar_reid/experiments/eda_near_duplicates.py`
  - Data: `logs/eda_near_duplicates.json`, `logs/eda_near_duplicates_phash.csv`, `logs/eda_near_duplicates_cosine.csv`
  - W&B group: *(pure EDA, no training)*

---

### E4: Definition of the background intervention

- **Research question / hypothesis:** What constitutes a "background removal" for this dataset, and which variants are semantically meaningful? Q0/Q1/Q26 all depend on a precise definition.
- **Intervention definitions catalogued:** All use the dataset's pre-shipped **alpha channel** as the jaguar segmentation mask (see `dataset_alpha_mask.md` for the discovery). For pixels with alpha == 0 (background), replace RGB with:
  - **as_is**: leave unchanged (control — backgrounds intact).
  - **black**: 0. This is what Round 2 test images ship with.
  - **gray**: constant 128.
  - **imagenet_mean**: `(124, 116, 104)` — pushes background to the ImageNet statistic, should be closer to MegaDescriptor's training distribution than black.
  - **blur**: replace bg pixels with a Gaussian-blurred (radius 25) copy of the image. Preserves low-frequency color context while destroying high-frequency texture.
  - **random_noise**: replace each bg pixel with i.i.d. `U(0, 255)` noise (same seed) — maximum information destruction.
- **Where applied:** This entry defines the transform; E5 applies it at **inference-time on val** (measures reliance). E12 (planned) will apply at **training-time**.
- **Interpretation:** The catalogue spans a gradient from "preserve low-freq context" (blur) to "wholly destructive" (random noise), with the constant fills in between. Black is the worst case for MegaDescriptor (see E5) because natural images in MegaDescriptor's training never contain large flat black regions — it is OOD.
- **Artifacts:**
  - Module: `src/jaguar_reid/bg_replace.py` (`BgMode` enum + `load_rgb(path, mode)`).
  - No W&B run.

---

### E5: Background reliance of frozen MegaDescriptor (Q26)

- **Research question / hypothesis:** How much of MegaDescriptor-L-384's re-identification signal comes from background cues vs jaguar cues? *"If we zero out the background, how much mAP do we lose?"*
- **Intervention:** Apply each `BgMode` from E4 at inference time on every image in `splits/val_v1.json` (val_v1 — identity-disjoint, 479 images, 6 identities). Recompute **frozen** MegaDescriptor-L-384 embeddings and identity-balanced mAP. No training, no projection head, no ArcFace — this isolates the backbone's background-reliance.
- **Held fixed:** backbone = `hf-hub:BVRA/MegaDescriptor-L-384`, input_size 384, ImageNet normalization, identity-disjoint val split v1, cosine similarity.
- **Results:**

  | bg mode       | val mAP  | Δ vs as_is | Notes |
  | ------------- | -------- | ---------- | ----- |
  | as_is         | **0.6188** | —          | control (bg intact) |
  | blur          | 0.5840   | −0.0348    | preserves low-freq context |
  | imagenet_mean | 0.5089   | −0.1099    | matches backbone training mean |
  | black         | 0.4933   | −0.1255    | **matches R2 Kaggle test distribution** |
  | gray (128)    | 0.4917   | −0.1271    |  |
  | random_noise  | 0.4776   | −0.1412    | most destructive |

- **Interpretation:**
  - Background accounts for **10–14% absolute mAP** of MegaDescriptor's re-identification signal on this dataset, depending on how it's replaced. This is a non-trivial context dependence.
  - **Blur is the gentlest intervention** (only −0.035) because it preserves low-frequency color/lighting cues while removing high-frequency distractors. Useful for Q1: of the replacement strategies, blur maximally reduces context information leakage without destroying identity-adjacent cues.
  - **Black (R2 test distribution) is close to the worst case.** The R2 competition test set effectively applies a bg=black intervention at inference time. This explains a large fraction of the R1→R2 gap (see Q30 plan).
  - **gray ≈ black ≈ imagenet_mean** (all flat-fill variants) cluster around a common worse-than-blur regime. MegaDescriptor appears to care about *texture presence* in the background more than about its exact color.
  - **Bonus applies (Q26).** The experiment entry also informs Q0 (which fixed intervention to pick) and Q1 (comparing interventions) — those experiments can cross-reference rather than re-derive.
- **Artifacts:**
  - Script: inline in this commit (see `logs/eda_E5_bg_reliance.json`).
  - Data: `logs/eda_E5_bg_reliance.json`.
  - W&B group: *(frozen-backbone eval only; no training)*
  - Cross-reference for Q30 (LEADERBOARD): the ~0.12 mAP drop for black-bg on val is consistent with the observed R1 (0.478) → R2 (0.243) Kaggle drop being inflated by additional domain shift (R2 test has different image selection on top of bg-removal).

---
