# Jaguar Re-Identification Q\&A: What Counts as a Valid Experiment? 

*Last revision: February 22nd, 2026*

# **Required Deliverables**

You must submit **TWO** Markdown documents in your GitHub repo. Both must include links to the GitHub repository with the exact Jupyter notebooks used and the corresponding Weights & Biases runs or groups.

### **Document A: EDA\_EXPERIMENTS.md (Exploratory Data Analysis Experiments)**

Experiments focused on understanding the dataset and model behavior, building analysis tooling, and answering diagnostic questions.

Examples include:

* Image dataset analysis (identity distribution, duplicates or near-duplicates, quality, background patterns)  
* Embedding analysis (clustering, nearest neighbors, cohesion or separation)  
* Triplet or quadruplet generation and mining analysis  
* Interpretability (for example, LRP) with sanity and faithfulness tests  
* Background intervention analysis (include vs remove background), with a precise definition of the intervention

### **Document B: LEADERBOARD\_EXPERIMENTS.md (Public Kaggle Leaderboard Optimization Experiments)**

Experiments focused on achieving the highest possible score on the public Kaggle leaderboard, with end-to-end traceability from experiment to code to W\&B to Kaggle submission.

Each leaderboard entry should also include:

* Kaggle submission identifier and public score at the time of submission

You may cross-reference between documents, but each experiment must have a primary home in one document.

---

## **Definition of a Valid Experiment**

A valid experiment answers one clear research question using a systematic, controlled approach and includes:

* Research question or hypothesis (explicit, testable)  
* Defined intervention (what changes vs baseline, and what stays fixed)  
* Evaluation protocol (appropriate to the question)  
* Interpretation (what you learned, why it happened, and what you would do next)

### **What “evaluation” means (depends on the question)**

* Performance questions: report identity-balanced mAP (primary)  
* Interpretability questions: include randomization sanity tests and masking faithfulness tests  
* Efficiency questions: report parameters, FLOPs, latency, or memory, plus mAP when relevant  
* Robustness questions: report stress test deltas, plus mAP when relevant

---

## **What Counts as One Experiment?**

One research question equals one experiment.  
Multiple runs (ablations, variants, seeds) that answer the same question count as one experiment.  
Runs strengthen evidence and do not create extra experiment credits.

---

## **Credits for experiments**

* Valid (1.0): Clear question, controlled intervention, appropriate evaluation, and strong interpretation, with evidence that supports the conclusion.  
* Partial (0.5): Measured and documented change, with weaker evidence (thin controls, too few variants, shallow analysis) or engineering polish without a reusable insight.  
* Not valid (0.0): Uncontrolled changes, “try values until it works,” leaderboard-driven tweaking without a plan, missing evaluation, or missing interpretation or artifacts.

### **Bonus scoring (only for specific comparison experiments)**

Some comparisons receive bonus points only if the experiment meets the Valid (1.0) bar (controlled setup and analysis). If it is Partial (0.5) or Not valid (0.0), no bonus applies.

* Backbone comparison bonus: see Q5  
* Loss-function comparison bonus: see Q11 and Q12

### **Documentation is required (all scores)**

* Reproducible code, fixed validation protocol, clearly named W\&B runs  
* Report section: question → intervention → evaluation → results → analysis  
* If results are noisy: multiple seeds or stability evidence

---

## **Q\&A Entries (with “Where to document this”)**

### **Q0: I measured the difference in identity-balanced mAP when including background vs removing background. Does this count?**

Ruling: 1.0 (Valid experiment)  
Where to document this: EDA\_EXPERIMENTS.md (cross-reference in leaderboard document only if it becomes part of a submission)

Rationale: Key re-identification question: are predictions driven by identity cues or background cues?

You MUST define the background intervention. Valid options include:

* Replace non-jaguar pixels with a constant value (for example, black/gray)  
* Replace non-jaguar pixels with a blurred version of the same image  
* Replace non-jaguar pixels with random noise  
* Use a segmentation method to produce a jaguar mask, then apply one of the replacements outside the mask

What to document:

* Exact intervention definition (including mask generation if used)  
* Where it is applied (train only, eval only, or both) and the reason  
* Identity-balanced mAP under each condition  
* Error analysis: which identities improve or worsen 

---

### **Q1: I compared multiple background interventions (gray vs blur vs noise vs random background). Is this valid?**

Ruling: 1.0 (Valid experiment)  
Where to document this: EDA\_EXPERIMENTS.md

Rationale: Systematic ablation answering one question: which background intervention reduces context reliance without harming identity cues?

What to document:

* Table of interventions and identity-balanced mAP  
* Controls: same model and training protocol across interventions  
* Interpretation: what changes reveal about reliance on background

---

### **Q2: I ran interpretability to visualize which regions the model uses. Does this count?**

Ruling: 1.0 (Valid experiment) only if you include sanity and faithfulness tests  
Where to document this: EDA\_EXPERIMENTS.md

Required minimum bar:

Randomization sanity check:

* Re-run after randomizing weights (or shuffling labels and retraining)  
* Explanation maps should degrade or become uninformative

Masking faithfulness test (measurable):

* Use the explanation map to select top-k% “important” pixels or regions  
* Mask those regions and measure:  
  * Drop in similarity (embedding similarity for true matches vs impostors) and or  
  * Drop in identity-balanced mAP  
* Compare against masking a same-sized random region (control)

What to document:

* Method used (LRP, GradCAM, IG, attention-based)  
* Sanity check evidence  
* Masking protocol (k%, mask value, controls) and similarity or mAP deltas  
* Examples for correct and incorrect retrievals

---

### **Q3: I curated data to use triplet or quadruplet loss and compared it with ArcFace. Does this count?**

Ruling: 1.0 (Valid experiment)

Where to document this:

* Primary goal: mining or structure understanding: EDA\_EXPERIMENTS.md  
* Primary goal: improving public leaderboard: LEADERBOARD\_EXPERIMENTS.md  
  Choose one as primary and cross-reference the other if needed.

What to document:

* Mining strategy and identity balance  
* Controlled comparison (same backbone and schedule when possible)  
* mAP and stability notes  
* Interpretation: why one objective fits this dataset better

---

### **Q4: I modified the learning rate and batch size in the baseline notebook to get better results. Does this count?**

Default ruling: 0.0 (Not valid)  
Where to document this: If it remains ad hoc: nowhere

When it can become valid (0.5 to 1.0):

* Predefined plan (ranges, method), controlled setup  
* Stability evidence (multiple seeds preferred)  
* Analysis of convergence, failure modes, and the reason settings matter

Where to document if reframed:

* Understanding sensitivity: EDA\_EXPERIMENTS.md  
* Final pipeline change: LEADERBOARD\_EXPERIMENTS.md

---

### **Q5: I compared multiple backbones (for example, ResNet18 vs DINOv3 vs EfficientNet vs MegaDescriptor). How is this scored?**

Ruling: One experiment. Score depends on control and analysis, plus backbone-count bonus.  
Where to document this: LEADERBOARD\_EXPERIMENTS.md (optional deeper diagnostics in EDA with cross-reference)

Base requirements:

* Same training protocol, loss, schedule, augmentation, evaluation  
* Same embedding dimension (or justification)  
* Report mAP and at least one efficiency metric

Scoring for backbone comparison:

* Base score: 1.0 if Valid criteria are met  
* Bonus: \+0.20 per backbone included in the controlled comparison  
  * 2 backbones: 1.20  
  * 3 backbones: 1.40  
  * 4 backbones: 1.60  
  * 5 or more backbones: 2.00 (cap)

What to document:

* Why these backbones  
* Table: mAP and efficiency metrics  
* Interpretation: what characteristics matter and why

---

### **Q6: I performed Neural Architecture Search to beat baseline mAP. Is this valid?**

Ruling: 1.0 (Valid experiment)  
Where to document this: LEADERBOARD\_EXPERIMENTS.md

Fairness requirement: Report compute budget and compare to a budget-matched baseline.

What to document:

* Search space, algorithm, budget (number of trials or GPU-hours)  
* Best model vs baseline and budget-matched baselines  
* Patterns discovered

---

### **Q7: I created an ensemble of multiple models. Does this count?**

Ruling: 1.0 (Valid experiment) if you analyze diversity and tradeoffs  
Where to document this: LEADERBOARD\_EXPERIMENTS.md

What to document:

* Which models and the reason for diversity  
* Fusion method  
* mAP: individual models vs ensemble  
* Diversity evidence (error overlap, per-identity gains)  
* Compute or latency tradeoff

---

### **Q8: I ran an extensive hyperparameter sweep and found a better configuration. Is this valid?**

Ruling: 1.0 (Valid experiment)  
Where to document this: LEADERBOARD\_EXPERIMENTS.md if it improves leaderboard performance, otherwise EDA\_EXPERIMENTS.md

What to document:

* Search space, method, number of trials  
* Best configuration vs baseline  
* Hyperparameter importance or sensitivity analysis  
* Use of learning rate scheduler 

---

### **Q9: I implemented data augmentation and it improved identity-balanced mAP. Is this valid?**

Ruling: 0.5 to 1.0 depending on rigor

Where to document this:

* Pipeline improvement: LEADERBOARD\_EXPERIMENTS.md  
* Invariance or robustness analysis: EDA\_EXPERIMENTS.md

Scoring guidance:

* 0.5: standard recipe and on or off ablation  
* 1.0: hypothesis-driven study with controlled component ablations and insight

---

### **Q10: I compared hyperbolic vs hyperspherical embedding spaces. How is this scored?**

Ruling: 1.0 (Valid experiment)  
Where to document this: LEADERBOARD\_EXPERIMENTS.md (optional embedding diagnostics in EDA with cross-reference)

What to document:

* Motivation and implementation details  
* Controlled setup  
* mAP and stability notes  
* Interpretation; optional embedding diagnostics

---

### **Q11: I compared two loss functions. Does this count as a valid experiment?**

Ruling: Yes. One experiment, valid if controlled and analyzed.  
Where to document this: LEADERBOARD\_EXPERIMENTS.md (optional deeper analysis in EDA with cross-reference)

Rationale: Comparing loss functions is a modeling choice and answers a clear research question: which objective works better for this setup?

Validity requirements:

* Controlled comparison (same backbone, schedule, augmentations, embedding dimension, evaluation)  
* Report identity-balanced mAP and training stability notes  
* Interpretation of why the better loss fits this dataset

Loss-function comparison scoring (bonus scheme):  
Apply only if the experiment meets the Valid (1.0) bar.

* 2 loss functions: 1.00  
* 3 loss functions: 1.50  
* 4 loss functions: 2.00  
* 5 loss functions: 2.50  
* 6 loss functions: 3.00  
* 7 loss functions: 3.50  
* 8 loss functions: 4.00  
  Cap: 4.00 (8 or more losses)

---

### **Q12: I compared ArcFace, ArcCos, Focal Loss, and Cross Entropy. How many experiments is this?**

Ruling: One experiment  
Where to document this: LEADERBOARD\_EXPERIMENTS.md

Scoring: With 4 loss functions, this can score 2.00 if it meets the Valid bar.

What to document:

* Why these losses  
* Controlled setup  
* Table of mAP results and convergence or stability differences  
* Interpretation: what properties matter

---

### **Q13: I curated the dataset to select the most unique, high-quality, representative samples for training. Is this valid?**

Ruling: 1.0 (Valid experiment)

Where to document this:

* Data understanding focus: EDA\_EXPERIMENTS.md  
* Final pipeline component: LEADERBOARD\_EXPERIMENTS.md  
  Choose one as primary.

What to document:

* Selection criteria and method  
* mAP vs full dataset and efficiency gains  
* Interpretation of what valuable samples look like

---

### **Q14: I deduplicated the training dataset and measured the performance of the trained model. Does this count as a valid experiment?**

Ruling: Yes. 1.0 (Valid experiment)  
Where to document this: EDA\_EXPERIMENTS.md (cross-reference in leaderboard document only if it becomes part of your final pipeline)

Rationale: The training set contains near-duplicate images. Exploring approaches to detect and remove near-duplicates supports data quality and generalization.

What to document:

* Definition of near-duplicate (exact duplicate vs similarity threshold)  
* Methods tried (perceptual hashing, SSIM, embedding similarity, clustering, kNN pruning, hybrids)  
* Scope (within-identity only vs across identities) and the reason  
* mAP before and after, sensitivity to threshold  
* Effects on training dynamics and mining behavior when relevant

---

### **Q15: I built a classifier to filter retrieval results by jaguar pose. Is this valid?**

Ruling: 1.0 (Valid experiment)

Where to document this:

* If it changes ranking or submission: LEADERBOARD\_EXPERIMENTS.md  
* If it is diagnostic: EDA\_EXPERIMENTS.md

What to document:

* Pose classifier performance  
* Effect on ranking  
* mAP with and without, per-pose breakdown  
* Failure modes

---

### **Q16: I created interpretability visualizations using LRP. Is this valid?**

Ruling: 1.0 (Valid experiment) only if you meet Q2 requirements  
Where to document this: EDA\_EXPERIMENTS.md

What to document: Same requirements as Q2.

---

### **Q17: I compared classical HotSpotter with my best neural approach. Is this valid?**

Ruling: 1.0 (Valid experiment)  
Where to document this: EDA\_EXPERIMENTS.md (cross-reference in leaderboard document if it informs the final approach)

What to document:

* Reproducible HotSpotter setup  
* mAP and runtime comparison  
* Error analysis: complementary strengths and weaknesses

---

### **Q18: I created a smaller-parameter model than MegaDescriptor that achieves better performance. Is this valid?**

Ruling: 1.0 (Valid experiment)  
Where to document this: LEADERBOARD\_EXPERIMENTS.md

What to document:

* Parameters, FLOPs, latency, and mAP  
* Design method (distillation, efficient blocks, pruning)  
* Ablations and interpretation

---

### **Q19: I created a model soup by averaging weights of multiple trained models. Is this valid?**

Ruling: 1.0 (Valid experiment)  
Where to document this: LEADERBOARD\_EXPERIMENTS.md

What to document:

* Which models and compatibility conditions  
* Compare: individual vs prediction ensemble vs soup  
* Interpretation: when it helps or fails

---

### **Q20: I tried three learning rates (0.001, 0.0001, 0.00001) to see which works best. How is this scored?**

Default ruling: 0.0 (Not valid)  
Where to document this: If ad hoc: nowhere

To make it valid:

* Predefined sweep strategy (range and method)  
* Controlled setup, stability evidence  
* Analysis of training behavior and variance

Where to document if reframed:

* Understanding sensitivity: EDA\_EXPERIMENTS.md  
* Final pipeline improvement: LEADERBOARD\_EXPERIMENTS.md

---

### **Q21: I implemented progressive resizing (train at increasing resolutions). Is this valid?**

Ruling: 0.5 to 1.0 depending on insight  
Where to document this: LEADERBOARD\_EXPERIMENTS.md unless it is an efficiency or optimization study, then EDA\_EXPERIMENTS.md

Scoring guidance:

* 0.5: on or off and basic schedule  
* 1.0: study of efficiency vs accuracy, robustness, or stability, with ablations

---

### **Q22: I ran my best experiment according to mAP five to ten times with different random seeds. Does this count as a valid experiment?**

Ruling: 1.0 (Valid experiment)  
Where to document this: LEADERBOARD\_EXPERIMENTS.md (cross-reference in EDA\_EXPERIMENTS.md if you include a variance or stability discussion)

Rationale: Yes. Running the same configuration across five to ten random seeds increases the significance of the result and reduces the chance of reporting a lucky run.

What to document:

* The exact configuration being repeated (model, loss, training schedule, augmentation, validation protocol)  
* The list of random seeds used  
* Identity-balanced mAP for each seed  
* Mean and standard deviation of identity-balanced mAP across seeds  
* Interpretation:  
  * If the standard deviation is small, the result supports the claim of improvement  
  * If the standard deviation is large, the result indicates instability and limits the strength of the claim

Add this as **Q23**, and renumber the current Edge cases question from **Q23 → Q24**.

### **Q23: I compared different optimizers (Adam, AdamW, Muon, SGD with momentum) and learning rate schedulers (cosine annealing, one cycle policy, reduce-on-plateau). Does this count as a valid experiment?**

Ruling: 1.0 (Valid experiment)  
Where to document this:

* If the goal is understanding training stability and sensitivity: EDA\_EXPERIMENTS.md  
* If the goal is improving the public leaderboard and it changes the final pipeline: LEADERBOARD\_EXPERIMENTS.md  
  Choose one as primary and cross-reference the other if needed.

Rationale: Yes. Optimizer choice and learning rate scheduling are core training design decisions. A controlled comparison answers a clear research question, such as “Which optimizer or scheduler yields the best identity-balanced mAP and stability for this dataset and model?”

How to count experiments:

* Comparing multiple optimizers under one fixed scheduler is one experiment (“Which optimizer works best?”).  
* Comparing multiple schedulers under one fixed optimizer is one experiment (“Which scheduler works best?”).  
* Comparing optimizer–scheduler pairs as a grid is one experiment if framed as one question (“Which combination works best?”), but it must be documented as a structured study, not ad hoc tuning.

Validity requirements (minimum):

* Controlled setup: same backbone, loss, augmentations, batch size, embedding dimension, training length, and evaluation protocol  
* Clear definitions: optimizer hyperparameters (weight decay, betas, momentum) and scheduler settings (warmup, max LR, cycle length, patience)  
* Report identity-balanced mAP plus training stability indicators (divergence rate, variance across seeds, convergence curves)

What to document:

* The comparison plan (which optimizers, which schedulers, and why)  
* A results table with identity-balanced mAP for each condition  
* Training dynamics: convergence speed, stability, and sensitivity  
* Mean and standard deviation across seeds for top contenders  
* Interpretation: why the best choice fits this task (regularization, noisy gradients, batch size effects)

### **Q24: I classified images by view type (left flank, right flank, face) and compared jaguars only within the same view type. Does this count as a valid experiment?**

Ruling: 1.0 (Valid experiment)  
Where to document this:

* If it changes retrieval and affects submissions: LEADERBOARD\_EXPERIMENTS.md  
* If it is a diagnostic study of where identity cues come from: EDA\_EXPERIMENTS.md  
  Choose one as primary and cross-reference the other if needed.

Rationale: Yes. View-type filtering is a structured method change to the retrieval pipeline. Jaguar flanks, the jaguar’s face, and the rosette patterns on their foreheads contain distinct identity cues. It is useful to test whether restricting comparisons to the same view type increases re-identification performance by reducing viewpoint mismatch and forcing the model to compare like-with-like.

How to evaluate (recommended):

* Train baseline re-ID model unchanged.  
* Add a view-type classifier (or a rule-based proxy if justified).  
* At retrieval time, filter or re-rank candidates so that a query image is compared first (or only) against the same view type.  
* Report identity-balanced mAP:  
  * Baseline retrieval (no view filtering)  
  * View-filtered retrieval  
  * Optional: two-stage retrieval (same view first, fallback to all views)

What to document:

* View classes and labeling method (manual labels, model predictions, or heuristics)  
* View classifier performance (accuracy/confusion matrix)  
* Exact filtering or re-ranking rule  
* mAP change overall and per view type  
* Failure modes: misclassified views, ambiguous views, partial occlusions  
* Interpretation: when view filtering helps and when it hurts

**Q25: I trained 4 different backbones and compared them across 3 different loss functions (a 4×3 grid). How many experiment credits is this worth?**  
**Ruling:** It depends on how many *research questions* you claim and document.

**Answer:**

* **If you frame the whole 4×3 grid as ONE research question** (for example: “Which (backbone, loss) combination works best under one fixed training and evaluation protocol?”), then it counts as **one experiment**.  
  * If it meets the **Valid (1.0)** bar, you may apply both bonuses:  
    * Backbone bonus: 4 backbones → **1.60 total** for the backbone comparison component (1.0 base \+ 0.60 bonus)  
    * Loss bonus: 3 losses → **1.50 total** for the loss comparison component (1.0 base \+ 0.50 bonus)  
    * Combined (single experiment) credit: **2.10** (= 1.0 base \+ 0.60 backbone bonus \+ 0.50 loss bonus)  
  * If it is **Partial (0.5)** or **Not valid (0.0)**, **no bonuses apply**, so it is **0.5** or **0.0**.  
* **If you split it into TWO distinct research questions (two documented experiments),** you can score them separately:  
  * **Backbone comparison** with a fixed loss → **1.60** (Valid, 4 backbones)  
  * **Loss comparison** with a fixed backbone → **1.50** (Valid, 3 losses)  
  * Total across two experiments: **3.10** credits.

**Where to document this:** LEADERBOARD\_EXPERIMENTS.md (with optional deeper diagnostics in EDA and cross-references).

### **Q26: I measured the performance drop of my top-performing model when including background information vs disregarding it. Does this count as a valid experiment?**

Ruling: 1.0 (Valid experiment)  
Where to document this: EDA\_EXPERIMENTS.md (cross-reference in LEADERBOARD\_EXPERIMENTS.md if it is part of your final submission report)

**Rationale**: Yes. This is a critical analysis of whether the top-performing model relies on background cues rather than identity cues. It tests robustness and helps interpret improvements on the leaderboard.

**Bonus:** Any experiment entry that includes this comparison receives a **\+0.5** bonus, applied once per experiment entry.

What to document:

* The exact top-performing configuration (model, loss, training protocol, validation protocol)  
* The background intervention definition (use of included alpha mask or other methods like custom segmentation, gray replacement, random replacement, synthetic background, etc.)  
* Identity-balanced mAP with background included vs background disregarded  
* The mAP delta (performance drop) and a short interpretation of what the drop suggests about context reliance

### **Q27: I compared Bayesian search with random search and/or grid search during hyperparameter tuning. Does this count as a valid experiment?**

Ruling: 1.0 (Valid experiment)

Where to document this: `LEADERBOARD_EXPERIMENTS.md`

**Rationale:**  
 Comparing at least two hyperparameter search methods for your most promising model setup constitutes a valid experiment. A common and recommended workflow is to begin hyperparameter tuning with a Bayesian approach and then refine the most promising regions of the search space using grid search.

### **Q28: I evaluated the use of k-reciprocal re-ranking to improve the performance of my re-identification pipeline. Does this count as a valid experiment?** 

Ruling: 1.0 (Valid experiment)

Where to document this: `LEADERBOARD_EXPERIMENTS.md`

**Rationale:**

Post-processing the similarity neighborhood using k-reciprocal re-ranking can improve the performance of the re-identification algorithm. Fine tuning its parameters (k1, k2, lambda) is a valid experiment.

**Background:**

In k-reciprocal re-ranking for closed-set all-vs-all re-identification, there are three core parameters to tune. **k₁** controls the size of the k-reciprocal neighborhood and has the largest impact: smaller values are conservative and noise-robust, while larger values incorporate more contextual structure but risk over-smoothing; typical values lie in 10–30, with 20 as a common default. **k₂** sets the number of neighbors used for local query expansion, mainly stabilizing the Jaccard distance; it is usually much smaller (3–10), with 6 widely used. **λ** blends the original distance with the Jaccard distance, balancing raw feature similarity and reciprocal structure; values around 0.2–0.5 are standard, with 0.3 being a typical choice.

In practice, most implementations fix k₂ and tune only k₁ and λ, optimizing for mAP. Reasonable defaults (k₁=20, k₂=6, λ=0.3) already work well for most re-ID benchmarks, and gains beyond that are marginal compared to improving feature quality. Secondary factors,  such as L2 feature normalization and the choice of distance metric (Euclidean vs cosine) affect stability but are often implicit rather than exposed as tunable hyperparameters.

**Reference: [https://openaccess.thecvf.com/content\_cvpr\_2017/papers/Zhong\_Re-Ranking\_Person\_Re-Identification\_CVPR\_2017\_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)** 

### Q29: How do I obtain the 25 extra points for the assignment?

**Ruling:** Two independent paths exist. Each has its own requirements.

**Path A: Experiment volume.** Students who complete the highest number of valid (1.0) or partial (0.5) experiments across both EDA\_EXPERIMENTS.md and LEADERBOARD\_EXPERIMENTS.md qualify for the 25 extra points. Bonus-weighted credits (from backbone, loss, or background comparisons) count toward the total.

**Path B: Kaggle leaderboard placement.** Placing in the top 10% of the public Kaggle leaderboard qualifies for the 25 extra points **only if** the student also fulfills the minimum required number of valid experiments. A high leaderboard score without meeting the experiment minimum does not qualify.

**Rationale:** Path A rewards breadth, rigor, and scientific curiosity. Path B rewards strong modeling results, but guards against leaderboard-only optimization by requiring that the underlying experimental work also meets the course standard. Both paths reinforce the principle that documented, controlled experimentation is the foundation of the assignment.

### Q30: There are two published competitions: [Round 1](https://www.kaggle.com/competitions/jaguar-re-id) and [Round 2](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge). What is the difference? Can I use both for experiments? Which one counts for the 25-point bonus?

**Ruling:** Both competitions are valid settings for experimentation and for the leaderboard bonus.

**Key difference:** In Round 2, the background has been removed from all images in the test set. Round 1 uses unmodified images with backgrounds intact.

**Using both rounds as an experiment:** Submitting the same model to both competitions and reporting the difference in identity-balanced mAP constitutes a valid experiment (1.0) if documented with proper controls and interpretation. This directly measures how much your model relies on background context at inference time.

**Where to document this:** LEADERBOARD\_EXPERIMENTS.md

**What to document:**

- The exact model configuration submitted to both rounds (must be identical)  
- Public leaderboard score and Kaggle submission identifier for each round  
- The mAP delta between Round 1 and Round 2  
- Interpretation: what the delta reveals about background reliance in your model  
- Connection to Q0/Q26 if you have also run background intervention experiments during training

**Bonus eligibility:** Either competition qualifies for the top-10% leaderboard bonus (subject to the minimum experiment requirement from Q29). Round 2 currently has fewer participants, which may make the top-10% threshold more reachable.

---

**Q31: I want to implement Layerwise Relevance Propagation (LRP) to interpret classification or similarity results. How should I approach this, and how will it be graded?**

Ruling: Each individual exploratory analysis using LRP counts as 1.0 (Valid experiment), subject to the sanity and faithfulness requirements in Q2.

Where to document this: **EDA\_EXPERIMENTS.md**

Rationale: LRP implementation differs along two axes. First, the attribution target: attributing relevance to individual class labels (e.g., Medrosa vs Patricia) is a different analysis than attributing relevance to pairwise image similarity (BiLRP, which identifies mutual activation patterns between two images of the same individual). Second, the architecture: convolutional networks, MLPs, and transformers each require different propagation rules. Because each combination answers a distinct research question, each counts as a separate experiment.

Note that BiLRP is the most relevant method for interpreting the closed-set re-identification pipeline, which is based on similarity. If you use a transformer-based backbone, you will need to combine BiLRP with AttnLRP.

What to document: Same requirements as Q2, including randomization sanity checks and masking faithfulness tests.

Resources:

- LRP tutorial chapter: [MonXAI19](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf)  
- LRP applied to DINOv2 in GorillaWatch: [attnlrp\_dinov2](https://github.com/Ivenjs/attnlrp_dinov2)  
- BiLRP paper: [arxiv 2003.05431](https://arxiv.org/abs/2003.05431)  
- BiLRP notebook: [`Explain_Similarity_with_BiLRP.ipynb`](https://colab.research.google.com/drive/1GfnUz7ZF9OZ_4cGtNzslXdwSoguGG_Uh?usp=sharing)  
- LRP library (xAI group, TU Berlin): [zennit docs](https://zennit.readthedocs.io/en/stable/getting-started.html)  
- AttnLRP paper: [arxiv 2402.05602](https://arxiv.org/abs/2402.05602)

---

## **Key Principles for Students**

* Two deliverables: EDA\_EXPERIMENTS.md and LEADERBOARD\_EXPERIMENTS.md. Each entry links to GitHub notebooks and W\&B runs or groups.  
* One question equals one experiment.  
* Control matters: isolate one factor when possible.  
* Interpretability requires randomization sanity tests and masking faithfulness tests.  
* Engineering counts when it is hypothesis-driven and analyzed.  
* Negative results can count when execution and analysis support a clear lesson.  
* Leaderboard scores do not replace a validation protocol. Use a fixed validation approach and document it.  
* Bonus scoring applies to backbone comparisons (Q5) and loss comparisons (Q11) when the experiment meets the Valid bar.

