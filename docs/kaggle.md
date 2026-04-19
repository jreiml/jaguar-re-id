# Jaguar Re-Identification: Kaggle Competition Assignment

**Course**: Applied Hands-on Computer Vision  
**Assignment Weight**: 70% of final grade  
**Deadline**: 17 March 2026, 23:59 CET  
**Competition URLs**:

*  [https://www.kaggle.com/competitions/jaguar-re-id/](https://www.kaggle.com/competitions/jaguar-re-id/)  
* [https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/) 

**Last Updated:** January 31st, 2026

---

## Overview

You will apply computer vision and deep learning techniques to solve a real-world animal re-identification problem: identifying individual jaguars from camera trap images.

### Competition Format

- Platform: Kaggle  
- Task: Match individual jaguars across images  
- Evaluation Metric: Identity-balanced Mean Average Precision (mAP)  
- Duration: 2 months

### Required Starting Point

You must begin your work from the baseline notebook:

 **[Jaguar Re-Identification: MegaDescriptor \+ ArcFace Loss](https://www.kaggle.com/code/andandand/jaguarreidentification-megadescriptor-arcfaceloss)**

Fork this notebook on Kaggle and use it as the foundation for your experiments.

### Bonus Points

Students who place in the **top 10% of the open leaderboard** receive **25 bonus points** added to their assignment grade (maximum possible grade: 125 points). The two groups of students who perform the highest number of high quality experiments (including explainability and data quality explorations) qualify for this as well. 

**To claim bonus points**: Register your Kaggle username in this spreadsheet:  
[https://docs.google.com/spreadsheets/d/16Kvs2oagaDd85PbKdxeIH\_XZ10SccYzJQh5xTszOe0c/edit?usp=sharing](https://docs.google.com/spreadsheets/d/16Kvs2oagaDd85PbKdxeIH_XZ10SccYzJQh5xTszOe0c/edit?usp=sharing)

---

## Team Structure

| Team Size | Minimum Experiments | Deliverables |
| :---- | :---- | :---- |
| 2 participants | 12 distinct experiments | W\&B runs \+ 2 markdown summary files \+ report |
| 1 participant (by exception) | 6 distinct experiments | W\&B runs \+ 2 markdown summary files \+ report |

Single-participant teams are held to the same quality standards as 2-person teams.

---

## Deliverables

### 1\. Weights & Biases (W\&B) Training Runs

Track all experiments in W\&B with:

- All hyperparameters logged  
- Training and validation metrics tracked throughout training  
- **Number of model parameters logged** (use `wandb.log` or `wandb.config`)  
- Model checkpoints saved as artifacts  
- Visualizations where applicable  
- Clear run naming convention  
- W\&B project named as: `jaguar-reid-[team-name]` or `jaguar-reid-[student-name]`

Example for logging model parameters:

`wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})`

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

### 3\. Written Report (PDF, 1-2 pages)

Structure your report as one page PDF with two sections

- Exploratory Data Analysis  
- Model training and Evaluation

These two sections should summarize your findings described on the markdown documents.   
---

## Minimum Requirements

### Experiment Requirements

Each experiment must have a **meaningfully different configuration**:

**Valid variations:**

- Use of classifiers to filter jaguars by positions  
- Dataset curation curation to select representative or high quality samples   
- Model architecture (MegaDescriptor, DINOv3, ConvNeXt v2, EfficientNet)  
- Loss function (ArcFace, Triplet Loss, CosFace)  
- Data augmentation strategy  
- Significant hyperparameter changes (learning rate scheduler, embedding dimension, weight decay)  
- Training data composition  
- Model ensembling

Complex experiments defined in the Q\&A document may earn multiple 'Experiment Credits.' You must accumulate a total of 12 Experiment Credits (Team) or 6 Experiment Credits (Solo). A standard valid experiment is worth 1.0 Credit.

**Invalid variations:**

- Randomly changing hyperparameters ('guessing') without a defined search strategy or hypothesis.  
- Running a new seed without analysis or aggregation (see [Q22](https://docs.google.com/document/d/1C_cWFKgkgxne_uq4RjjpI8EiBFiI-SKzO-4GzSwpbAA/edit?usp=sharing) for valid stability experiments).  
- Identical configuration with different checkpoint saves

## Baseline Improvement Requirement:

Single-run experiments using the default MegaDescriptor+ArcFace configuration that fail to beat the baseline (e.g 0.741 mAP) are invalid. However, **structured studies** (sweeps, ablations) involving this architecture are valid **regardless** of the mAP, provided they include analysis of why performance degraded.  
---

## Valid Experiment Categories

### 1\. Domain Understanding

- **Jaguars have distinct patterns on their flanks:** each flank of a jaguar is unique, the rosette pattern on their foreheads is highly distinctive. Can you exploit this information? Can you use classifiers to filter the data and make the comparisons more precise? 

### 2\. Model Architecture

- **Foundation Models**: MegaDescriptor, DINOv3, ConvNeXt v2, Swin Transformers  
- **Lightweight Models**: EfficientNet, NFNet, MobileNet v3 (analyze trade-offs for edge deployment)  
- **Loss Functions**: ArcFace, Triplet Loss, combined losses  
- **Classical Approaches**: HotSpotter algorithm

### 3\. Model Combinations

- Ensemble methods and stacking  
- Early, intermediate, and late fusion strategies  
- Edge detection fusion  
- Interpretability with LRP visualization

### 4\. Data & Training

- Data augmentation (geometric, color, Cutout, MixUp, CutMix)  
- Hard negative mining  
- Curriculum learning  
- Test-time augmentation (TTA)  
- Cross-validation strategies

### 5\. Hyperparameter Optimization

- Learning rate schedules  
- Optimizer comparisons (Adam, SGD, AdamW, Muon)  
- Embedding dimension and hidden layer sizes  
- ArcFace margin and scale parameters

---

## Baseline Notebook & Configuration

Start from the baseline notebook on Kaggle:

[**Jaguar Re-Identification: MegaDescriptor \+ ArcFace Loss**](https://www.kaggle.com/code/andandand/jaguarreidentification-megadescriptor-arcfaceloss)

This notebook implements the full pipeline (data loading, MegaDescriptor backbone, ArcFace training, submission generation). Fork it and modify the configuration below for your experiments:

`config = {`

    `# Paths`

    `"data_dir": Path("/kaggle/input/jaguar-re-id"),`

    `"checkpoint_dir": Path("checkpoints"),`

    `# Model Architecture`

    `"megadescriptor_model": "hf-hub:BVRA/MegaDescriptor-L-384",`

    `"input_size": 384,`

    `"embedding_dim": 256,`

    `"hidden_dim": 512,`

    `# ArcFace Loss`

    `"arcface_margin": 0.5,`

    `"arcface_scale": 64.0,`

    `"dropout": 0.3,`

    `# Training`

    `"batch_size": 32,`

    `"learning_rate": 1e-4,`

    `"weight_decay": 1e-4,`

    `"num_epochs": 50,`

    `"patience": 10,`

    `"val_split": 0.2,`

    `"seed": 42,`

`}`

---

## Grading Rubric (70% of course grade)

| Component | Weight | Key Criteria |
| :---- | :---- | :---- |
| Experimentation Breadth & Depth | 30% | 12+ diverse experiments (6+ for solo), systematic exploration, technical understanding |
| Technical Quality | 25% | Code quality, implementation correctness, reproducibility |
| Report Quality | 20% | Structure, clarity, analysis depth, documentation |
| Improvement over Baseline | 15% | Relative performance vs MegaDescriptor baseline |
| W\&B Tracking Quality | 10% | Completeness, organization, hyperparameters logged |

### Grade Expectations

**90-100 points**: 12+ diverse experiment credits across 4+ categories, professional code, insightful analysis, strong leaderboard performance

**80-89 points**: 12+ experiment credits across 3 categories, solid implementations, good analysis

**70-79 points**: 12 experiments credits with limited variety, adequate code and report

**60-69 points**: 10-11 experiments or minimal variation, basic implementations

**50-59 points**: Fewer than 10 experiments, poor code quality, weak analysis

**Below 50 points**: Fewer than required experiments, invalid experiments, broken code, no baseline improvement, missing deliverables

---

### Component Scoring Details

**Experimentation (30 points max)**

- 27-30: 12+ experiments across 4+ categories, deep technical understanding  
- 24-26: 12 experiments across 3 categories, solid implementations  
- 21-23: 12 experiments with limited variety (1-2 categories)  
- 18-20: 10-11 experiments or minimal variation  
- Below 18: Fewer than 10 experiments or invalid experiments

**Technical Quality (25 points max)**

- 23-25: Professional-quality code, fully reproducible, no errors  
- 20-22: Good code quality, mostly reproducible, minor issues  
- 17-19: Acceptable code, some reproducibility issues  
- 15-16: Poor code quality, difficult to reproduce  
- Below 15: Code does not run, major errors

**Report Quality (20 points max)**

- 18-20: Publication-quality report with deep insights  
- 16-17: Well-written report with good analysis  
- 14-15: Adequate report, basic analysis  
- 12-13: Poorly structured, superficial analysis  
- Below 12: Incomplete or missing report

**Kaggle Performance (15 points max)**

- 14-15: Top 20% of class, significant improvement over baseline  
- 12-13: Top 40% of class, moderate improvement  
- 10-11: Top 60% of class, some improvement  
- 9: Bottom 40%, minimal improvement  
- Below 9: Bottom 20%, no improvement over baseline

**W\&B Tracking (10 points max)**

- 9-10: Exemplary W\&B usage, easy to navigate, comprehensive  
- 8: Good tracking, most information logged  
- 7: Basic tracking, some missing information  
- 6: Poor tracking, difficult to understand experiments  
- Below 6: Minimal or no W\&B tracking

---

## Submission Checklist

### W\&B Tracking

- [ ] Minimum experiments logged (12 for teams, 6 for solo)  
- [ ] All hyperparameters logged including **number of model parameters**  
- [ ] Training/validation metrics tracked  
- [ ] Project shared with instructors or set to public  
- [ ] Clear run naming convention

### Kaggle Notebooks

- [ ] Top-1 and Top-2 solution notebooks submitted and public  
- [ ] Both notebooks execute on Kaggle  
- [ ] W\&B run links included

### Written Report

- [ ] All sections complete  
- [ ] Experiment table with W\&B links  
- [ ] Screenshots of W\&B workspace  
- [ ] Team contribution statement (for 2-person teams)  
- [ ] 1 page, PDF format

### Validity

- [ ] All experiments meaningfully different  
- [ ] MegaDescriptor+ArcFace experiments beat 0.741 mAP  
- [ ] No test set contamination  
- [ ] Kaggle username registered in spreadsheet (for bonus points)

---

## Timeline Recommendations

| Weeks | Focus |
| :---- | :---- |
| 1-2 | Fork the baseline notebook, explore dataset, run baseline, set up W\&B |
| 3-6 | Core experimentation (aim for 10+ experiments for teams, 5+ for solo) |
| 7-8 | Refine top solutions, ensemble attempts |
| 9-10 | Final experiments, polish notebooks |
| 11-12 | Write report, prepare submission |

---

## Resources

### W\&B Documentation

- [W\&B 101 Course](https://wandb.ai/site/courses/101/)  
- [W\&B Sweeps](https://docs.wandb.ai/models/sweeps)  
- [W\&B with PyTorch](https://docs.wandb.ai/guides/integrations/pytorch)

### Key Papers

**Metric Learning:**

- ArcFace: [Deng et al., 2019](https://arxiv.org/abs/1801.07698)  
- Triplet Loss: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)

**Foundation Models:**

- DINOv3: [Scaling Vision Transformers](https://arxiv.org/abs/2508.10104)  
- ConvNeXt v2: [Woo et al., 2023](https://arxiv.org/abs/2301.00808)

**Efficient Models:**

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)  
- NFNet: [Brock et al., 2021](https://arxiv.org/abs/2102.06171)

**Wildlife Re-ID:**

- MegaDescriptor: [Universal Animal Re-ID](https://arxiv.org/abs/2311.09118)  
- HotSpotter: [Computer Vision for Conservation](https://arxiv.org/html/2508.17605v1)  
- GorillaWatch: [An Automated System for In-the-Wild Gorilla Re-Identification and Population Monitoring](https://www.arxiv.org/abs/2512.07776) 

---

## Academic Integrity

- Discuss approaches at a high level with other teams  
- **Do not share code** with other teams  
- **Do not copy code** from Kaggle, GitHub, or other sources; all submitted code must be your own implementation  
- Cite all external resources, papers, and code references  
- Both team members must understand all submitted code  
- Plagiarism results in course failure

---

## Getting Help

1. Check this document first  
2. Search the Slack channel  
3. Post specific questions in Slack  
4. Email as a last resort 

---

Good luck and have fun\! 🐆
