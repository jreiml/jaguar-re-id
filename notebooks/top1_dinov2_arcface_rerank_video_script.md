# Top-1 Notebook — Video Script

## [0:00 — Title cell on screen]

Hi, I'm walking through my top-1 submission for the Jaguar Re-Identification challenge. The notebook is `top1_dinov2_arcface_rerank.ipynb`. In the next ten minutes I'll cover the research question, the four design choices that matter, and the numbers they produced on our held-out validation set and on the Kaggle Round-2 leaderboard.

## [0:30 — Agenda]

The pipeline has four moving parts: a **DINOv2-ViT-Large** backbone, a small **projection head trained with ArcFace**, a **background-fill trick at inference time**, and **k-reciprocal re-ranking** as post-processing. Each one is motivated by a specific hypothesis — I'll state the hypothesis, show the lines of code, and give the measured mAP delta.

## [1:00 — Problem framing]

The task is closed-set re-identification: given a query image of a jaguar and a gallery of candidate images, rank candidates by the probability they're the same individual. The official metric is identity-balanced mean average precision — critically, it's *balanced per identity*, so rare jaguars count as much as common ones. The Round-2 test set has backgrounds masked out with an alpha channel, which — as we'll see — is not free.

My research question, in one sentence: *can a frozen, general-purpose self-supervised vision transformer, combined with a lightweight metric-learning head and cheap post-processing, outperform the MegaDescriptor baseline that was trained specifically for animal re-identification?*

The hypothesis going in was yes, because DINOv2's self-supervised features are unusually strong on fine-grained texture — and jaguar identification is exactly a fine-grained texture problem: each individual has a unique rosette pattern.

## [2:00 — Scroll to the config cell]

Let's walk through the choices. First, the **backbone**. I'm using `vit_large_patch14_reg4_dinov2.lvd142m` from timm, at its native 518-pixel input resolution, frozen. "Frozen" is important — I never fine-tune the 304-million-parameter backbone, I only train a small head on top. That keeps training fast, protects against overfitting to our 25 training identities, and lets me extract features just once per image and reuse them.

Why DINOv2 and not MegaDescriptor? I ran a four-way backbone comparison as a separate experiment — MegaDescriptor-L, DINOv2-L, ConvNeXtV2-L, and EfficientNetV2-L, all with the exact same head and training recipe. DINOv2-L won by seven points of mAP over MegaDescriptor, despite MegaDescriptor being the animal-re-ID-specialised model. The mechanism: the 518-resolution ViT preserves fine-grained rosette texture that the 384-resolution CNNs compress away.

## [3:00 — Scroll to the projection + ArcFace cell]

Second, the **head**. It's a two-layer projection — 1024 to 512 to 256 — with batch norm, ReLU, and 0.3 dropout. The 256-dimensional output is L2-normalised and fed into an **ArcFace** classifier. ArcFace is a metric-learning loss: it takes cross-entropy over *angular* distance on the unit hypersphere, and it adds a fixed angular margin — here, 0.5 radians — between each sample and its correct class centroid, with a scale factor of 64. In plain English: it forces same-identity embeddings to cluster tightly and different identities to separate widely. This is exactly the property we want for retrieval.

I ran a loss comparison — ArcFace, CosFace, sub-center ArcFace, and semi-hard Triplet — with the backbone held fixed. ArcFace won, though the margin over CosFace was within seed noise.

## [4:00 — Scroll to the split cell]

Third, the **evaluation protocol**. This is the part I care most about methodologically. The split at the top of the notebook is **identity-disjoint**: the six validation identities never appear in training. That matches the Kaggle test setting — at test time the model sees individuals it's never been trained on — and it's the only protocol that gives you honest generalisation numbers. A random image-level split would catastrophically overestimate mAP here because the model would memorise individual jaguars.

The split is deterministic, seeded, and persisted — every experiment in the project uses the exact same six validation identities. That's what makes my ablation numbers comparable.

## [5:00 — Scroll to the background loader]

Fourth, the **background trick** — this is the detail that bought me the most Kaggle mAP for the least effort. The Round-2 test images are RGBA, where the alpha channel is a pre-computed jaguar segmentation mask. The *RGB channels are pre-zeroed outside the mask*, meaning the jaguar sits on a pure-black background. But my backbone was trained — and my model fine-tuned — on natural camera-trap images with real backgrounds. That's a domain shift at inference time.

My fix is two lines: I load the RGBA, and wherever alpha equals zero, I fill with neutral gray — value 128. That pulls the test distribution back toward the training distribution and, on Round-2, it moved our public Kaggle mAP by about six points. I validated this wasn't just an artefact by rendering a handful of test images before and after, and by confirming that on Round 1 — where backgrounds are intact — the same intervention *hurts* performance. It's specifically compensating for the Round-2 masking.

## [6:30 — Scroll to training loop]

Training runs for up to 30 epochs with AdamW, learning rate `1e-4`, weight decay `1e-4`, and `ReduceLROnPlateau` on validation mAP. Early stopping with patience 10. On my hardware this converges in about 25 epochs, under two minutes of head training — because the backbone is frozen and I cached its features.

After training, the frozen DINOv2 on its own gives val mAP 0.640. Adding the projection head plus ArcFace takes it to 0.682 — a four-point lift from metric learning. That's the model I check-pointed.

## [7:30 — Scroll to the re-ranking cell]

Last piece: **k-reciprocal re-ranking**, from Zhong et al., CVPR 2017. The idea is: after you compute cosine distances between all test embeddings, you refine them. For each query you find its top-k1 nearest neighbours. Then — and this is the "reciprocal" part — you keep only the neighbours that *also have the query in their own top-k1*. Those mutual neighbours define a local neighbourhood, and you compute a Jaccard distance between neighbourhoods. The final distance is a weighted sum: lambda times the original cosine distance, plus one-minus-lambda times the Jaccard distance.

I tuned (k1, k2, λ) with a separate Bayesian-then-grid search — 130 trials, all on val only, never touching test. The optimum lands at k1 equals 35, k2 equals 6, lambda equals 0.2. Applied to our val embeddings it takes mAP from 0.682 to 0.708 — another two-and-a-half points, free at inference time, no retraining.

## [8:30 — Scroll to final submission cell]

Putting it all together on the Round-2 test set: embed every test image with the gray-fill loader, run it through the trained projection head, build the full test-by-test distance matrix, apply k-reciprocal re-ranking, convert distances back to similarities, and write the submission CSV.

The numbers: frozen DINOv2 alone gives **0.640** val mAP. Adding the projection and ArcFace gets us **0.682**. Re-ranking pushes it to **0.708**. On Kaggle Round-2 public leaderboard, the final submission scores **0.302**, versus the MegaDescriptor-plus-ArcFace baseline at **0.243** — a relative improvement of roughly twenty-four percent.

## [9:30 — Honest limitations]

Two honest caveats. One: my val set has only six identities, so the noise floor is non-trivial — I ran a five-seed repeatability study and the standard deviation of val mAP is about 0.015, so treat sub-one-point differences as ties. Two: this is a late submission, scored but not ranked on the public leaderboard. And three — the Kaggle absolute numbers are lower than val because Round-2 test has many more identities than our six-identity val slice.

That's the notebook. Thanks for watching.
