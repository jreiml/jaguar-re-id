"""E15: GradCAM interpretability for ConvNeXtV2 re-ID embeddings (Q2/Q16).

Generates GradCAM heatmaps for the best ConvNeXtV2 ArcFace embeddings and
runs two mandatory sanity-and-faithfulness tests required by Q2:

  1. **Randomization sanity check** — re-run GradCAM with a randomly
     re-initialised projection head. Explanation maps should degrade
     (lose structure / become noise-like). We score this by the
     correlation between the trained and random maps: near-zero correlation
     means the explanations are actually using the trained weights.

  2. **Masking faithfulness test** — use the GradCAM map to pick the top-k%
     "important" pixels, mask them, and re-embed. Measure similarity drop
     to the un-masked embedding for the same query, and compare against
     masking a same-sized **random** region as a control. If GradCAM is
     faithful, the targeted-mask drop is larger than the random-mask drop.

We use ConvNeXtV2 (not DINOv2) because GradCAM is defined via the last
conv layer's activations + gradients, and ConvNeXtV2's stem is natively
convolutional. DINOv2 (ViT) would need an attention-based variant (AttnLRP),
a deeper follow-up.

The GradCAM target is the L2-norm of the projected 256-d embedding — a
scalar that reflects how "confident" the model is about the image in its
ReID space. Its gradient w.r.t. the last conv feature map gives the
importance weights used to weight the feature channels, which are then
spatially average-pooled and ReLU'd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import timm

from ..data import IdentityDisjointSplit, load_train_df, train_images_dir, IMAGENET_MEAN, IMAGENET_STD
from ..model import EmbeddingProjection
from ..paths import CHECKPOINTS, KAGGLE_R1, LOGS, SPLITS


def _load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone = timm.create_model(cfg["backbone"], pretrained=True, num_classes=0).eval().to(device)
    feat_dim = int(ckpt["backbone_feature_dim"])
    projection = EmbeddingProjection(feat_dim, int(cfg["hidden_dim"]), int(cfg["embedding_dim"]), float(cfg["dropout"])).to(device)
    # E2-style checkpoint stores the full ArcFaceModel; E6 stores projection
    # separately. Support both by filtering keys.
    if "projection_state_dict" in ckpt:
        projection.load_state_dict(ckpt["projection_state_dict"])
    else:
        msd = ckpt["model_state_dict"]
        proj_sd = {k.replace("embedding_net.", ""): v for k, v in msd.items() if k.startswith("embedding_net.")}
        projection.load_state_dict(proj_sd)
    projection.eval()
    return backbone, projection, cfg, feat_dim


def _gradcam_map(backbone: torch.nn.Module, projection: torch.nn.Module, image: torch.Tensor, device: str) -> np.ndarray:
    """Return a (H', W') GradCAM map. Image: (1, 3, H, W) on device.

    Target: squared L2 norm of the UNNORMALIZED projection output. This has
    a non-trivial gradient w.r.t. the backbone feature map, unlike the
    normalized embedding whose squared norm is a constant 1.
    """
    feat = backbone.forward_features(image)
    feat.requires_grad_(True)
    pooled = feat.mean(dim=[2, 3])
    emb = projection(pooled)  # NOT normalized
    score = (emb ** 2).sum()
    grads = torch.autograd.grad(score, feat, create_graph=False)[0]
    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    denom = cam.amax(dim=[2, 3], keepdim=True) + 1e-9
    cam = cam / denom
    return cam.squeeze().detach().cpu().numpy()


def _embed(backbone, projection, image):
    with torch.no_grad():
        feat = backbone.forward_features(image)
        pooled = feat.mean(dim=[2, 3])
        emb = F.normalize(projection(pooled), 2, 1)
    return emb


def _resize_cam(cam: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.from_numpy(cam)[None, None]
    t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def _masked_image(image: torch.Tensor, mask: np.ndarray, *, fill: float = 0.0) -> torch.Tensor:
    """image: (1, 3, H, W), already ImageNet-normalized. mask: (H, W) boolean,
    True = mask out. Fill=0.0 in normalized space corresponds to the ImageNet
    mean grey (a neutral mask, not "black").
    """
    out = image.clone()
    m = torch.from_numpy(mask).to(out.device)
    for c in range(out.shape[1]):
        out[0, c][m] = fill
    return out


def run(checkpoint: str, n_samples: int = 20, topk_frac: float = 0.2) -> dict:
    device = "cuda"
    split = IdentityDisjointSplit.load(SPLITS / "val_v1.json")
    imgs_dir = train_images_dir(KAGGLE_R1)
    sample_fns = split.val_filenames[:n_samples]

    backbone, projection, cfg, feat_dim = _load_checkpoint(Path(checkpoint), device)

    # Randomization sanity check: randomize BOTH the backbone (= the network
    # whose activations+gradients produce the GradCAM) AND the projection
    # head. This is the standard Q2 sanity baseline. Re-initialise the
    # backbone to random weights by creating a fresh timm model with
    # pretrained=False; re-initialise the projection.
    rand_backbone = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0).eval().to(device)
    rand_proj = EmbeddingProjection(feat_dim, int(cfg["hidden_dim"]), int(cfg["embedding_dim"]), float(cfg["dropout"])).to(device)
    rand_proj.eval()

    preprocess = transforms.Compose([
        transforms.Resize((int(cfg["input_size"]), int(cfg["input_size"]))),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    corrs = []  # sanity check
    faith_targeted = []
    faith_random = []
    rng = np.random.default_rng(0)

    for fn in sample_fns:
        img = Image.open(imgs_dir / fn).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        H, W = int(cfg["input_size"]), int(cfg["input_size"])

        cam_trained = _gradcam_map(backbone, projection, x, device)
        cam_trained = _resize_cam(cam_trained, (H, W))
        cam_random = _gradcam_map(rand_backbone, rand_proj, x, device)
        cam_random = _resize_cam(cam_random, (H, W))
        # Sanity-check correlation.
        corr = float(np.corrcoef(cam_trained.flatten(), cam_random.flatten())[0, 1])
        if np.isnan(corr): corr = 0.0
        corrs.append(corr)

        # Faithfulness: mask top-k% of GradCAM vs a random mask of same size.
        n_pixels = H * W
        k = int(topk_frac * n_pixels)
        order = np.argsort(-cam_trained.flatten())
        top_mask = np.zeros(n_pixels, dtype=bool); top_mask[order[:k]] = True
        top_mask = top_mask.reshape(H, W)
        rand_mask = np.zeros(n_pixels, dtype=bool); rand_mask[rng.choice(n_pixels, k, replace=False)] = True
        rand_mask = rand_mask.reshape(H, W)

        emb_ref = _embed(backbone, projection, x)
        emb_top = _embed(backbone, projection, _masked_image(x, top_mask))
        emb_rand = _embed(backbone, projection, _masked_image(x, rand_mask))

        drop_top = 1.0 - float((emb_ref * emb_top).sum().item())
        drop_rand = 1.0 - float((emb_ref * emb_rand).sum().item())
        faith_targeted.append(drop_top)
        faith_random.append(drop_rand)

    results = {
        "checkpoint": checkpoint,
        "n_samples": n_samples,
        "topk_frac": topk_frac,
        "sanity_correlation_trained_vs_random": {
            "mean": float(np.mean(corrs)),
            "std": float(np.std(corrs)),
            "min": float(np.min(corrs)),
            "max": float(np.max(corrs)),
        },
        "faithfulness_similarity_drop": {
            "targeted_top": {"mean": float(np.mean(faith_targeted)), "std": float(np.std(faith_targeted))},
            "random_mask": {"mean": float(np.mean(faith_random)), "std": float(np.std(faith_random))},
            "targeted_minus_random": float(np.mean(faith_targeted) - np.mean(faith_random)),
        },
    }
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "exp_E15_gradcam.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/E2-convnextv2-large.pth")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--topk-frac", type=float, default=0.2)
    args = p.parse_args()
    run(args.checkpoint, n_samples=args.n_samples, topk_frac=args.topk_frac)
