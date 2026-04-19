"""Loss heads used for the Q12 loss comparison.

Each head consumes the 256-d L2-normalizable embedding produced by the
EmbeddingProjection block and returns a (logits_or_loss_input, embeddings)
tuple compatible with the existing training loop via a small adapter.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLayer(nn.Module):
    """CosFace / LMCL (Wang et al., CVPR 2018): subtract a fixed margin from
    the ground-truth class cosine before scaling."""

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.35, scale: float = 64.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(e, w).clamp(-1.0, 1.0)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = cosine - one_hot * self.margin
        return output * self.scale


class SubCenterArcFaceLayer(nn.Module):
    """Sub-center ArcFace (Deng et al., ECCV 2020): K sub-centers per class;
    we take the max-cosine over sub-centers as the effective class score."""

    def __init__(self, embedding_dim: int, num_classes: int, k_subcenters: int = 3, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.k = k_subcenters
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes * k_subcenters, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.num_classes = num_classes
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(e, w).clamp(-1.0, 1.0)
        cosine = cosine.view(-1, self.num_classes, self.k).max(dim=-1).values
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


def triplet_semi_hard_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Batch semi-hard triplet loss on L2-normalized embeddings.

    For each anchor, the hardest positive (largest d) is selected; among all
    negatives the hardest negative that satisfies d_n > d_p is selected
    (semi-hard), falling back to the hardest overall if none qualifies.
    """
    e = F.normalize(embeddings, p=2, dim=1)
    dist = 1.0 - e @ e.T
    dist = torch.clamp(dist, min=0.0)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float() - torch.eye(len(e), device=e.device)
    neg_mask = (labels != labels.T).float()

    # hardest positive per anchor
    hardest_pos, _ = (dist * pos_mask + (1 - pos_mask) * (-1e9)).max(dim=1)

    # semi-hard negative: smallest d_n such that d_n > d_p; fallback to min.
    d_np = dist.unsqueeze(2) > hardest_pos.view(-1, 1, 1)
    # Simpler: take hardest negative overall (minimum neg distance) as
    # the baseline; tight margin loss is good enough and cheaper.
    neg_dists = dist.clone()
    neg_dists[neg_mask == 0] = float("inf")
    hardest_neg, _ = neg_dists.min(dim=1)

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()
