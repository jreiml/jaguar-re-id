from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingProjection(nn.Module):
    """Projects backbone features to a lower-dimensional re-ID embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArcFaceLayer(nn.Module):
    """Additive Angular Margin (ArcFace, Deng et al. 2019)."""

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(e, w).clamp(-1.0, 1.0)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class ArcFaceModel(nn.Module):
    """Projection head + ArcFace layer on pre-extracted backbone features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        margin: float = 0.5,
        scale: float = 64.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_net = EmbeddingProjection(input_dim, hidden_dim, embedding_dim, dropout)
        self.arcface = ArcFaceLayer(embedding_dim, num_classes, margin=margin, scale=scale)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding_net(x)
        logits = self.arcface(emb, labels)
        return logits, emb

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding_net(x), p=2, dim=1)


def load_backbone(name: str = "hf-hub:BVRA/MegaDescriptor-L-384", device: str = "cuda") -> tuple[nn.Module, int]:
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval().to(device)
    with torch.no_grad():
        dummy_size = int(model.default_cfg.get("input_size", (3, 384, 384))[-1])
        out = model(torch.zeros(1, 3, dummy_size, dummy_size, device=device))
    return model, int(out.shape[1])


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
