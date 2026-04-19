"""Generate a 1-page PDF report summarizing the EDA and Model Training &
Evaluation findings. Output: /code/jaguar-re-id/report.pdf.

No matplotlib dependency — all content is text + a small bar chart drawn
directly in reportlab primitives.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


REPO = Path(__file__).resolve().parents[1]


def _bar_chart(c: canvas.Canvas, x: float, y: float, w: float, h: float, data: list[tuple[str, float]], title: str) -> None:
    max_v = max(v for _, v in data)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x, y + h + 2, title)
    bar_w = w / (len(data) + 1)
    for i, (lbl, v) in enumerate(data):
        bx = x + i * bar_w + 4
        bh = (v / max_v) * h * 0.9
        c.setFillColor(HexColor("#2C5282"))
        c.rect(bx, y, bar_w - 8, bh, stroke=0, fill=1)
        c.setFillColor(black)
        c.setFont("Helvetica", 6.5)
        c.drawString(bx, y - 8, lbl)
        c.drawRightString(bx + bar_w - 10, y + bh + 1, f"{v:.3f}")


def build(out_path: Path) -> Path:
    c = canvas.Canvas(str(out_path), pagesize=A4)
    W, H = A4
    ml = 15 * mm
    mt = 12 * mm
    y = H - mt

    # Title
    c.setFont("Helvetica-Bold", 15)
    c.drawString(ml, y, "Jaguar Re-Identification — 1-page Report")
    y -= 5 * mm
    c.setFont("Helvetica", 8.5)
    c.drawString(ml, y, "Solo track. GitHub: https://github.com/jreiml/jaguar-re-id  |  W&B: wandb.ai/zyna/jaguar-reid-jreiml  |  Kaggle user: jreiml")
    y -= 4 * mm
    c.line(ml, y, W - ml, y); y -= 4 * mm

    # Section 1: EDA
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.5, leading=10.5)

    c.setFont("Helvetica-Bold", 11)
    c.drawString(ml, y, "Exploratory Data Analysis")
    y -= 5 * mm

    eda_text = (
        "<b>Dataset (Kaggle R1 + R2).</b> 1,895 training images across 31 jaguar identities; min/max/median per-identity image count 13/183/45 — "
        "long-tailed. Median image size 3035×1964 (landscape full-body crops). <b>Identity &amp; quality distribution (E1, Q13)</b>: "
        "sharpness spread 50×, brightness mean 97/255 (night imagery); texture features dominate over colour.<br/>"
        "<b>Alpha-channel segmentation (key discovery).</b> Both R1 and R2 PNGs are RGBA — alpha encodes a precomputed jaguar-vs-background "
        "mask. R1 keeps RGB intact (intact backgrounds); R2 pre-zeros RGB in the background region. No trained segmenter needed for Q0/Q26.<br/>"
        "<b>Near-duplicate analysis (E3, Q14).</b> Exact-pHash duplicates: 209 pairs, 100% within-identity (no labelling leak). "
        "Cosine ≥ 0.98 on frozen MegaDescriptor: 555 pairs within-identity, 1 cross. Training set carries substantial burst-frame redundancy "
        "but the duplicates are <i>informative</i> — exact-pHash dedup of 95 images costs −0.034 val mAP (E11) because the near-duplicates "
        "encode small pose/lighting variations the model learns from.<br/>"
        "<b>Background intervention (E4, Q0/Q1) &amp; reliance (E5, Q26).</b> Frozen MegaDescriptor val mAP under different bg fills: "
        "as_is 0.619, blur 0.584, mean 0.509, black 0.493, gray 0.492, noise 0.478. The backbone loses 10–14% mAP when the background is "
        "removed — direct behavioural evidence of context reliance. Blur is the gentlest intervention (preserves low-frequency context); "
        "black (matching R2 test distribution) is near the worst."
    )
    p = Paragraph(eda_text, body)
    w, h = p.wrap(W - 2 * ml, 60 * mm)
    p.drawOn(c, ml, y - h)
    y -= h + 4 * mm

    # Section 2: Model Training & Evaluation
    c.setFont("Helvetica-Bold", 11)
    c.drawString(ml, y, "Model Training & Evaluation")
    y -= 5 * mm

    model_text = (
        "<b>Protocol.</b> All experiments evaluate on a fixed <i>identity-disjoint</i> split (val_v1: 25 train / 6 val identities, 1416 / "
        "479 images). This is strictly harder than the stratified split used in the published baseline — the published 0.741 R1 Kaggle "
        "figure reflects that easier protocol; our honest identity-disjoint calibration point for MegaDescriptor+ArcFace is <b>0.598 val "
        "mAP</b> (E0). Identity-balanced mAP is the primary metric; secondary efficiency metrics (parameter counts) are reported where "
        "relevant.<br/>"
        "<b>Backbone comparison (E2, Q5, 4 backbones, 1.60 credits).</b> Under identical ArcFace + projection recipe: DINOv2-ViT-L/14 "
        "<b>0.669</b> (304M params) &gt; ConvNeXtV2-L-384 0.644 (196M) &gt; MegaDescriptor-L-384 0.598 (195M) &gt; EfficientNetV2-L 0.518 "
        "(117M). Self-supervised DINOv2 at 518-px wins; animal-specialised MegaDescriptor surprisingly loses to a general-purpose "
        "ImageNet-pretrained CNN.<br/>"
        "<b>Loss comparison (E6, Q12, 5 losses, 2.50 credits)</b> on DINOv2: ArcFace 0.682 ≈ CosFace 0.681 ≈ Circle 0.680 ≈ Triplet 0.672 "
        "&gt; SubCenter-ArcFace 0.665. Differences within top group are below the <b>5-seed std of 0.011</b> (E13, Q22, 1.0 credit) — "
        "loss family dominates only SubCenter vs others. Best single-seed run: DINOv2+ArcFace at seed 2024, val mAP <b>0.6945</b>.<br/>"
        "<b>k-reciprocal re-ranking (E7, Q28+Q27, 2.00 credits).</b> Comparing 4 search methods at matched compute: Bayesian (TPE, 40 "
        "trials) + grid refinement around optimum beats grid (25 trials) and random (40 trials) inside the same compute budget, raising "
        "val mAP 0.682 → <b>0.708</b> (+0.025). Optimum at k1≈40, k2=4, λ≈0.05.<br/>"
        "<b>Round 1 vs Round 2 delta (E9, Q30, 1.0 credit).</b> Same Phase-0 baseline checkpoint: R1 Kaggle public 0.478, R2 Kaggle public "
        "0.243. Δ = <b>−0.235</b> — MegaDescriptor is heavily context-dependent; most of the drop is the bg-removal shift (cf. E5). An "
        "R2-targeted submission (DINOv2+ArcFace, bg=gray fill, k-reciprocal rerank) raises R2 score to <b>0.302</b>.<br/>"
        "<b>Summary of credits claimed (9 experiments, sum ≈ 13.6).</b> EDA: E1 (1.0) + E3 (1.0) + E4 (1.0) + E5 (1.5 w/ Q26 bonus) + "
        "E11 (1.0). Leaderboard: E2 (1.60) + E6 (2.50) + E7 (2.00) + E9 (1.00) + E13 (1.00)."
    )
    p2 = Paragraph(model_text, body)
    w2, h2 = p2.wrap(W - 2 * ml, 130 * mm)
    p2.drawOn(c, ml, y - h2)
    y -= h2 + 3 * mm

    # Small bar chart: backbone val mAP
    bar_y = 30 * mm
    _bar_chart(c, ml, bar_y, 85 * mm, 18 * mm,
               [("EffV2-L", 0.5175), ("Mega-L", 0.5976), ("ConvNxtV2-L", 0.6440), ("DINOv2-L", 0.6685)],
               "Val mAP per backbone (E2, 30 epochs, identical head + schedule)")
    _bar_chart(c, W - ml - 85 * mm, bar_y, 85 * mm, 18 * mm,
               [("no-rerank", 0.6822), ("Grid", 0.6901), ("Random", 0.6946), ("Bayes+refine", 0.7076)],
               "E7: k-reciprocal rerank val mAP by search method")

    # Footer
    c.setFont("Helvetica-Oblique", 7.2)
    c.drawString(ml, 10 * mm,
                 "Late submissions: R2 deadline 2026-03-17 passed; instructor approved post-deadline scoring via Kaggle API "
                 "(late submissions are scored but not ranked on public LB).")
    c.drawString(ml, 10 * mm - 3 * mm,
                 "Top models: DINOv2+ArcFace (E13-seed2024 0.6945 val). Best R2 Kaggle score: 0.302 (E6-arcface + bg=gray + k-reciprocal "
                 "k1=35/λ=0.2).")
    c.showPage()
    c.save()
    return out_path


if __name__ == "__main__":
    p = build(REPO / "report.pdf")
    print(f"Wrote {p} ({p.stat().st_size / 1024:.1f} KB)")
