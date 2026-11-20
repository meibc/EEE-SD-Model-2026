"""Figure style shared by the results scripts. Colors are Okabe-Ito (distinguishable under common
color-vision deficiencies); the feedback values also differ in marker shape, so no figure relies on color
alone."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e5e5e5"
OBSERVED = "#4d4d4d"
BLUE, ORANGE, GREEN, VERMILLION = "#0072B2", "#E69F00", "#009E73", "#D55E00"
ACCENT = GREEN
# PrEP -> AHS feedback values (reporting/sources.py FEEDBACK), keyed as there
COLORS = {"neg075": GREEN, "0": BLUE, "pos15": VERMILLION}
MARKERS = {"neg075": "v", "0": "o", "pos15": "^"}
LABELS = {"neg075": "Feedback (−0.075)", "0": "No feedback", "pos15": "Feedback (+0.15)"}   # meanings in the captions
SHORT_LABELS = {"neg075": "−0.075", "0": "0", "pos15": "+0.15"}
UNADJUSTED = "#CC79A7"       # AHS not adjusted for the 2022 item change (supplement)

MEASURE_LABELS = {
    "stigma_ahs": "Anticipated healthcare stigma", "stigma_gss": "General social stigma",
    "stigma_family": "Family stigma", "out_gid": "Disclosure to provider", "seehcp": "Healthcare provider visit (past year)",
    "risk_behavior": "Condomless anal sex with a partner of positive or unknown HIV status", "prep_used": "Past-year PrEP use", "hivtest12": "HIV testing",
}


FONT = 8              # every text element, pt (AIDS and Behavior: 8-12 pt at final size)
WIDTH = 174 / 25.4    # full-width figure, inches (Springer double column 174 mm; Nature 183 mm)


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
        "ytick.labelsize": FONT, "legend.fontsize": FONT, "legend.title_fontsize": FONT, "figure.titlesize": FONT,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embedded TrueType: text stays editable in the PDF
    })


def save(fig, path: Path) -> None:
    """Save as PNG (review) and PDF (manuscript)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
