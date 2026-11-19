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
LABELS = {"neg075": "Feedback −0.075 (affirming care)", "0": "No feedback (reference)",
          "pos15": "Feedback +0.15 (best fit)"}
SHORT_LABELS = {"neg075": "−0.075", "0": "0", "pos15": "+0.15"}
UNADJUSTED = "#CC79A7"       # AHS not adjusted for the 2022 item change (supplement)

MEASURE_LABELS = {
    "stigma_ahs": "Anticipated healthcare stigma", "stigma_gss": "General social stigma",
    "stigma_family": "Family stigma", "out_gid": "Disclosure to provider", "seehcp": "Healthcare provider visit (past year)",
    "risk_behavior": "HIV-related risk behavior", "prep_used": "Past-year PrEP use", "hivtest12": "HIV testing",
}


def apply() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path: Path) -> None:
    """Save as PNG (review) and PDF (manuscript)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
