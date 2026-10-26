"""Figure style shared by the results scripts. Colors are Okabe-Ito (distinguishable under common
color-vision deficiencies); the two projection assumptions also differ in marker shape, so no figure
relies on color alone."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e5e5e5"
OBSERVED = "#4d4d4d"
CONTINUED = "#0072B2"       # fitted PrEP -> AHS relationship continued (primary)
NOT_CONTINUED = "#E69F00"   # not continued
ACCENT = "#009E73"
MARKERS = {"continued": "o", "not_continued": "s"}
COLORS = {"continued": CONTINUED, "not_continued": NOT_CONTINUED}
LABELS = {"continued": "Fitted PrEP–AHS relationship continued",
          "not_continued": "Fitted PrEP–AHS relationship not continued"}

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
