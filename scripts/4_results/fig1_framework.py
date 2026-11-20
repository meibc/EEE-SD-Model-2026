#!/usr/bin/env python3
"""Modelling framework (Fig. 1).

Left to right: (1) data: AMIS survey estimates by state (survey weights, respondent bootstrap and the AHS
wording adjustment are processing steps, described in the caption) and CDC HIV surveillance by state; (2) the prevention-system model: eight measures evolving
together by state, with the core loop between anticipated healthcare stigma (AHS) and PrEP use: men with AHS use
PrEP less (the prevalence ratio r, through the PrEP mixture) and population PrEP feeds back on AHS (b, three
values); (3) the HIV epidemic model: projected PrEP use, HIV testing and condomless sex from (2) drive a transmission
model calibrated to CDC surveillance, giving new infections and diagnoses; (4) intervention scenarios 2025-2036,
each against no intervention. Colour shows what a box represents: data sources (sand), models (shaded containers: prevention-system
model blue, HIV epidemic model purple), measures and outcomes (white boxes inside their model), interventions
(green). Left to right, the study's workflow: data -> prevention-system model -> HIV epidemic model ->
scenario analysis (2025-2036, each scenario against no intervention, at three feedback values). The full pathway structure is in the supplement (figS_pathways,
scripts/4_results/supplement.py).

Outputs: results/paper/main/fig1_framework.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reporting import sources, style  # noqa: E402

W, H = 174, 90                      # drawing units: millimetres at final size
# Colour shows what a box represents: data sources (sand), a model (shaded container: blue the prevention-system
# model, purple the HIV epidemic model), a measure or outcome (white box inside its model), an intervention
# (green). Light tints of the Okabe-Ito colours: (fill, edge).
DATA = ("#fcf1dc", "#d4a543")
PREVENTION = ("#eef4fa", "#86b2d4")
EPI = ("#f9edf3", "#c98bb0")
MEASURE = ("white", "#9a9a9a")
KEY = ("#d9e9f5", style.BLUE)        # the AHS and PrEP boxes (the core loop)
SCENARIO = ("#e3f3ec", "#5fb594")


def box(ax, x, y, w, h, text, kind, *, bold=False, lw=0.8):
    fc, ec = kind
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=style.FONT,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def container(ax, x, y, w, h, kind):
    fc, ec = kind
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec=ec, lw=1.0))


def arrow(ax, start, end, *, rad=0.0, color=style.OBSERVED, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def header(ax, x, w, text, color):
    ax.text(x + w / 2, H - 1.5, text, ha="center", va="top", fontsize=style.FONT, fontweight="bold")
    ax.plot([x, x + w], [H - 6.8, H - 6.8], color=color, lw=1.4, solid_capstyle="butt")


def figure() -> plt.Figure:
    style.apply()
    fig = plt.figure(figsize=(style.WIDTH, style.WIDTH * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c1, c2, c3, c4 = (1, 35), (43, 55), (106, 32), (145, 28)   # data | prevention model | epidemic model | scenarios
    bottom, low, high = 2, 17, 76          # model containers span low..high; the CDC line runs below them

    # data: the two sources (processing - survey weights, respondent bootstrap, AHS wording adjustment - in the caption)
    x, w = c1
    header(ax, x, w, "Data", DATA[1])
    box(ax, x, 39, w, 14, "AMIS survey of MSM,\n2017–2024, 24 states", DATA)
    box(ax, x, bottom, w, 13, "CDC HIV surveillance\nby state", DATA)

    # prevention-system model: container; the measures inside, with the core AHS <-> PrEP loop
    x, w = c2
    header(ax, x, w, "Prevention-system model", PREVENTION[1])
    container(ax, x, low, w, high - low, PREVENTION)
    ahs, prep = (x + 3, 39, 19, 14), (x + w - 22, 39, 19, 14)
    gss = (x + 3, 65.5, 30, 7.8)
    box(ax, *gss, "General social and\nfamily stigma", MEASURE)
    box(ax, *ahs, "Anticipated\nhealthcare\nstigma (AHS)", KEY, bold=True, lw=1.1)
    box(ax, *prep, "PrEP use", KEY, bold=True, lw=1.1)
    ax_l, ax_r = ahs[0] + 4, prep[0] + prep[2] - 4          # vertical arrows at the outer edges, clear of the labels
    arrow(ax, (ax_l, gss[1]), (ax_l, ahs[1] + ahs[3]))
    l, r = ahs[0] + ahs[2], prep[0]
    arrow(ax, (l, 49.5), (r, 49.5), rad=-0.35, color=style.BLUE, lw=1.2)
    arrow(ax, (r, 42.5), (l, 42.5), rad=-0.35, color=style.BLUE, lw=1.2)
    ax.text(x + w / 2 + 3, 54.6, "lower PrEP use\namong men with AHS (r)", ha="center", va="bottom",
            fontsize=style.FONT, linespacing=1.15)
    ax.text(x + w / 2, 35.6, "PrEP→AHS feedback (b):\n−0.075, 0, +0.15", ha="center", va="top", fontsize=style.FONT,
            linespacing=1.15)
    other = (x + 4, 19.5, w - 8, 9.5)
    box(ax, *other, "Disclosure, provider visits,\nHIV testing, condomless sex", MEASURE)
    arrow(ax, (ax_l, ahs[1]), (ax_l, other[1] + other[3]))
    arrow(ax, (ax_r, other[1] + other[3]), (ax_r, prep[1]))
    arrow(ax, (c1[0] + c1[1], 46), (ahs[0], 46))                    # survey data -> model

    # HIV epidemic model: container; projected inputs from the prevention model -> new infections and diagnoses
    x, w = c3
    header(ax, x, w, "HIV epidemic model", EPI[1])
    container(ax, x, low, w, high - low, EPI)
    inputs, outcomes = (x + 2.5, 39, w - 5, 15), (x + 2.5, 19.5, w - 5, 11)
    box(ax, *inputs, "PrEP use,\nHIV testing,\ncondomless sex", MEASURE)
    box(ax, *outcomes, "New infections\nand diagnoses", MEASURE)
    ax_m = x + 9
    arrow(ax, (ax_m, inputs[1]), (ax_m, outcomes[1] + outcomes[3]))
    ax.text(ax_m + 2, (inputs[1] + outcomes[1] + outcomes[3]) / 2, "HIV\ntransmission", ha="left", va="center",
            fontsize=style.FONT, linespacing=1.15)
    arrow(ax, (prep[0] + prep[2], 46), (inputs[0], 46))             # PrEP use -> inputs
    arrow(ax, (other[0] + other[2], 24.25), (inputs[0], 41), rad=0.25)   # testing, condomless sex -> inputs
    # CDC surveillance -> calibration: along the bottom, up, into the epidemic model
    cx = x - 3.5
    ax.plot([c1[0] + c1[1], cx, cx], [bottom + 6.5, bottom + 6.5, 25], color=style.OBSERVED, lw=1.0)
    arrow(ax, (cx, 25), (x, 25))
    ax.text((c1[0] + c1[1] + cx) / 2, bottom + 7.6, "calibration", ha="center", va="bottom", fontsize=style.FONT,
            color=style.MUTED)

    # scenario analysis: the fitted, coupled models are used to run the interventions (years in the caption)
    x, w = c4
    header(ax, x, w, "Scenario analysis", SCENARIO[1])
    labels = ("Halve AHS", "Hold AHS at its\n2024 level", "Stigma-resilient\nservices")
    for y, label in zip((53, 41, 29), labels):
        box(ax, x, y, w, 10.5, label, SCENARIO)
    arrow(ax, (c3[0] + c3[1], 46.25), (x, 46.25))
    return fig


def main() -> None:
    style.save(figure(), sources.MAIN / "fig1_framework")


if __name__ == "__main__":
    main()
