#!/usr/bin/env python3
"""Regenerate the current manuscript figure set from saved plotting inputs.

This script is intentionally a *figure regeneration* script, not a model-fitting
or data-preparation script. It reads the canonical CSVs prepared by
`98_prepare_paper_figure_inputs.py` and rebuilds the five manuscript figures so
labels, colors, figure sizes, and panel layouts can be edited in one place.

Run from EEE-SD-Model-2026:

    python paper_figure_scripts/98_prepare_paper_figure_inputs.py --overwrite
    python paper_figure_scripts/99_regenerate_paper_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


ROBUST_REGION_ORDER = ["Northeast", "Midwest", "South", "West"]

VARIABLE_ORDER = [
    "stigma_ahs",
    "stigma_gss",
    "stigma_family",
    "out_gid",
    "seehcp",
    "risk_behavior",
    "prep_used",
    "hivtest12",
]

VARIABLE_LABELS = {
    "stigma_ahs": "Anticipated\nhealthcare\nstigma",
    "stigma_gss": "General\nsocial\nstigma",
    "stigma_family": "Family\nstigma",
    "out_gid": "Out to\nhealthcare\nprovider",
    "seehcp": "Saw\nhealthcare\nprovider",
    "risk_behavior": "Risk\nbehavior",
    "prep_used": "PrEP\nuse",
    "hivtest12": "HIV\ntesting",
}

PATHWAY_ORDER = [
    "AHS → PrEP use",
    "AHS → HIV testing",
    "AHS → disclosure",
    "Disclosure → PrEP use",
    "Disclosure → HIV testing",
    "Saw HCP → AHS",
    "Saw HCP → HIV testing",
    "Saw HCP → PrEP use",
]

PATHWAY_LABELS = {
    "AHS → PrEP use": "AHS → PrEP use",
    "AHS → HIV testing": "AHS → HIV testing",
    "AHS → disclosure": "AHS → disclosure",
    "Disclosure → PrEP use": "Disclosure → PrEP use",
    "Disclosure → HIV testing": "Disclosure → HIV testing",
    "Saw HCP → AHS": "Saw HCP → AHS",
    "Saw HCP → HIV testing": "Saw HCP → HIV testing",
    "Saw HCP → PrEP use": "Saw HCP → PrEP use",
}

PATHWAY_COLORS = {
    "AHS to prevention": "#4C72B0",
    "AHS to disclosure": "#8172B2",
    "Disclosure to prevention": "#DD8452",
    "Care-stigma feedback": "#55A868",
    "Care to prevention": "#8C6D31",
}

REGION_COLORS = {
    "Northeast": "#4C78A8",
    "Midwest": "#72B7B2",
    "South": "#F58518",
    "West": "#54A24B",
}

SCENARIO_ORDER = [
    "s7_combined_stigma_package",
    "s1_reduce_ahs",
    "s4_weaken_ahs_to_prep",
]

SCENARIO_LABELS = {
    "s7_combined_stigma_package": "Combined stigma package",
    "s1_reduce_ahs": "Reduce AHS",
    "s4_weaken_ahs_to_prep": "Weaken AHS → PrEP barrier",
}

SCENARIO_COLORS = {
    "s7_combined_stigma_package": "#6F51A5",
    "s1_reduce_ahs": "#4E79A7",
    "s4_weaken_ahs_to_prep": "#C77C2E",
}

PROFILE_COLORS = {
    "layered stigma action": "#7B3FB2",
    "proximal healthcare stigma action": "#4C78A8",
    "pressure / low translation": "#59A14F",
    "low detectable aggregate stigma leverage": "#7F7F7F",
}

POP_COLORS = {
    "Black, non-Hispanic": "#6F4BA1",
    "Hispanic/Latino": "#CC64B0",
    "White, non-Hispanic": "#4E79A7",
    "<=24": "#F28E2B",
    "25-34": "#59A14F",
    "35+": "#E15759",
}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure1(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "figure1_observed_change_heatmap_data.csv"
    df = pd.read_csv(src)
    df = df[df["variable"].isin(VARIABLE_ORDER)].copy()

    state_order = []
    region_for_state = {}
    for region in ROBUST_REGION_ORDER:
        states = sorted(df.loc[df["region"] == region, "state"].unique())
        state_order.extend(states)
        region_for_state.update({s: region for s in states})

    pivot = (
        df.pivot_table(index="state", columns="variable", values="change_pp")
        .reindex(index=state_order, columns=VARIABLE_ORDER)
    )
    median_row = pivot.median(axis=0)
    plot_df = pd.concat([pivot, pd.DataFrame([median_row], index=["Median"])])

    cmap = LinearSegmentedColormap.from_list(
        "green_white_purple", ["#006D2C", "#F7F7F7", "#762A83"], N=256
    )
    norm = TwoSlopeNorm(vcenter=0, vmin=-30, vmax=30)

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    im = ax.imshow(plot_df.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(VARIABLE_ORDER)))
    ax.set_xticklabels([VARIABLE_LABELS[v] for v in VARIABLE_ORDER], fontsize=11)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=11)
    ax.tick_params(length=0)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.iloc[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8.5)

    # Construct/domain dividers.
    for x in [2.5, 4.5]:
        ax.axvline(x, color="black", lw=1.4)

    # Region dividers and labels.
    start = 0
    for region in ROBUST_REGION_ORDER:
        n = sum(region_for_state.get(s) == region for s in state_order)
        if n == 0:
            continue
        end = start + n
        ax.axhline(end - 0.5, color="black", lw=1.0)
        ax.text(
            -1.15,
            start + (n - 1) / 2,
            region,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
        start = end
    ax.axhline(len(plot_df.index) - 1.5, color="black", lw=1.4)

    # Header labels.
    header_y = -1.95
    ax.text(1, header_y, "Stigma context", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(
        3.5,
        header_y,
        "Disclosure / care engagement",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(6, header_y, "Prevention / behavior", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.axhline(-1.35, color="black", lw=1.2, clip_on=False)

    ax.set_title(
        "Observed percentage-point change among robust states, 2018–2024",
        fontsize=15,
        pad=34,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label(
        "Percentage-point change\n(green = decrease, purple = increase)",
        fontsize=11,
    )
    ax.text(
        0.01,
        -0.13,
        "Rows are grouped by U.S. Census region and ordered alphabetically within region. "
        "The median row summarizes the 17 robust states.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    save_figure(fig, output_dir, "figure1A_change_heatmap_region_grouped_neutral_colors")


def _apply_panel_a_label_offsets(row: pd.Series) -> tuple[float, float]:
    offsets = {
        "FL": (0.000, 0.026),
        "MI": (-0.012, 0.004),
        "WA": (0.010, 0.006),
        "MA": (0.010, 0.002),
        "NC": (0.006, -0.014),
        "GA": (0.008, -0.003),
    }
    return offsets.get(str(row["state"]), (0.006, 0.006))


def plot_figure2(input_dir: Path, output_dir: Path) -> None:
    scatter_src = input_dir / "figure2_structural_position_scores.csv"
    pathways_src = input_dir / "figure2_pathway_heterogeneity_data.csv"
    scatter = pd.read_csv(scatter_src)
    pathways = pd.read_csv(pathways_src)
    pathways = pathways[pathways["pathway"].isin(PATHWAY_ORDER)].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.45], "wspace": 0.30},
    )
    ax = axes[0]
    for region, sub in scatter.groupby("region"):
        ax.scatter(
            sub["x"],
            sub["y"],
            s=34,
            color=REGION_COLORS.get(region, "#777777"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.90,
            label=region,
        )
    for _, row in scatter.iterrows():
        dx, dy = _apply_panel_a_label_offsets(row)
        ax.text(row["x"] + dx, row["y"] + dy, row["state"], fontsize=7, color="#444444")
    ax.axvline(scatter["x"].median(), color="#BBBBBB", lw=1, ls="--")
    ax.axhline(scatter["y"].median(), color="#BBBBBB", lw=1, ls="--")
    ax.set_xlabel("Proximal anticipated healthcare stigma leverage score", fontsize=9)
    ax.set_ylabel("Cross-context stigma reinforcement score", fontsize=9)
    ax.set_title("A. Structural pathway position", loc="left", fontsize=10, fontweight="bold")
    ax.grid(True, color="#E0E0E0", lw=0.6, alpha=0.7)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    pathway_to_y = {p: i for i, p in enumerate(PATHWAY_ORDER[::-1])}
    rng = np.random.default_rng(12)
    for p, sub in pathways.groupby("pathway"):
        y = pathway_to_y[p]
        family = sub["pathway_family"].iloc[0]
        color = PATHWAY_COLORS.get(family, "#666666")
        jitter = rng.uniform(-0.055, 0.055, size=len(sub))
        ax.scatter(
            sub["p50"],
            y + jitter,
            s=8,
            color=color,
            alpha=0.35,
            edgecolor="none",
        )
        med = sub["p50"].median()
        q25 = sub["p50"].quantile(0.25)
        q75 = sub["p50"].quantile(0.75)
        ax.plot([q25, q75], [y, y], color=color, lw=3.0, alpha=0.75, solid_capstyle="round")
        ax.scatter([med], [y], s=28, color=color, zorder=4, edgecolor="white", linewidth=0.5)

    for y in range(len(PATHWAY_ORDER)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F3F3F3", zorder=-5)
    ax.axvline(0, color="#9E9E9E", lw=1.0, ls="--")
    ax.set_yticks(range(len(PATHWAY_ORDER)))
    ax.set_yticklabels([PATHWAY_LABELS[p] for p in PATHWAY_ORDER[::-1]], fontsize=8)
    ax.set_xlabel("Modeled pathway coefficient (J matrix, centered-logit scale)", fontsize=9)
    ax.set_title("B. Key pathway strengths across robust states", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#E0E0E0", lw=0.6, alpha=0.7)
    ax.tick_params(axis="x", labelsize=8)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="none", markersize=5, label=k)
        for k, c in PATHWAY_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        title="Pathway family",
        loc="lower left",
        fontsize=6.5,
        title_fontsize=6.5,
        frameon=False,
    )

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.19)
    fig.text(
        0.02,
        0.03,
        "AHS = anticipated healthcare stigma; HCP = healthcare provider. "
        "Panel B small dots are states; large dots and thick segments show cross-state medians and IQRs.",
        fontsize=7,
        ha="left",
    )
    save_figure(fig, output_dir, "figure2_two_panel_structure_scatter_and_pathways_v8_fl_vertical")


def plot_figure3(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "figure3_scenario_response_long.csv"
    df = pd.read_csv(src)
    df = df[df["scenario"].isin(SCENARIO_ORDER)].copy()
    combined = df[df["scenario"] == "s7_combined_stigma_package"].set_index("state")["p50"]
    states = combined.sort_values().index.tolist()
    y_base = {s: i for i, s in enumerate(states)}
    offsets = {
        "s7_combined_stigma_package": -0.22,
        "s1_reduce_ahs": 0.00,
        "s4_weaken_ahs_to_prep": 0.22,
    }

    fig, ax = plt.subplots(figsize=(6.2, 8.8))
    for i, state in enumerate(states):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F5F5F5", zorder=-10)

    for scenario in SCENARIO_ORDER:
        sub = df[df["scenario"] == scenario].copy()
        sub["y"] = sub["state"].map(y_base) + offsets[scenario]
        color = SCENARIO_COLORS[scenario]
        ax.hlines(sub["y"], sub["p25"], sub["p75"], color=color, lw=1.6, alpha=0.35)
        ax.scatter(sub["p50"], sub["y"], s=18, color=color, edgecolor="white", linewidth=0.4, label=SCENARIO_LABELS[scenario], zorder=5)

    ax.axvline(0, color="#B8B8B8", lw=1.0, ls="--")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Change in projected 2036 HIV incidence\nrelative to baseline (%)", fontsize=10)
    ax.set_ylabel("State", fontsize=10)
    ax.set_title("Projected incidence responses to\nstigma-related scenarios", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#E0E0E0", lw=0.7, alpha=0.7)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.02,
        0.02,
        "Dots show median Monte Carlo scenario contrasts; lighter horizontal bars show interquartile ranges. "
        "AHS = anticipated healthcare stigma.",
        fontsize=7,
        ha="left",
    )
    fig.subplots_adjust(left=0.16, bottom=0.12)
    save_figure(fig, output_dir, "figure3_final_single_panel_iqr_tall_legend_inside_reordered")


def _format_profile(profile: str) -> str:
    mapping = {
        "layered stigma action": "Layered stigma action",
        "proximal healthcare stigma action": "Healthcare-stigma action",
        "pressure / low translation": "Pressure / low translation",
        "low detectable aggregate stigma leverage": "Low aggregate leverage",
    }
    return mapping.get(profile, profile)


def plot_figure4(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "figure4_structural_alignment_3d_data.csv"
    df = pd.read_csv(src)
    zcol = "scenario_translation_magnitude"
    z_floor = 0.0

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=24, azim=-56)

    for profile, sub in df.groupby("integrated_policy_profile_primary"):
        color = PROFILE_COLORS.get(profile, "#777777")
        label = _format_profile(profile)
        ax.scatter(
            sub["proximal_score_p50"],
            sub["cross_score_p50"],
            sub[zcol],
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            depthshade=False,
        )
        ax.scatter(
            sub["proximal_score_p50"],
            sub["cross_score_p50"],
            np.full(len(sub), z_floor),
            s=38,
            color=color,
            alpha=0.16,
            edgecolor="none",
            depthshade=False,
        )
        for _, row in sub.iterrows():
            ax.plot(
                [row["proximal_score_p50"], row["proximal_score_p50"]],
                [row["cross_score_p50"], row["cross_score_p50"]],
                [z_floor, row[zcol]],
                color=color,
                alpha=0.22,
                lw=1.0,
            )

    label_offsets = {
        "WA": (0.018, 0.000, 0.2),
        "FL": (0.004, -0.010, 0.5),
        "OH": (-0.018, -0.006, 0.5),
    }
    for _, row in df.iterrows():
        dx, dy, dz = label_offsets.get(row["state"], (0.006, 0.004, 0.8))
        ax.text(
            row["proximal_score_p50"] + dx,
            row["cross_score_p50"] + dy,
            row[zcol] + dz,
            row["state"],
            fontsize=7,
            color="#333333",
            zorder=10,
        )

    ax.set_xlabel("Proximal AHS leverage", labelpad=8)
    ax.set_ylabel("Cross-context reinforcement", labelpad=9)
    ax.set_zlabel("Scenario response (%)", labelpad=7)
    ax.set_zlim(0, max(52, df[zcol].max() * 1.08))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.02), frameon=False, fontsize=8)
    save_figure(fig, output_dir, "figure4_three_dimension_structural_alignment_3d_shadow_label_auto_wa_right")


def plot_figure5(input_dir: Path, output_dir: Path) -> None:
    panel_a_src = input_dir / "figure5_priority_population_national_panel_data.csv"
    panel_b_src = input_dir / "figure5_california_bridge_panel_data.csv"
    ref_src = input_dir / "figure5_state_reference_panel_data.csv"
    a = pd.read_csv(panel_a_src)
    b = pd.read_csv(panel_b_src)
    ref = pd.read_csv(ref_src)

    pathways = [
        "AHS → disclosure",
        "AHS → PrEP use",
        "Disclosure → PrEP use",
        "Disclosure → HIV testing",
        "Saw HCP → AHS",
        "Saw HCP → HIV testing",
    ]
    y_map = {p: i for i, p in enumerate(pathways[::-1])}
    pop_order = [
        "Black, non-Hispanic",
        "Hispanic/Latino",
        "White, non-Hispanic",
        "<=24",
        "25-34",
        "35+",
    ]
    pop_offsets = {
        "Black, non-Hispanic": -0.20,
        "Hispanic/Latino": -0.12,
        "White, non-Hispanic": -0.04,
        "<=24": 0.04,
        "25-34": 0.12,
        "35+": 0.20,
    }
    group_offsets = {
        "California aggregate": -0.12,
        "California Hispanic/Latino": 0.12,
    }
    group_colors = {
        "California aggregate": "#2C7FB8",
        "California Hispanic/Latino": "#CC64B0",
    }
    group_markers = {
        "California aggregate": "s",
        "California Hispanic/Latino": "o",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.7), gridspec_kw={"width_ratios": [1.50, 1.05], "wspace": 0.30})

    ax = axes[0]
    for y in range(len(pathways)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F5F5F5", zorder=-10)
    for pop in pop_order:
        sub = a[(a["population_label"] == pop) & (a["pathway"].isin(pathways))].copy()
        sub["y"] = sub["pathway"].map(y_map) + pop_offsets[pop]
        color = POP_COLORS[pop]
        ax.hlines(sub["y"], sub["p25"], sub["p75"], color=color, lw=2.0, alpha=0.45)
        ax.scatter(sub["p50"], sub["y"], color=color, s=38, edgecolor="white", linewidth=0.5, label=pop, zorder=5)
    ax.axvline(0, color="#8A8A8A", lw=1.0, ls="--")
    ax.set_title("A. National demographic-group pathway patterns", loc="left", fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(pathways)))
    ax.set_yticklabels([PATHWAY_LABELS.get(p, p) for p in pathways[::-1]], fontsize=11)
    ax.set_xlabel("Fitted pathway coefficient (centered-logit scale)", fontsize=10)
    ax.grid(axis="x", color="#E0E0E0", lw=0.7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.52, -0.22), ncol=3, frameon=False, fontsize=9)

    ax = axes[1]
    for y in range(len(pathways)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F5F5F5", zorder=-10)
    for _, row in ref.iterrows():
        y = y_map[row["pathway"]]
        ax.hlines(y, row["state_q25"], row["state_q75"], color="#CFCFCF", lw=5.0, alpha=0.75, zorder=1)
        ax.scatter(row["state_median"], y, marker="D", s=44, color="#6F6F6F", edgecolor="white", linewidth=0.5, zorder=4)
    for group, sub in b[b["pathway"].isin(pathways)].groupby("group"):
        sub = sub.copy()
        sub["y"] = sub["pathway"].map(y_map) + group_offsets[group]
        color = group_colors[group]
        ax.hlines(sub["y"], sub["p25"], sub["p75"], color=color, lw=2.4, alpha=0.48)
        ax.scatter(
            sub["p50"],
            sub["y"],
            marker=group_markers[group],
            color=color,
            s=48,
            edgecolor="white",
            linewidth=0.6,
            label=group,
            zorder=5,
        )
    ax.scatter([], [], marker="D", s=44, color="#6F6F6F", label="Robust-state median")
    ax.plot([], [], color="#CFCFCF", lw=5.0, label="Robust-state IQR")
    ax.axvline(0, color="#8A8A8A", lw=1.0, ls="--")
    ax.set_title("B. California Hispanic/Latino bridge case", loc="left", fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(pathways)))
    ax.set_yticklabels([PATHWAY_LABELS.get(p, p) for p in pathways[::-1]], fontsize=11)
    ax.set_xlabel("Fitted pathway coefficient (centered-logit scale)", fontsize=10)
    ax.grid(axis="x", color="#E0E0E0", lw=0.7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.52, -0.22), ncol=2, frameon=False, fontsize=9)

    for ax in axes:
        ax.set_xlim(-0.42, 0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, output_dir, "figure5_priority_population_bridge_v7_corrected_wide_singleline")


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(description="Regenerate final manuscript figures from plotting CSV inputs.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "output/results_refinement/final_figure_inputs",
        help="Prepared plotting-input directory created by 98_prepare_paper_figure_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output/results_refinement/final_figure_regeneration",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        choices=["1", "2", "3", "4", "5"],
        help="Subset of figures to regenerate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if "1" in args.figures:
        plot_figure1(input_dir, output_dir)
    if "2" in args.figures:
        plot_figure2(input_dir, output_dir)
    if "3" in args.figures:
        plot_figure3(input_dir, output_dir)
    if "4" in args.figures:
        plot_figure4(input_dir, output_dir)
    if "5" in args.figures:
        plot_figure5(input_dir, output_dir)

    print(f"Regenerated figures in: {output_dir}")


if __name__ == "__main__":
    main()
