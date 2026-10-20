from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/supplementary_figures"
OUT.mkdir(parents=True, exist_ok=True)
ROBUST = ["CA", "FL", "GA", "IL", "IN", "MA", "MD", "MI", "NC", "NJ", "NY", "OH", "PA", "TN", "TX", "VA", "WA"]

core_dir = ROOT / "output/final_direct_bootstrap_core_ablation_n100"
cov_dir = ROOT / "output/final_direct_bootstrap_covariance_gradient_n100"
prior_dir = ROOT / "output/final_direct_bootstrap_prior_regularization_n100"

core_summary = pd.read_csv(core_dir / "ablation_summary.csv").set_index("variant")
cov_summary = pd.read_csv(cov_dir / "ablation_summary.csv").set_index("variant")
prior_summary = pd.read_csv(prior_dir / "ablation_summary.csv").set_index("variant")
fixed = pd.read_csv(
    ROOT / "output/amis_joint_bootstrap_direct_v2_n500/absolute_stigma_reduction_direct_bootstrap_n500/comparison_metrics.csv"
)


def load_fit(path):
    z = np.load(path, allow_pickle=True)
    return {
        "J": np.asarray(z["Jmeans_stack"], dtype=float),
        "units": np.asarray(z["unit_order"]).astype(str).tolist(),
        "vars": np.asarray(z["v_names"]).astype(str).tolist(),
    }


def edge_indices(v_names):
    priors = pd.read_csv(Path("/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/config/sem_structural_priors_v3.csv"))
    return [(v_names.index(r.target), v_names.index(r.source), f"{r.source}_to_{r.target}") for _, r in priors.iterrows()]


def within_pathway_rank_summary(folder, variant):
    primary = load_fit(folder / "fits/primary.npz")
    alt = load_fit(folder / f"fits/{variant}.npz")
    pidx = [primary["units"].index(s) for s in ROBUST]
    aidx = [alt["units"].index(s) for s in ROBUST]
    edges = edge_indices(primary["vars"])
    pmed = np.nanmedian(primary["J"][:, pidx], axis=0)
    amed = np.nanmedian(alt["J"][:, aidx], axis=0)
    rhos = []
    for i, j, _ in edges:
        rho = spearmanr(pmed[:, i, j], amed[:, i, j]).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))
    return np.quantile(rhos, [0.25, 0.50, 0.75]), np.asarray(rhos)


specs = [
    ("Cov removed", core_dir, "trajectory_only", core_summary.loc["trajectory_only", "combined_response_rank_correlation"]),
    ("Cov weight = 0.10", cov_dir, "covariance_moderate", cov_summary.loc["covariance_moderate", "combined_response_rank_correlation"]),
    ("Cov weight = 0.40", cov_dir, "covariance_balanced", cov_summary.loc["covariance_balanced", "combined_response_rank_correlation"]),
    ("No geographic pooling", core_dir, "no_hierarchy", core_summary.loc["no_hierarchy", "combined_response_rank_correlation"]),
    ("Sign constraints only", prior_dir, "sign_only", prior_summary.loc["sign_only", "combined_response_rank_correlation"]),
    ("Common weak zero-centered", prior_dir, "common_weak", prior_summary.loc["common_weak", "combined_response_rank_correlation"]),
    ("Half PrEP drift", core_dir, "half_prep_drift", core_summary.loc["half_prep_drift", "combined_response_rank_correlation"]),
    ("Zero PrEP drift", core_dir, "zero_prep_drift", core_summary.loc["zero_prep_drift", "combined_response_rank_correlation"]),
]

rank_rows = []
for label, folder, variant, response_rho in specs:
    qs, rhos = within_pathway_rank_summary(folder, variant)
    rank_rows.append({
        "label": label, "variant": variant,
        "pathway_rho_q25": qs[0], "pathway_rho_median": qs[1], "pathway_rho_q75": qs[2],
        "response_rho": float(response_rho), "n_pathways": len(rhos),
    })
rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUT / "supplementary_sensitivity_within_pathway_rank_summary.csv", index=False)


def prior_pair(variant):
    primary = load_fit(prior_dir / "fits/primary.npz")
    alt = load_fit(prior_dir / f"fits/{variant}.npz")
    pidx = [primary["units"].index(s) for s in ROBUST]
    aidx = [alt["units"].index(s) for s in ROBUST]
    edges = edge_indices(primary["vars"])
    pmed = np.nanmedian(primary["J"][:, pidx], axis=0)
    amed = np.nanmedian(alt["J"][:, aidx], axis=0)
    x, y = [], []
    for i, j, _ in edges:
        x.extend(pmed[:, i, j])
        y.extend(amed[:, i, j])
    x, y = np.asarray(x), np.asarray(y)
    median_abs = float(np.median(np.abs(y - x)))
    alt_draws = alt["J"][:, aidx]
    vals = np.concatenate([alt_draws[:, :, i, j].ravel() for i, j, _ in edges])
    zero_share = float(np.mean(np.isclose(vals, 0.0, atol=1e-6)))
    return x, y, median_abs, zero_share


blue = "#3E6FA8"
orange = "#D98235"
purple = "#7B5AA6"
dark = "#262626"
grid = "#D9D9D9"

plt.rcParams.update({"font.family": "Arial", "font.size": 9.2, "axes.titlesize": 10.5, "axes.labelsize": 9.5})

# ---------------------------------------------------------------------
# TWO-PANEL FIGURE LAYOUT
# ---------------------------------------------------------------------
# Panel A occupies the wider left column. Panel B occupies the right
# column and contains two vertically stacked coefficient comparisons.
# Dedicated title rows prevent panel headings from colliding with the
# overall figure title or the axes subtitles.
# Keep the figure 8.5 inches wide. The lower grid row is reserved for
# Panel A's legend, allowing the Panel A plotting frame itself to align
# with the bottom of B2.
fig = plt.figure(figsize=(8.5, 5.90), facecolor="white")

outer = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[1.25, 1.0],
    height_ratios=[0.84, 0.16],
    left=0.22,
    right=0.98,
    bottom=0.10,
    top=0.98,
    wspace=0.28,
    hspace=0.18,
)

panel_A_grid = outer[0, 0].subgridspec(
    nrows=2,
    ncols=1,
    height_ratios=[0.11, 0.89],
    hspace=0,
)

# Preserve Panel A's natural plot-and-legend spacing. Give Panel B an
# independent vertical extent so the bottom of B2's x-axis text aligns
# with the bottom of Panel A's legend text.
panel_B_slot = outer[:, 1].get_position(fig)
panel_B_grid = fig.add_gridspec(
    nrows=2,
    ncols=1,
    left=panel_B_slot.x0,
    right=panel_B_slot.x1,
    bottom=0.22,
    top=0.98,
    height_ratios=[0.11, 0.89],
    hspace=0,
)

title_A = fig.add_subplot(panel_A_grid[0])
title_B = fig.add_subplot(panel_B_grid[0])

for title_ax in (title_A, title_B):
    title_ax.set_axis_off()

title_A.text(
    0, 0.55, "A. Agreement in state rankings",
    ha="left", va="center", fontsize=11, fontweight="bold",
)
title_B.text(
    0, 0.55, "B. Prior sensitivity of coefficients",
    ha="left", va="center", fontsize=9.6, fontweight="bold",
)

axA = fig.add_subplot(panel_A_grid[1])
legend_A = fig.add_subplot(outer[1, 0])
legend_A.set_axis_off()

panel_B_content = panel_B_grid[1].subgridspec(
    nrows=2,
    ncols=1,
    height_ratios=[1, 1],
    hspace=0.48,
)
axB1 = fig.add_subplot(panel_B_content[0])
axB2 = fig.add_subplot(panel_B_content[1])


# ---------------------------------------------------------------------
# PANEL A
# ---------------------------------------------------------------------
groups = {
    "Covariance influence": [
        "Cov removed",
        "Cov weight = 0.10",
        "Cov weight = 0.40",
    ],
    "Geographic borrowing": [
        "No geographic pooling",
    ],
    "Pathway regularization": [
        "Sign constraints only",
        "Common weak zero-centered",
    ],
    "PrEP secular drift": [
        "Half PrEP drift",
        "Zero PrEP drift",
    ],
}

ypos = []
ytick_labels = []
row_to_y = {}
y = 12.5

for group_name, items in groups.items():
    group_y = y
    y -= 0.7

    for item in items:
        ypos.append(y)
        ytick_labels.append(item)
        row_to_y[item] = y
        y -= 1.0

    y -= 0.5
    row_to_y[f"__group__{group_name}"] = group_y

for row in rank_rows:
    label = row["label"]
    y_val = row_to_y.get(label)

    if y_val is None:
        continue

    axA.errorbar(
        row["pathway_rho_median"],
        y_val,
        xerr=[
            [row["pathway_rho_median"] - row["pathway_rho_q25"]],
            [row["pathway_rho_q75"] - row["pathway_rho_median"]],
        ],
        fmt="o",
        ms=6.5,
        color=blue,
        ecolor=blue,
        elinewidth=1.5,
        capsize=2.5,
        zorder=3,
    )

    axA.scatter(
        row["response_rho"],
        y_val - 0.15,
        marker="s",
        s=42,
        color=orange,
        zorder=4,
    )

# Group headings are positioned in axes coordinates so constrained layout
# reserves room for them within the complete Panel A footprint.
for group_name in groups:
    group_y = row_to_y[f"__group__{group_name}"]

    axA.text(
        -0.02,
        group_y,
        group_name,
        transform=axA.get_yaxis_transform(),
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=9,
        color=dark,
        clip_on=False,
    )

# Separators between groups.
group_boundaries = []
y_temp = 12.5

for _, items in groups.items():
    y_temp -= 0.7

    for _ in items:
        y_temp -= 1.0

    group_boundaries.append(y_temp + 0.25)
    y_temp -= 0.5

for y_boundary in group_boundaries[:-1]:
    axA.axhline(
        y_boundary,
        color=grid,
        linewidth=0.8,
        linestyle="-",
        xmin=0,
        xmax=1,
    )

axA.set_yticks(ypos)
axA.set_yticklabels(ytick_labels)
axA.set_xlim(0, 1.02)
axA.set_ylim(min(ypos) - 0.8, 13.0)
axA.set_xticks(np.arange(0, 1.01, 0.2))

axA.axvline(
    1,
    color="#999999",
    linestyle="--",
    linewidth=0.8,
    zorder=1,
)

axA.grid(
    axis="x",
    color=grid,
    linewidth=0.5,
    alpha=0.6,
)

axA.set_xlabel("Spearman correlation with primary model")
axA.tick_params(axis="y", length=0, pad=6)

axA.plot(
    [],
    [],
    "o-",
    color=blue,
    label="State ordering within pathways\n(median and IQR across pathways)",
)

axA.plot(
    [],
    [],
    "s",
    color=orange,
    label="State ordering by combined-scenario response",
)

handles, legend_labels = axA.get_legend_handles_labels()
legend_A.legend(
    handles,
    legend_labels,
    frameon=False,
    loc="center left",
    fontsize=8.0,
    handletextpad=0.5,
    labelspacing=0.35,
    borderaxespad=0,
)



# ---------------------------------------------------------------------
# PANEL B
# ---------------------------------------------------------------------
for ax, variant, subtitle, color in [
    (
        axB1,
        "sign_only",
        "B1. Sign constraints only",
        blue,
    ),
    (
        axB2,
        "common_weak",
        "B2. Common weak zero-centered",
        purple,
    ),
]:
    x, y, median_abs_change, boundary_share = prior_pair(variant)

    limit = max(
        np.max(np.abs(x)),
        np.max(np.abs(y)),
    ) * 1.08

    limit = max(limit, 0.30)

    ax.scatter(
        x,
        y,
        s=13,
        alpha=0.42,
        color=color,
        edgecolor="none",
    )

    ax.plot(
        [-limit, limit],
        [-limit, limit],
        color="#666666",
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0,
        color=grid,
        linewidth=0.7,
    )

    ax.axvline(
        0,
        color=grid,
        linewidth=0.7,
    )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")

    # These are subplot subtitles, not full panel headings.
    ax.set_title(
        subtitle,
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        pad=8,
    )

    ax.text(
        0.03,
        0.96,
        (
            f"Median |Δ| = {median_abs_change:.3f}\n"
            f"Boundary share = {boundary_share:.1%}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=dark,
    )

    ax.set_xlabel("Primary coefficient")
    ax.set_ylabel("Alternative coefficient")


# ---------------------------------------------------------------------
# CONSISTENT AXIS STYLING
# ---------------------------------------------------------------------
for ax in (axA, axB1, axB2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#666666")

axA.spines["left"].set_visible(False)


# ---------------------------------------------------------------------
# PREVIEW OR SAVE
# ---------------------------------------------------------------------
png = (
    OUT
    / "supplementary_figure_sensitivity_summary_v9.png"
)

pdf = (
    OUT
    / "supplementary_figure_sensitivity_summary_v9.pdf"
)

PREVIEW_ONLY = False

if PREVIEW_ONLY:
    print(
        "Preview mode: close the figure window to end the script. "
        "No output files will be overwritten."
    )
    plt.show()
else:
    fig.savefig(
        png,
        dpi=300,
        facecolor="white",
    )

    fig.savefig(
        pdf,
        facecolor="white",
    )

    plt.close(fig)

    print(png)
    print(pdf)
