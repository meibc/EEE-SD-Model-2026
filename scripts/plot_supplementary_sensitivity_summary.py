from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/supplementary_figures"
OUT.mkdir(parents=True, exist_ok=True)

core = pd.read_csv(ROOT / "output/final_direct_bootstrap_core_ablation_n100/ablation_summary.csv").set_index("variant")
cov = pd.read_csv(ROOT / "output/final_direct_bootstrap_covariance_gradient_n100/ablation_summary.csv").set_index("variant")
prior = pd.read_csv(ROOT / "output/final_direct_bootstrap_prior_regularization_n100/ablation_summary.csv").set_index("variant")
fixed = pd.read_csv(
    ROOT / "output/amis_joint_bootstrap_direct_v2_n500/absolute_stigma_reduction_direct_bootstrap_n500/comparison_metrics.csv"
)

specs = [
    ("Covariance removed", core.loc["trajectory_only"]),
    ("Covariance weight = 0.10", cov.loc["covariance_moderate"]),
    ("Covariance weight = 0.40", cov.loc["covariance_balanced"]),
    ("No geographic pooling", core.loc["no_hierarchy"]),
    ("Sign constraints only", prior.loc["sign_only"]),
    ("Common weak zero-centered", prior.loc["common_weak"]),
    ("Half PrEP drift", core.loc["half_prep_drift"]),
    ("Zero PrEP drift", core.loc["zero_prep_drift"]),
]

group_labels = {
    0: "Covariance information",
    3: "Geographic borrowing",
    4: "Pathway regularization",
    6: "PrEP secular drift",
}

blue = "#3E6FA8"
orange = "#D98235"
dark = "#252525"
grid = "#D8D8D8"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11.6, 6.6), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.62}
)

# Panel A: allocate extra vertical space for group headings.
y_positions = [10.7, 9.7, 8.7, 6.9, 5.1, 4.1, 2.3, 1.3]
labels = [x[0] for x in specs]
path_rho = [float(x[1]["pathway_rank_correlation"]) for x in specs]
resp_rho = [float(x[1]["combined_response_rank_correlation"]) for x in specs]

for y, p, r in zip(y_positions, path_rho, resp_rho):
    ax1.plot([p, r], [y, y], color="#BDBDBD", lw=1.2, zorder=1)
ax1.scatter(path_rho, y_positions, s=50, color=blue, marker="o", label="Pathway ranking", zorder=3)
ax1.scatter(resp_rho, y_positions, s=50, color=orange, marker="s", label="Combined-response ranking", zorder=3)

for y, p, r in zip(y_positions, path_rho, resp_rho):
    ax1.text(p, y + 0.24, f"{p:.2f}", ha="center", va="bottom", fontsize=7.7, color=blue)
    ax1.text(r, y - 0.25, f"{r:.2f}", ha="center", va="top", fontsize=7.7, color=orange)

for idx, heading in group_labels.items():
    ax1.text(0.50, y_positions[idx] + 0.68, heading, ha="left", va="bottom", fontsize=9, weight="bold", color=dark)

ax1.set_yticks(y_positions, labels)
ax1.set_xlim(0.50, 1.015)
ax1.set_ylim(0.55, 12.0)
ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax1.axvline(1.0, color="#777777", ls="--", lw=1)
ax1.grid(axis="x", color=grid, lw=0.7)
ax1.set_xlabel("Spearman correlation with primary specification")
ax1.set_title("A. Stability across model specifications", loc="left", weight="bold", pad=10)
ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.20), ncol=1, handletextpad=0.6)
ax1.tick_params(axis="y", length=0, pad=6)

# Panel B: fixed-dose rankings versus the corresponding proportional scenario.
family_label = {"ahs": "AHS reduction", "combined": "Combined package"}
dose_label = {"10pp": "Fixed 10 percentage points", "5pp": "Fixed 5 percentage points"}
fixed["label"] = fixed.apply(lambda r: dose_label[r["absolute_dose"]], axis=1)
order = [("ahs", "10pp"), ("ahs", "5pp"), ("combined", "10pp"), ("combined", "5pp")]
y2 = [5.4, 4.4, 2.5, 1.5]
vals = []
labs = []
for fam, dose in order:
    row = fixed[(fixed["family"] == fam) & (fixed["absolute_dose"] == dose)].iloc[0]
    vals.append(float(row["spearman_rank_correlation"]))
    labs.append(dose_label[dose])

ax2.scatter(vals, y2, s=62, color=[blue, blue, orange, orange], marker="o", zorder=3)
for x, y, c in zip(vals, y2, [blue, blue, orange, orange]):
    ax2.text(x, y + 0.27, f"{x:.2f}", ha="center", va="bottom", fontsize=8, color=c)
ax2.text(0.50, 6.05, "AHS reduction", ha="left", va="bottom", fontsize=9, weight="bold", color=dark)
ax2.text(0.50, 3.15, "Combined package", ha="left", va="bottom", fontsize=9, weight="bold", color=dark)
ax2.set_yticks(y2, labs)
ax2.set_xlim(0.50, 1.015)
ax2.set_ylim(0.7, 6.55)
ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.axvline(1.0, color="#777777", ls="--", lw=1)
ax2.grid(axis="x", color=grid, lw=0.7)
ax2.set_xlabel("Spearman correlation with proportional scenario")
ax2.set_title("B. Stability across scenario definitions", loc="left", weight="bold", pad=10)
ax2.tick_params(axis="y", length=0, pad=6)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#666666")

fig.suptitle(
    "Comparative state rankings were generally preserved across alternative specifications",
    x=0.06, y=0.985, ha="left", fontsize=14, weight="bold",
)
fig.text(
    0.06, 0.025,
    "Model-specification analyses used 100 paired production draws; fixed-dose comparisons used all 500 draws. "
    "Response-profile classifications were stable across the prespecified threshold grid for 15 of 17 states; "
    "Massachusetts and New Jersey crossed a category boundary.",
    ha="left", va="bottom", fontsize=8.5, color="#444444",
)
fig.subplots_adjust(left=0.25, right=0.98, top=0.89, bottom=0.23)

png = OUT / "supplementary_figure_sensitivity_summary_draft.png"
pdf = OUT / "supplementary_figure_sensitivity_summary_draft.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(png)
print(pdf)
