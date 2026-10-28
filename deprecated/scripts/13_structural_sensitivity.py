#!/usr/bin/env python3
from __future__ import annotations

"""Coupling strength between anticipated healthcare stigma (AHS) and PrEP use, by state.

For each state and refit, the AHS -> PrEP coupling strength over H = 6 two-year
steps (2024-2036) and its decomposition by first channel out of AHS (directly to
PrEP, or via disclosure, healthcare contact or HIV testing) are computed by
models/sbm/structural.py. "Engagement" is the sum of the three indirect channels.

Outputs: state summaries with uncertainty, draw-level values, the same summaries
under each sensitivity specification, the link to the AHS-scenario PrEP gain,
and a figure.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
from models.sbm.structural import coupling_strength  # noqa: E402
from pipeline.state_fit import SPECIFICATIONS as ALL_SPECIFICATIONS  # noqa: E402

CHANNELS = {
    "prep_used": "Direct (AHS -> PrEP)",
    "out_gid": "Via disclosure",
    "seehcp": "Via healthcare contact",
    "hivtest12": "Via HIV testing",
}
SPECIFICATIONS = tuple(s for s in ALL_SPECIFICATIONS if s != "primary")
MEANINGFUL = -0.1  # centered-logit units; threshold for a non-negligible contribution
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def state_draws(fit_path: Path, H: int) -> pd.DataFrame:
    fit = np.load(fit_path, allow_pickle=True)
    v_names = fit["v_names"].astype(str).tolist()
    units = fit["unit_order"].astype(str).tolist()
    frames = []
    for state in fit["selected_states"].astype(str):
        J = np.asarray(fit["Jmeans_stack"][:, units.index(state)], dtype=float)
        frames.append(pd.DataFrame({"state": state, "draw": np.arange(J.shape[0]), **coupling_strength(J, v_names, H)}))
    return pd.concat(frames, ignore_index=True)


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, g in draws.groupby("state"):
        row = {"state": state}
        for key in ["total", "prep_used", "engagement", *[c for c in CHANNELS if c != "prep_used"]]:
            q = np.quantile(g[key], [0.025, 0.25, 0.5, 0.75, 0.975])
            row.update({f"{key}_p025": q[0], f"{key}_p25": q[1], f"{key}_p50": q[2],
                        f"{key}_p75": q[3], f"{key}_p975": q[4]})
        row["p_direct_at_zero"] = float((g["prep_used"] == 0).mean())
        row["p_engagement_meaningful"] = float((g["engagement"] < MEANINGFUL).mean())
        row["engagement_share_of_median"] = row["engagement_p50"] / row["total_p50"] if row["total_p50"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("state").sort_values("total_p50")


def specification_comparison(spec_dir: Path, H: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = summarize(state_draws(spec_dir / "primary.npz", H))
    by_state, overall = [base.assign(specification="primary")], []
    for spec in SPECIFICATIONS:
        path = spec_dir / f"{spec}.npz"
        if not path.exists():
            continue
        alt = summarize(state_draws(path, H))
        by_state.append(alt.assign(specification=spec))
        idx = base.index.intersection(alt.index)
        rho = lambda col: spearmanr(alt.loc[idx, col], base.loc[idx, col]).correlation
        overall.append({
            "specification": spec,
            "median_total": alt["total_p50"].median(),
            "median_direct": alt["prep_used_p50"].median(),
            "median_engagement": alt["engagement_p50"].median(),
            "rank_rho_total": rho("total_p50"),
            "rank_rho_direct": rho("prep_used_p50"),
            "rank_rho_engagement": rho("engagement_p50"),
        })
    cols = ["specification", "total_p50", "prep_used_p50", "engagement_p50"]
    return pd.concat(by_state).reset_index()[["state", *cols]], pd.DataFrame(overall)


def scenario_link(summary: pd.DataFrame, sem_draws_path: Path) -> pd.DataFrame:
    """Median PrEP gain (percentage points) under the AHS-reduction scenario."""
    sem = pd.read_csv(sem_draws_path)
    sem = sem[sem["scenario"].eq("s1_reduce_ahs") & sem["variable"].eq("prep_used")]
    gain = (100 * sem.groupby(["unit_id", "year"])["delta"].median()).unstack("year")
    return summary.join(gain.add_prefix("ahs_scenario_prep_gain_pp_"), how="left")


def plot(summary: pd.DataFrame, path: Path, gain_year: int = 2036) -> None:
    s = summary.sort_values("total_p50", ascending=False)  # strongest coupling at the bottom
    y = np.arange(len(s))
    gain_col = f"ahs_scenario_prep_gain_pp_{gain_year}"
    fig = plt.figure(figsize=(18, 6.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    ax_tot = fig.add_subplot(grid[0])
    ax_rt = fig.add_subplot(grid[1], sharey=ax_tot)
    ax_sc = fig.add_subplot(grid[2])

    def interval(ax, key, offset, color, label):
        lo95, lo, med, hi, hi95 = (-s[f"{key}_{q}"] for q in ("p975", "p75", "p50", "p25", "p025"))
        ax.hlines(y + offset, lo95, hi95, color=color, alpha=0.35, linewidth=1.5)
        ax.hlines(y + offset, lo, hi, color=color, linewidth=4)
        ax.scatter(med, y + offset, s=50, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)

    interval(ax_tot, "total", 0, BLUE, "Total coupling")
    ax_tot.set_title("A. Total AHS-PrEP coupling strength", loc="left", color=INK, fontsize=11)
    ax_tot.set_xlabel("Cumulative PrEP response to AHS (centered logit, sign reversed)", color=INK)

    interval(ax_rt, "prep_used", 0.17, BLUE, "Direct (AHS -> PrEP)")
    interval(ax_rt, "engagement", -0.17, ORANGE, "Via engagement (disclosure, healthcare contact, testing)")
    ax_rt.set_title("B. Contribution by route", loc="left", color=INK, fontsize=11)
    ax_rt.set_xlabel("Contribution (centered logit, sign reversed)", color=INK)
    ax_rt.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax_tot.set_yticks(y, s.index)
    plt.setp(ax_rt.get_yticklabels(), visible=False)

    if gain_col in s:
        x, g = -s["total_p50"], s[gain_col]
        ax_sc.scatter(x, g, s=50, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        for state, xi, gi in zip(s.index, x, g):
            ax_sc.annotate(state, (xi, gi), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK)
        rho = spearmanr(x, g).correlation
        ax_sc.set_title(f"C. Coupling strength and PrEP gain (Spearman ρ = {rho:.2f})", loc="left", color=INK, fontsize=11)
        ax_sc.set_xlabel("Total coupling strength (sign reversed)", color=INK)
        ax_sc.set_ylabel(f"PrEP gain under AHS reduction, {gain_year} (percentage points)", color=INK)
    for ax in (ax_tot, ax_rt, ax_sc):
        if ax is not ax_sc:
            ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.grid(axis="x" if ax is not ax_sc else "both", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.text(0.01, 0.01, "Points: medians; thick bars: IQR; thin bars: 95% interval across 500 bootstrap refits.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", type=Path, default=RESULTS / "03_fits/primary/fits/primary.npz")
    parser.add_argument("--specification-dir", type=Path, default=RESULTS / "05_sensitivity/model_specification/fits")
    parser.add_argument("--scenario-sem-draws", type=Path,
                        default=RESULTS / "04_scenarios/primary/scenario_endpoint_sem_draws.csv")
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "07_structural")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    draws = state_draws(args.fit, args.horizon_steps)
    summary = summarize(draws)
    if args.scenario_sem_draws.exists():
        summary = scenario_link(summary, args.scenario_sem_draws)
        for year in (2032, 2036):
            col = f"ahs_scenario_prep_gain_pp_{year}"
            if col in summary:
                print(f"Rank rho, coupling strength vs AHS-scenario PrEP gain in {year}: "
                      f"{spearmanr(-summary['total_p50'], summary[col]).correlation:.2f}")
    draws.to_csv(args.output_dir / "state_coupling_draws.csv.gz", index=False)
    summary.to_csv(args.output_dir / "state_coupling_summary.csv")
    plot(summary, args.output_dir / "state_coupling.png")

    if args.specification_dir.exists():
        by_state, overall = specification_comparison(args.specification_dir, args.horizon_steps)
        by_state.to_csv(args.output_dir / "state_coupling_by_specification.csv", index=False)
        overall.to_csv(args.output_dir / "state_coupling_specification_summary.csv", index=False)
        print(overall.round(2).to_string(index=False))

    cols = ["total_p50", "total_p25", "total_p75", "prep_used_p50", "engagement_p50",
            "p_direct_at_zero", "p_engagement_meaningful", "engagement_share_of_median"]
    cols += [c for c in summary.columns if c.startswith("ahs_scenario_prep_gain_pp_")]
    print(summary[cols].round(2).to_string())
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
