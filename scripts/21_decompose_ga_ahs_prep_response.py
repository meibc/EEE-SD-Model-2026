"""Decompose Georgia's 2036 PrEP response to the 50% AHS scenario.

The three paired simulations are:
1. final-step only: impose the AHS reduction at 2034 only, so it enters the
   2034 -> 2036 transition once;
2. sustained, no PrEP carryover: use the production sustained/ramped AHS
   scenario, but remove J[PrEP, PrEP] times the prior scenario-generated PrEP
   difference at each transition;
3. full sustained: the production scenario, including fitted PrEP persistence.

All simulations use the production bootstrap fits, drift, centered-logit
references, and draw-specific AMIS starting values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIT_PATH = ROOT / "output/amis_joint_bootstrap_direct_v2_n500/sem_fits_direct_bootstrap_n500.npz"
INPUT_PATH = ROOT / "output/amis_joint_bootstrap_direct_v2_n500/sem_input_draws_direct_bootstrap_n500.npz"
OUT_DIR = ROOT / "output/amis_joint_bootstrap_direct_v2_n500/diagnostics"
YEARS = np.arange(2018, 2038, 2)
RAMP_START_INDEX = 4  # 2026; scale is zero at the first ramp step
RAMP_END_INDEX = 7    # full effect at 2032 and thereafter


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


def ramp_scale(t):
    if t < RAMP_START_INDEX:
        return 0.0
    if t >= RAMP_END_INDEX:
        return 1.0
    return (t - RAMP_START_INDEX) / (RAMP_END_INDEX - RAMP_START_INDEX)


def impose_half_baseline_ahs(x_next, baseline_p, ahs_idx, ref_logits, scale):
    if scale <= 0:
        return x_next
    out = x_next.copy()
    p_current = expit(out[ahs_idx] + ref_logits[ahs_idx])
    p_target = np.clip(0.5 * baseline_p[ahs_idx], 1e-6, 1 - 1e-6)
    p_new = (1 - scale) * p_current + scale * p_target
    out[ahs_idx] = logit(p_new) - ref_logits[ahs_idx]
    return out


def simulate_draw(J, drift, x0, refs, ahs_idx, prep_idx, mode):
    ref_logits = logit(refs)
    n_var, n_year = x0.size, YEARS.size
    xb = np.zeros((n_var, n_year))
    xs = np.zeros((n_var, n_year))
    xb[:, 0] = x0
    xs[:, 0] = x0

    for t in range(1, n_year):
        xb[:, t] = J @ xb[:, t - 1] + drift
        candidate = J @ xs[:, t - 1] + drift

        if mode == "final_step_only":
            # Alter AHS at 2034 only; this affects PrEP once in 2036.
            scale = 1.0 if YEARS[t] == 2034 else 0.0
        else:
            scale = ramp_scale(t)
        candidate = impose_half_baseline_ahs(
            candidate, expit(xb[:, t] + ref_logits), ahs_idx, ref_logits, scale
        )

        if mode == "sustained_no_prep_carryover":
            # Preserve the production baseline, but prevent the preceding
            # scenario-generated PrEP difference from entering PrEP through
            # its diagonal persistence coefficient.
            prior_prep_difference = xs[prep_idx, t - 1] - xb[prep_idx, t - 1]
            candidate[prep_idx] -= J[prep_idx, prep_idx] * prior_prep_difference

        xs[:, t] = candidate

    return expit(xb + ref_logits[:, None]), expit(xs + ref_logits[:, None])


def main():
    fit = np.load(FIT_PATH, allow_pickle=True)
    inp = np.load(INPUT_PATH, allow_pickle=True)
    units = [str(x) for x in fit["unit_order"]]
    variables = [str(x) for x in fit["v_names"]]
    ga = units.index("GA")
    ahs = variables.index("stigma_ahs")
    prep = variables.index("prep_used")
    refs = np.asarray(fit["reference_probs"], dtype=float)

    J = np.asarray(fit["Jmeans_stack"][:, ga], dtype=float)
    drift = np.asarray(fit["drift_stack"][:, ga], dtype=float)
    y0 = np.asarray(inp["Xbars_samples"][:, :, 0, ga], dtype=float)
    x0 = logit(y0) - logit(refs)[None, :]

    modes = [
        ("final_step_only", "One final-period application"),
        ("sustained_no_prep_carryover", "Sustained reduction, no PrEP carryover"),
        ("full_sustained", "Sustained reduction with fitted persistence"),
    ]
    rows = []
    trajectory_rows = []
    for d in range(J.shape[0]):
        for mode, label in modes:
            base, scenario = simulate_draw(J[d], drift[d], x0[d], refs, ahs, prep, mode)
            for yi, year in enumerate(YEARS):
                trajectory_rows.append(
                    {
                        "draw": d,
                        "mode": mode,
                        "label": label,
                        "year": year,
                        "baseline_prep": base[prep, yi],
                        "scenario_prep": scenario[prep, yi],
                        "prep_difference_pp": 100 * (scenario[prep, yi] - base[prep, yi]),
                        "baseline_ahs": base[ahs, yi],
                        "scenario_ahs": scenario[ahs, yi],
                        "ahs_difference_pp": 100 * (scenario[ahs, yi] - base[ahs, yi]),
                    }
                )
            delta = 100 * (scenario[prep, -1] - base[prep, -1])
            rows.append({"draw": d, "mode": mode, "label": label, "prep_difference_pp_2036": delta})

    draws = pd.DataFrame(rows)
    trajectories = pd.DataFrame(trajectory_rows)
    summary = (
        draws.groupby(["mode", "label"], sort=False)["prep_difference_pp_2036"]
        .quantile([0.025, 0.25, 0.5, 0.75, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "q025", 0.25: "q25", 0.5: "median", 0.75: "q75", 0.975: "q975"})
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUT_DIR / "ga_ahs_prep_decomposition_draws.csv", index=False)
    trajectories.to_csv(OUT_DIR / "ga_ahs_prep_decomposition_trajectories.csv", index=False)
    summary.to_csv(OUT_DIR / "ga_ahs_prep_decomposition_summary.csv", index=False)

    order = [label for _, label in modes]
    plot = summary.set_index("label").loc[order].reset_index()
    y = np.arange(len(plot))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.errorbar(
        plot["median"], y,
        xerr=np.vstack([plot["median"] - plot["q25"], plot["q75"] - plot["median"]]),
        fmt="o", color="#3B73AF", ecolor="#9BB9D8", capsize=3,
    )
    ax.axvline(0, color="0.55", lw=1, ls="--")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Difference in projected PrEP use in 2036 (percentage points)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ga_ahs_prep_decomposition.png", dpi=300)
    fig.savefig(OUT_DIR / "ga_ahs_prep_decomposition.pdf")
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nMedian fitted GA AHS->PrEP:", np.median(J[:, prep, ahs]))
    print("Median fitted GA PrEP persistence:", np.median(J[:, prep, prep]))


if __name__ == "__main__":
    main()
