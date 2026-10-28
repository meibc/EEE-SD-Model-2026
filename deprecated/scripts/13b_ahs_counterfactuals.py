#!/usr/bin/env python3
from __future__ import annotations

"""How anticipated healthcare stigma (AHS) has shaped PrEP use in each state.

Two counterfactuals on each state's fitted system, for every draw, starting from the draw's
observed 2018 values; every other measure evolves through the fitted pathways and the drift is
the same in both runs of each pair (models/sbm/structural.py):

1. History (source_history_counterfactual): AHS on its observed path vs held at its 2018 level,
   continued two steps (2026, 2028) with AHS at its 2024 vs 2018 level. The difference is the
   part of each trajectory due to the observed change in AHS, including the change already set in
   motion by 2024 (PrEP responds one two-year step after AHS).
2. AHS-free PrEP (source_free_counterfactual): AHS as observed through 2024 and then evolving
   through the fitted system to 2036, vs AHS held from 2018 at a low benchmark (the AHS reference
   probability, the 5th percentile of state-period estimates). The gap is how far PrEP is held
   below what it would be with benchmark AHS, and how that gap develops as AHS and PrEP move.

States are centred at their own anchors (mean logit over the observed periods, per draw).

Outputs (results/07_structural/):
  ahs_history_counterfactual_summary.csv   by state: observed AHS and PrEP change, PrEP effect by year
  ahs_history_counterfactual_draws.csv.gz  by state, draw, measure and year
  ahs_free_prep_summary.csv                by state and year: AHS, PrEP, AHS-free PrEP, gap (median, IQR)
  ahs_free_prep_draws.csv.gz               by state, draw and year (AHS and PrEP)
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.sem import SEMConfig  # noqa: E402
from models.sbm.structural import source_free_counterfactual, source_history_counterfactual  # noqa: E402

RESULTS = ROOT / "results"
EPS = 1e-2  # probability clip used by the fit (FeatureTransformer)


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def load_state(b, g: int) -> dict:
    """One state's draws: J, drift, anchors, observed probabilities and anchored logits."""
    anchors = np.asarray(b["sem_anchor_logits"][:, g], dtype=float)            # draw x measure
    observed = np.asarray(b["sem_obs_samples"][:, g], dtype=float)             # draw x measure x period
    return {"J": np.asarray(b["sem_J_samples"][:, g], dtype=float), "drift": np.asarray(b["sem_drift_samples"][:, g], dtype=float),
            "anchors": anchors, "observed": observed, "X": _logit(observed) - anchors[:, :, None]}


def long_frame(state, years, v_names, columns: dict) -> pd.DataFrame:
    draws, m, T = next(iter(columns.values())).shape
    return pd.DataFrame({"state": state, "draw": np.repeat(np.arange(draws), m * T),
                         "variable": np.tile(np.repeat(v_names, T), draws), "year": np.tile(years, draws * m),
                         **{k: v.ravel() for k, v in columns.items()}})


def history(b, v_names, years_obs, extra_steps: int) -> pd.DataFrame:
    years = np.concatenate([years_obs, years_obs[-1] + 2 * np.arange(1, extra_steps + 1)])
    frames = []
    for g, state in enumerate(b["geo_ids"].astype(str)):
        s = load_state(b, g)
        factual, held = source_history_counterfactual(s["J"], s["drift"], s["X"], v_names, extra_steps=extra_steps)
        pad = np.full(s["observed"].shape[:2] + (extra_steps,), np.nan)
        frames.append(long_frame(state, years, v_names, {
            "observed": np.concatenate([s["observed"], pad], axis=2),
            "factual": _expit(factual + s["anchors"][:, :, None]),
            "ahs_held": _expit(held + s["anchors"][:, :, None]),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["effect_pp"] = 100 * (out["factual"] - out["ahs_held"])
    return out


def summarize_history(draws: pd.DataFrame) -> pd.DataFrame:
    first, last = draws["year"].min(), draws.loc[draws["observed"].notna(), "year"].max()
    end = draws["year"].max()
    obs = draws[draws["variable"].isin(["stigma_ahs", "prep_used"])].pivot_table(
        index=["state", "draw"], columns=["variable", "year"], values="observed")
    rows = {}
    for state, g in draws.groupby("state"):
        o = obs.loc[state]
        row = {"ahs_change_observed_pp": 100 * np.median(o[("stigma_ahs", last)] - o[("stigma_ahs", first)]),
               "prep_change_observed_pp": 100 * np.median(o[("prep_used", last)] - o[("prep_used", first)])}
        prep = g[g["variable"].eq("prep_used")]
        for year, e in prep.groupby("year")["effect_pp"]:
            if year > first:
                row[f"prep_effect_pp_{year}"] = e.median()
        for year in (last, end):
            q = np.quantile(prep.loc[prep["year"].eq(year), "effect_pp"], [0.25, 0.75])
            row.update({f"prep_effect_pp_{year}_p25": q[0], f"prep_effect_pp_{year}_p75": q[1]})
        for var in ("out_gid", "seehcp", "hivtest12"):
            row[f"{var}_effect_pp_{end}"] = g.loc[g["variable"].eq(var) & g["year"].eq(end), "effect_pp"].median()
        rows[state] = row
    return pd.DataFrame(rows).T.sort_values(f"prep_effect_pp_{end}")


def ahs_free(b, v_names, years_obs, end_year: int, benchmark_prob: float) -> pd.DataFrame:
    extra = (end_year - years_obs[-1]) // 2
    years = np.concatenate([years_obs, years_obs[-1] + 2 * np.arange(1, extra + 1)])
    a, p = v_names.index("stigma_ahs"), v_names.index("prep_used")
    frames = []
    for g, state in enumerate(b["geo_ids"].astype(str)):
        s = load_state(b, g)
        benchmark = _logit(benchmark_prob) - s["anchors"][:, a]
        factual, held = source_free_counterfactual(s["J"], s["drift"], s["X"], v_names, benchmark, extra_steps=extra)
        anchors = s["anchors"][:, :, None]
        pf, ph = _expit(factual + anchors), _expit(held + anchors)
        draws = pf.shape[0]
        frames.append(pd.DataFrame({
            "state": state, "draw": np.repeat(np.arange(draws), len(years)), "year": np.tile(years, draws),
            "ahs": pf[:, a].ravel(), "prep": pf[:, p].ravel(), "prep_ahs_free": ph[:, p].ravel(),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["gap_pp"] = 100 * (out["prep_ahs_free"] - out["prep"])
    return out


def summarize_ahs_free(draws: pd.DataFrame) -> pd.DataFrame:
    g = draws.groupby(["state", "year"])
    out = g[["ahs", "prep", "prep_ahs_free", "gap_pp"]].median()
    out[["ahs", "prep", "prep_ahs_free"]] *= 100
    out["gap_pp_p25"] = g["gap_pp"].quantile(0.25)
    out["gap_pp_p75"] = g["gap_pp"].quantile(0.75)
    return out.rename(columns={"ahs": "ahs_pct", "prep": "prep_pct", "prep_ahs_free": "prep_ahs_free_pct"}).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=RESULTS / "03_fits/primary/standardized_research.npz")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "07_structural")
    parser.add_argument("--extra-steps", type=int, default=2, help="History counterfactual steps past the last observed period.")
    parser.add_argument("--end-year", type=int, default=2036, help="Last year of the AHS-free trajectories.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    b = np.load(args.bundle, allow_pickle=True)
    v_names = b["sem_v_names"].astype(str).tolist()
    years_obs = np.asarray(b["sem_obs_years"], dtype=int)

    hist = history(b, v_names, years_obs, args.extra_steps)
    hist.to_csv(args.output_dir / "ahs_history_counterfactual_draws.csv.gz", index=False)
    hist_summary = summarize_history(hist)
    hist_summary.to_csv(args.output_dir / "ahs_history_counterfactual_summary.csv", index_label="state")

    benchmark = float(SEMConfig().reference_probs()[v_names.index("stigma_ahs")])
    free = ahs_free(b, v_names, years_obs, args.end_year, benchmark)
    free.to_csv(args.output_dir / "ahs_free_prep_draws.csv.gz", index=False)
    free_summary = summarize_ahs_free(free)
    free_summary.to_csv(args.output_dir / "ahs_free_prep_summary.csv", index=False)

    print(hist_summary.round(2).to_string())
    print(f"\nAHS-free PrEP gap (pp; benchmark AHS {100 * benchmark:.1f}%), median across draws:")
    print(free_summary.pivot(index="state", columns="year", values="gap_pp").round(2).to_string())


if __name__ == "__main__":
    main()
