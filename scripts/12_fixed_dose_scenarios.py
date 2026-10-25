#!/usr/bin/env python3
"""Fixed-dose sensitivity: stigma reduced by fixed percentage points instead of by 50%.

The relative scenarios (scripts/11) halve each draw's baseline stigma, so states
with more stigma receive larger reductions. Here every state and draw receives the
same reduction in percentage points:
  matched   the median reduction under the relative all-stigma scenario (s7) in
            the dose year (2032, when the ramp reaches full effect), across the
            primary states and draws, separately for each stigma measure
  half      half of the matched dose
Each dose is applied to AHS alone and to all stigma levels
(config/interventions.py, register_fixed_dose_scenarios), through the same
projection as the primary scenarios (pipeline/sem_scenarios.py; same draw pairs).
State rankings are then compared with the relative scenarios, AHS alone (s1) and
all stigma levels (s7), for 2036 incidence and cumulative 2025-2036 incidence.

Outputs (results/05_sensitivity/fixed_dose/): the scenario endpoint files, plus
  doses.csv                                   dose by measure (probability scale)
  relative_vs_absolute_state_comparison.csv   median reductions by state
  comparison_metrics.csv                      Spearman rank agreement with the relative scenarios
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.interventions import FIXED_DOSE_STIGMA, register_fixed_dose_scenarios  # noqa: E402
from pipeline.sem_scenarios import SEMScenarioRunConfig, run_sem_scenarios  # noqa: E402

RESULTS = ROOT / "results"
RELATIVE = {"ahs": "s1_reduce_ahs", "combined": "s7_combined_stigma_package"}
OUTCOMES = ("incidence", "incidence_cumulative")


def matched_doses(sem_draws_path: Path, year: int) -> dict[str, dict[str, float]]:
    """Median percentage-point reduction of each stigma measure under s7 in ``year``."""
    sem = pd.read_csv(sem_draws_path)
    sem = sem[sem["scenario"].eq(RELATIVE["combined"]) & sem["year"].eq(year)]
    if sem.empty:
        raise ValueError(f"No {RELATIVE['combined']} results for {year} in {sem_draws_path}")
    matched = {var: float(-sem.loc[sem["variable"].eq(var), "delta"].median()) for var in FIXED_DOSE_STIGMA.values()}
    return {"matched": matched, "half": {var: d / 2 for var, d in matched.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standardized-input", type=Path, default=RESULTS / "03_fits/primary/standardized_research.npz")
    parser.add_argument("--relative-dir", type=Path, default=RESULTS / "04_scenarios/primary")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "05_sensitivity/fixed_dose")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dose-year", type=int, default=2032)
    parser.add_argument("--year", type=int, default=2036)
    args = parser.parse_args()

    doses = matched_doses(args.relative_dir / "scenario_endpoint_sem_draws.csv", args.dose_year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(doses).rename_axis("variable").to_csv(args.output_dir / "doses.csv")
    print(pd.DataFrame(doses).mul(100).round(1).to_string())
    codes = register_fixed_dose_scenarios(doses)

    run_sem_scenarios(SEMScenarioRunConfig(
        standardized_input_path=args.standardized_input, output_dir=args.output_dir,
        scenario_codes=codes, n_samples=args.n_samples, seed=args.seed,
        target_end_year=args.year, intervention_duration_steps=3, report_years=(args.year,),
        save_draw_level_endpoints=True,
    ))

    fixed = pd.read_csv(args.output_dir / "scenario_endpoint_epi_draws.csv")
    relative = pd.read_csv(args.relative_dir / "scenario_endpoint_epi_draws.csv")
    draws = pd.concat([fixed, relative[relative["scenario"].isin(RELATIVE.values())]])
    draws = draws[draws["year"].eq(args.year) & draws["variable"].isin(OUTCOMES)]
    reduction = -draws.groupby(["variable", "unit_id", "scenario"])["pct_change"].median().unstack("scenario")

    comparison, metrics = [], []
    for outcome in OUTCOMES:
        r = reduction.loc[outcome]
        for family, rel_code in RELATIVE.items():
            for dose in doses:
                abs_code = f"abs_{dose}_{family}"
                corr = spearmanr(r[rel_code], r[abs_code])
                metrics.append({
                    "outcome": outcome, "family": family, "absolute_dose": dose,
                    "spearman_rank_correlation": corr.statistic, "spearman_p_value": corr.pvalue,
                    "median_relative_reduction_pct": r[rel_code].median(),
                    "median_absolute_reduction_pct": r[abs_code].median(),
                })
        comparison.append(r.assign(outcome=outcome).reset_index())
    pd.concat(comparison).to_csv(args.output_dir / "relative_vs_absolute_state_comparison.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.output_dir / "comparison_metrics.csv", index=False)
    print(metrics.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
