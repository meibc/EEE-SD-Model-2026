#!/usr/bin/env python3
from __future__ import annotations

"""Compare relative and fixed percentage-point stigma perturbations.

This is a diagnostic around the final-v9 standardized artifact.  It does not
refit the SEM.  It uses paired SEM/CDC draws and changes only the probability-
scale state intervention:

* 10 percentage-point reduction in AHS alone;
* 10 percentage-point reduction in AHS, GSS, and family stigma plus the final
  50% attenuation of AHS -> PrEP;
* analogous 5 percentage-point scenarios.

The absolute target at each time is max(baseline probability - dose, eps), and
is linearly phased in using the same three-step ramp as the production run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = MODEL_ROOT / "output/standardized_input_v9.npz"
DEFAULT_RELATIVE = (
    MODEL_ROOT
    / "output/sem_scenarios_seehcpdiag_ref0p5_riskweakdrift_500/scenario_endpoint_epi_draws.csv"
)
DEFAULT_OUTPUT = Path("output/absolute_stigma_reduction_sensitivity_n500")

ROBUST_STATES = [
    "CA", "TX", "FL", "NY", "GA", "PA", "IL", "OH", "NC", "IN",
    "VA", "MD", "MI", "NJ", "MA", "WA", "TN",
]


def install_absolute_interventions() -> None:
    """Extend the runtime codebooks and StateIntervention for this diagnostic."""
    from config.interventions import INTERVENTION_CODEBOOK, SCENARIO_CODEBOOK
    from models.shared.intervention import StateIntervention, _logit

    for dose_label, dose in (("10pp", 0.10), ("5pp", 0.05)):
        for short, var, label in (
            ("ahs", "stigma_ahs", "anticipated healthcare stigma"),
            ("gss", "stigma_gss", "general social stigma"),
            ("family", "stigma_family", "family stigma"),
        ):
            INTERVENTION_CODEBOOK[f"reduce_{short}_abs_{dose_label}"] = {
                "var": var,
                "delta": -dose,
                "effect": "probability_absolute_to_baseline",
                "label": f"Reduce {label} by {dose_label}",
            }

        SCENARIO_CODEBOOK[f"abs_{dose_label}_ahs"] = {
            "state_codes": [f"reduce_ahs_abs_{dose_label}"],
            "relationship_codes": [],
            "label": f"Reduce anticipated healthcare stigma by {dose_label}",
        }
        SCENARIO_CODEBOOK[f"abs_{dose_label}_combined"] = {
            "state_codes": [
                f"reduce_ahs_abs_{dose_label}",
                f"reduce_gss_abs_{dose_label}",
                f"reduce_family_abs_{dose_label}",
            ],
            "relationship_codes": ["weaken_ahs_to_prep"],
            "label": f"Combined fixed-{dose_label} stigma package",
        }

    original_apply = StateIntervention.apply

    def patched_apply(self, X_next, X_prev, t, predictor, baseline_p=None):
        if self.effect != "probability_absolute_to_baseline":
            return original_apply(self, X_next, X_prev, t, predictor, baseline_p)
        if baseline_p is None:
            raise ValueError("probability_absolute_to_baseline requires baseline_p")
        X_new = X_next.copy()
        scale = self._scale(t)
        ref_logit = predictor.reference_logits
        if ref_logit is None:
            ref_logit = np.zeros(X_new.shape[0], dtype=float)
        p_current = float(predictor._inverse_x(X_new)[self.var_idx])
        p_target = np.clip(float(baseline_p[self.var_idx]) + self.delta, 1e-6, 1 - 1e-6)
        p_new = np.clip((1.0 - scale) * p_current + scale * p_target, 1e-6, 1 - 1e-6)
        X_new[self.var_idx] = _logit(p_new) - ref_logit[self.var_idx]
        return X_new

    StateIntervention.apply = patched_apply


def sample_stacks(output, unit_id: str, var: str) -> np.ndarray:
    return np.asarray([getattr(s.cdc_output, var) for s in output.results[unit_id].samples])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--relative-draws", type=Path, default=DEFAULT_RELATIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    sys.path.insert(0, str(MODEL_ROOT))
    install_absolute_interventions()

    from config.interventions import SCENARIO_CODEBOOK
    from input.standardized_runtime import (
        StandardizedBundle,
        StandardizedCDCParamsLoader,
        StandardizedSEMParamsLoader,
    )
    from models.shared.alignment import build_model_years
    from pipeline.joint_simulation import run_uncertainty

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = StandardizedBundle(args.standardized_input)
    sem_loader = StandardizedSEMParamsLoader(bundle)
    cdc_loader = StandardizedCDCParamsLoader(bundle)
    units = bundle.build_units()
    unit_ids = [s for s in ROBUST_STATES if s in units]
    years = build_model_years(cdc_loader.years, target_end_year=2036)
    year_idx = int(np.where(years == 2036)[0][0])

    common = dict(
        sem_loader=sem_loader,
        cdc_params_loader=cdc_loader,
        units=units,
        unit_ids=unit_ids,
        n_samples=args.n_samples,
        seed=args.seed,
        show_progress=False,
        model_years=years,
        intervention_duration_steps=3,
    )
    print("running paired baseline", flush=True)
    baseline = run_uncertainty(
        **common, state_intervention_codes=[], relationship_intervention_codes=[]
    )

    draw_rows: list[dict] = []
    scenario_codes = ["abs_10pp_ahs", "abs_10pp_combined", "abs_5pp_ahs", "abs_5pp_combined"]
    for code in scenario_codes:
        spec = SCENARIO_CODEBOOK[code]
        print(f"running {code}", flush=True)
        scenario = run_uncertainty(
            **common,
            state_intervention_codes=spec["state_codes"],
            relationship_intervention_codes=spec["relationship_codes"],
        )
        for state in unit_ids:
            b = sample_stacks(baseline, state, "incidence")[:, year_idx]
            q = sample_stacks(scenario, state, "incidence")[:, year_idx]
            pct = 100.0 * (q - b) / b
            for draw, (bv, qv, pv) in enumerate(zip(b, q, pct)):
                draw_rows.append({
                    "state": state,
                    "scenario": code,
                    "scenario_label": spec["label"],
                    "draw": draw,
                    "year": 2036,
                    "baseline_incidence": bv,
                    "scenario_incidence": qv,
                    "pct_change": pv,
                })

    draws = pd.DataFrame(draw_rows)
    draws.to_csv(args.output_dir / "absolute_scenario_incidence_draws.csv", index=False)
    summary = (
        draws.groupby(["state", "scenario"])["pct_change"]
        .agg(
            p025=lambda x: np.quantile(x, 0.025),
            p25=lambda x: np.quantile(x, 0.25),
            p50="median",
            p75=lambda x: np.quantile(x, 0.75),
            p975=lambda x: np.quantile(x, 0.975),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "absolute_scenario_incidence_summary.csv", index=False)

    relative = pd.read_csv(args.relative_draws)
    relative = relative[
        relative["unit_id"].isin(unit_ids)
        & relative["variable"].eq("incidence")
        & relative["scenario"].isin(["s1_reduce_ahs", "s7_combined_stigma_package"])
    ].copy()
    relative_med = relative.groupby(["unit_id", "scenario"])["pct_change"].median().unstack()
    absolute_med = summary.pivot(index="state", columns="scenario", values="p50")
    compare = pd.DataFrame(index=unit_ids)
    compare.index.name = "state"
    compare["relative_ahs_reduction_pct"] = -relative_med["s1_reduce_ahs"]
    compare["absolute_10pp_ahs_reduction_pct"] = -absolute_med["abs_10pp_ahs"]
    compare["absolute_5pp_ahs_reduction_pct"] = -absolute_med["abs_5pp_ahs"]
    compare["relative_combined_reduction_pct"] = -relative_med["s7_combined_stigma_package"]
    compare["absolute_10pp_combined_reduction_pct"] = -absolute_med["abs_10pp_combined"]
    compare["absolute_5pp_combined_reduction_pct"] = -absolute_med["abs_5pp_combined"]
    for col in [c for c in compare if c.endswith("reduction_pct")]:
        compare[col.replace("reduction_pct", "rank")] = compare[col].rank(ascending=False, method="min")
    compare.to_csv(args.output_dir / "relative_vs_absolute_state_comparison.csv")

    metric_rows = []
    for family in ("ahs", "combined"):
        rel = compare[f"relative_{family}_reduction_pct"]
        for dose in ("10pp", "5pp"):
            absolute = compare[f"absolute_{dose}_{family}_reduction_pct"]
            corr = spearmanr(rel, absolute)
            metric_rows.append({
                "family": family,
                "absolute_dose": dose,
                "spearman_rank_correlation": corr.statistic,
                "spearman_p_value": corr.pvalue,
                "median_relative_reduction_pct": rel.median(),
                "median_absolute_reduction_pct": absolute.median(),
            })
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "comparison_metrics.csv", index=False)
    print(compare.sort_values("absolute_10pp_combined_reduction_pct", ascending=False).to_string(), flush=True)


if __name__ == "__main__":
    main()
