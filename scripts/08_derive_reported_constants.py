#!/usr/bin/env python3
from __future__ import annotations

"""Derive data-based model constants reported in the supplement.

All three use the primary precision-screen states and the four model periods.
- Centered-logit references (S2.2): for each measure, the 5th percentile of the
  weighted state-period point estimates.
- PrEP drift regularization target (S2.3): unweighted mean of the three
  two-year changes in logit PrEP use.
- PrEP persistence (S2.4): pooled centered-logit AR(1) slope,
  x[t+1] = a + phi * x[t]. The intercept a is reported as the drift implied by
  a PrEP-only AR(1); the drift target above is the more conservative choice
  because cross-pathways into PrEP also contribute growth.

With --write-config the references, the drift target (3 decimals) and the PrEP
persistence target (phi rounded to the nearest 0.05) are written to config/.
Run after the input stage and the precision screen (scripts 01-07), before fitting.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

YEARS = [2018, 2020, 2022, 2024]


def logit(p: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", type=Path, default=RESULTS / "01_weights/state_model_inputs.xlsx")
    parser.add_argument("--primary-states", type=Path, default=RESULTS / "02_inputs/state_screen/primary_states.csv")
    parser.add_argument("--variable-config", type=Path, default=ROOT / "config/sem_variable_config_v3.csv")
    parser.add_argument("--diagonal-config", type=Path, default=ROOT / "config/sem_diagonal_priors_v3.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "02_inputs/reported_constants.csv")
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    states = pd.read_csv(args.primary_states)["state"].astype(str).tolist()
    trends = pd.read_excel(args.workbook, sheet_name="trends")
    trends = trends[trends["Geography"].isin(states) & trends["Year"].isin(YEARS)]
    if trends.groupby("Indicator")["Geography"].count().ne(len(states) * len(YEARS)).any():
        raise ValueError("Primary states lack a complete set of state-period estimates")

    variable_config = pd.read_csv(args.variable_config)
    references = trends.groupby("Indicator")["Value"].quantile(0.05)

    prep = trends[trends["Indicator"].eq("prep_used")].pivot(index="Geography", columns="Year", values="Value")[YEARS]
    drift_target = float(logit(prep).diff(axis=1).iloc[:, 1:].to_numpy().mean())

    x = logit(prep) - logit(np.array(references["prep_used"]))
    x_t = x[YEARS[:-1]].to_numpy().ravel()
    x_next = x[YEARS[1:]].to_numpy().ravel()
    design = np.column_stack([np.ones_like(x_t), x_t])
    coef = np.linalg.lstsq(design, x_next, rcond=None)[0]
    resid_var = float(((x_next - design @ coef) ** 2).sum() / (len(x_t) - 2))
    se = np.sqrt(np.diag(resid_var * np.linalg.inv(design.T @ design)))

    constants = pd.DataFrame(
        [
            {
                "constant": "prep_drift_target",
                "value": drift_target,
                "se": np.nan,
                "n_states": len(states),
                "n_obs": 3 * len(states),
                "definition": "mean two-year change in logit PrEP use, primary-screen states",
            },
            {
                "constant": "prep_persistence_phi",
                "value": float(coef[1]),
                "se": float(se[1]),
                "n_states": len(states),
                "n_obs": len(x_t),
                "definition": "pooled centered-logit AR(1) slope with intercept, primary-screen states",
            },
            {
                "constant": "prep_ar1_intercept",
                "value": float(coef[0]),
                "se": float(se[0]),
                "n_states": len(states),
                "n_obs": len(x_t),
                "definition": "intercept of the same AR(1) (drift implied by a PrEP-only model); not used",
            },
        ]
        + [
            {
                "constant": f"reference_{v}",
                "value": float(references[v]),
                "se": np.nan,
                "n_states": len(states),
                "n_obs": len(states) * len(YEARS),
                "definition": "5th percentile of weighted state-period estimates, primary-screen states",
            }
            for v in variable_config["variable"]
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    constants.to_csv(args.output, index=False)
    print(constants[["constant", "value", "se", "n_states", "n_obs"]].to_string(index=False))

    if args.write_config:
        variable_config["reference_prob"] = variable_config["variable"].map(references)
        variable_config["reference_note"] = (
            f"Fifth percentile of weighted point estimates across the {len(states)} primary states and four periods."
        )
        variable_config.loc[variable_config["variable"].eq("prep_used"), "drift_prior_mean"] = round(drift_target, 3)
        variable_config.to_csv(args.variable_config, index=False)
        diagonal = pd.read_csv(args.diagonal_config)
        diagonal.loc[diagonal["variable"].eq("prep_used"), "prior_mean"] = round(float(coef[1]) * 20) / 20
        diagonal.to_csv(args.diagonal_config, index=False)
        print(f"Wrote references, PrEP drift target {round(drift_target, 3)} and persistence target "
              f"{round(float(coef[1]) * 20) / 20:.2f} to config/")


if __name__ == "__main__":
    main()
