#!/usr/bin/env python3
from __future__ import annotations

"""Summarize the full-draw off-diagonal prior comparison.

Compares the final-v9 magnitude-informed model with:
1. sign_only: hard signs/zeros but no off-diagonal magnitude penalty;
2. common_weak: common zero-centered tau=0.70 off-diagonal penalty.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/final_v9_prior_regularization_comparison_n500"
PRIMARY_RESPONSE = ROOT / "output/final_state_screen_and_response_categories/state_response_categories.csv"
PRIMARY_MODEL = MODEL_ROOT / "output/standardized_input_v9.npz"
PRIOR_CONFIG = MODEL_ROOT / "config/sem_structural_priors_v3.csv"
VARIANTS = ("sign_only", "common_weak")


def markdown_table(df: pd.DataFrame, float_digits: int = 3, include_index: bool = False) -> str:
    shown = df.copy()
    if include_index:
        shown = shown.reset_index()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.{float_digits}f}")
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    divider = "|" + "|".join(["---"] * len(shown.columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def scenario_summary(path: Path, variant: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    med = data.groupby(["state", "scenario"])["pct_change"].median().unstack()
    out = pd.DataFrame(index=med.index)
    out[f"{variant}_combined_reduction"] = -med["s7_combined_stigma_package"]
    out[f"{variant}_ahs_reduction"] = -med["s1_reduce_ahs"]
    out[f"{variant}_ahs_share"] = (
        out[f"{variant}_ahs_reduction"] / out[f"{variant}_combined_reduction"]
    )
    return out


def classify(combined: pd.Series, share: pd.Series) -> pd.Series:
    result = pd.Series("Broader stigma responsiveness", index=combined.index, dtype=object)
    result.loc[combined < 10.0] = "Limited aggregate responsiveness"
    result.loc[(combined >= 10.0) & (share >= 0.80)] = "AHS-centered responsiveness"
    return result


def main() -> None:
    primary = pd.read_csv(PRIMARY_RESPONSE).set_index("state")
    selected_states = primary.index.astype(str).tolist()

    table = pd.DataFrame(index=selected_states)
    table["primary_combined_reduction"] = primary["combined_reduction_magnitude_pct"]
    table["primary_ahs_reduction"] = primary["ahs_reduction_magnitude_pct"]
    table["primary_ahs_share"] = primary["ahs_share_of_combined"]
    table["primary_category"] = primary["response_category"]
    for variant in VARIANTS:
        summary = scenario_summary(OUT / f"scenarios/{variant}.csv", variant)
        table = table.join(summary)
        table[f"{variant}_category"] = classify(
            table[f"{variant}_combined_reduction"], table[f"{variant}_ahs_share"]
        )

    for variant in ("primary",) + VARIANTS:
        table[f"{variant}_rank"] = table[f"{variant}_combined_reduction"].rank(
            ascending=False, method="min"
        ).astype(int)
    table["sign_only_rank_shift"] = table["sign_only_rank"] - table["primary_rank"]
    table["common_weak_rank_shift"] = table["common_weak_rank"] - table["primary_rank"]
    table = table.sort_values("primary_rank")
    table.to_csv(OUT / "full500_state_response_and_rank_comparison.csv", index_label="state")

    base = np.load(PRIMARY_MODEL, allow_pickle=True)
    geo_ids = np.asarray(base["geo_ids"]).astype(str).tolist()
    v_names = np.asarray(base["sem_v_names"]).astype(str).tolist()
    state_idx = [geo_ids.index(state) for state in selected_states]
    primary_J = np.asarray(base["sem_J_samples"], dtype=float)[:, state_idx]

    priors = pd.read_csv(PRIOR_CONFIG)
    v_idx = {name: i for i, name in enumerate(v_names)}
    edge_mask = np.zeros((len(v_names), len(v_names)), dtype=bool)
    for row in priors.itertuples(index=False):
        edge_mask[v_idx[str(row.target)], v_idx[str(row.source)]] = True

    rows = []
    primary_response = table["primary_combined_reduction"]
    primary_median_J = np.nanmedian(primary_J, axis=0)
    rows.append(
        {
            "variant": "primary",
            "pathway_rank_correlation": 1.0,
            "median_absolute_pathway_change": 0.0,
            "response_rank_correlation": 1.0,
            "category_agreement": 1.0,
            "fit_success_rate": 1.0,
            "permitted_coefficient_zero_share": float(np.mean(np.isclose(primary_J[:, :, edge_mask], 0.0, atol=1e-6))),
            "median_state_response_iqr_width": float(
                np.median(
                    np.abs(
                        primary["combined_p75_pct_change"]
                        - primary["combined_p25_pct_change"]
                    )
                )
            ),
        }
    )
    for variant in VARIANTS:
        fit = np.load(OUT / f"fits/{variant}.npz", allow_pickle=True)
        units = np.asarray(fit["unit_order"]).astype(str).tolist()
        idx = [units.index(state) for state in selected_states]
        J = np.asarray(fit["Jmeans_stack"], dtype=float)[:, idx]
        med_J = np.nanmedian(J, axis=0)
        a = primary_median_J[:, edge_mask].ravel()
        b = med_J[:, edge_mask].ravel()
        losses = pd.read_csv(OUT / f"fits/{variant}.loss_components.csv")
        state_losses = losses[losses["unit_id"].astype(str).isin(selected_states)]
        scenario_draws = pd.read_csv(OUT / f"scenarios/{variant}.csv")
        combined_draws = scenario_draws[
            scenario_draws["scenario"].eq("s7_combined_stigma_package")
        ]
        response_iqrs = combined_draws.groupby("state")["pct_change"].agg(
            lambda values: values.quantile(0.75) - values.quantile(0.25)
        )
        rows.append(
            {
                "variant": variant,
                "pathway_rank_correlation": float(spearmanr(a, b).statistic),
                "median_absolute_pathway_change": float(
                    np.nanmedian(np.abs(J[:, :, edge_mask] - primary_J[:, :, edge_mask]))
                ),
                "response_rank_correlation": float(
                    spearmanr(primary_response, table[f"{variant}_combined_reduction"]).statistic
                ),
                "category_agreement": float(
                    (table["primary_category"] == table[f"{variant}_category"]).mean()
                ),
                "fit_success_rate": float(state_losses["success"].mean()),
                "permitted_coefficient_zero_share": float(
                    np.mean(np.isclose(J[:, :, edge_mask], 0.0, atol=1e-6))
                ),
                "median_state_response_iqr_width": float(response_iqrs.median()),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "full500_comparison_summary.csv", index=False)

    stable_top = table.index[
        (table["primary_rank"] <= 5)
        & (table["sign_only_rank"] <= 5)
        & (table["common_weak_rank"] <= 5)
    ].tolist()
    stable_bottom = table.index[
        (table["primary_rank"] >= 13)
        & (table["sign_only_rank"] >= 13)
        & (table["common_weak_rank"] >= 13)
    ].tolist()

    report = [
        "# Full 500-draw prior-regularization comparison",
        "",
        "All specifications retain the final-v9 structural zeros, hard signs, diagonal persistence priors, drift, hierarchy, and stability penalty.",
        "",
        "- `primary`: evidence-tier-specific nonzero off-diagonal means and taus.",
        "- `sign_only`: no off-diagonal magnitude loss.",
        "- `common_weak`: zero-centered tau=0.70 loss for all permitted off-diagonal pathways.",
        "",
        "## Global comparison",
        "",
        markdown_table(comparison, float_digits=3),
        "",
        f"States remaining in the top five under all specifications: {', '.join(stable_top) or 'none'}.",
        f"States remaining in the bottom five under all specifications: {', '.join(stable_bottom) or 'none'}.",
        "",
        "## State responses and ranks",
        "",
        markdown_table(
            table[[
                "primary_combined_reduction", "sign_only_combined_reduction", "common_weak_combined_reduction",
                "primary_rank", "sign_only_rank", "common_weak_rank",
            ]],
            float_digits=2,
            include_index=True,
        ),
        "",
        "Absolute responses are conditional on regularization. Comparative ranking is assessed separately and should not be inferred from category agreement, which is threshold-dependent.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
