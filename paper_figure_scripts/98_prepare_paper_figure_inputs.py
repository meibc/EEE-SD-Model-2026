#!/usr/bin/env python3
"""Derive manuscript figure-input CSVs from model outputs.

This script is the auditable data-preparation layer for the paper figures. It
does *not* copy previous figure-level CSVs. Instead, it derives the plotting
inputs from upstream model artifacts:

- `standardized_input_v9.npz` for observed SEM inputs and fitted/J-draw matrices
- draw-level scenario endpoint files for intervention-response summaries
- priority-population J-draw files for Figure 5

The downstream figure script (`99_regenerate_paper_figures.py`) reads only the
CSV files produced here.

Run from EEE-SD-Model-2026:

    python paper_figure_scripts/98_prepare_paper_figure_inputs.py --overwrite
    python paper_figure_scripts/99_regenerate_paper_figures.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROBUST_STATE_REGIONS: dict[str, str] = {
    "MA": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "FL": "South",
    "GA": "South",
    "NC": "South",
    "TN": "South",
    "TX": "South",
    "VA": "South",
    "CA": "West",
    "CO": "West",
    "WA": "West",
}

REGION_ORDER = ["Northeast", "Midwest", "South", "West"]
ROBUST_STATES = [
    state
    for region in REGION_ORDER
    for state, state_region in ROBUST_STATE_REGIONS.items()
    if state_region == region
]

VARIABLE_LABELS = {
    "stigma_ahs": "Anticipated healthcare stigma",
    "stigma_gss": "General social stigma",
    "stigma_family": "Family stigma",
    "out_gid": "Out to healthcare provider",
    "seehcp": "Saw healthcare provider",
    "risk_behavior": "Risk behavior",
    "prep_used": "PrEP use",
    "hivtest12": "HIV testing",
}

CORE_PATHWAYS = [
    ("stigma_ahs_to_prep_used", "stigma_ahs", "prep_used", "AHS → PrEP use", "AHS to prevention"),
    ("stigma_ahs_to_hivtest12", "stigma_ahs", "hivtest12", "AHS → HIV testing", "AHS to prevention"),
    ("stigma_ahs_to_out_gid", "stigma_ahs", "out_gid", "AHS → disclosure", "AHS to disclosure"),
    ("out_gid_to_prep_used", "out_gid", "prep_used", "Disclosure → PrEP use", "Disclosure to prevention"),
    ("out_gid_to_hivtest12", "out_gid", "hivtest12", "Disclosure → HIV testing", "Disclosure to prevention"),
    ("seehcp_to_stigma_ahs", "seehcp", "stigma_ahs", "Saw HCP → AHS", "Care-stigma feedback"),
    ("seehcp_to_hivtest12", "seehcp", "hivtest12", "Saw HCP → HIV testing", "Care to prevention"),
    ("seehcp_to_prep_used", "seehcp", "prep_used", "Saw HCP → PrEP use", "Care to prevention"),
]

FIGURE5_PATHWAYS = [
    ("stigma_ahs", "out_gid", "AHS → disclosure"),
    ("stigma_ahs", "prep_used", "AHS → PrEP use"),
    ("out_gid", "prep_used", "Disclosure → PrEP use"),
    ("out_gid", "hivtest12", "Disclosure → HIV testing"),
    ("seehcp", "stigma_ahs", "Saw HCP → AHS"),
    ("seehcp", "hivtest12", "Saw HCP → HIV testing"),
]

SCENARIO_LABEL_SHORT = {
    "s1_reduce_ahs": "AHS reduction",
    "s4_weaken_ahs_to_prep": "Weaken AHS→PrEP",
    "s7_combined_stigma_package": "Combined package",
}

SCENARIO_LOGIC = {
    "s1_reduce_ahs": "Healthcare-facing stigma reduction",
    "s4_weaken_ahs_to_prep": "Stigma-bypass / access-buffering",
    "s7_combined_stigma_package": "Combined stigma strategy",
}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def quantile_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "p025": float(np.nanquantile(arr, 0.025)),
        "p25": float(np.nanquantile(arr, 0.25)),
        "p50": float(np.nanquantile(arr, 0.50)),
        "p75": float(np.nanquantile(arr, 0.75)),
        "p975": float(np.nanquantile(arr, 0.975)),
    }


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=True) as raw:
        return {k: raw[k] for k in raw.files}


def j_values_for_path(
    j_samples: np.ndarray,
    geo_index: int,
    variable_index: dict[str, int],
    source: str,
    target: str,
) -> np.ndarray:
    """Return J[target, source] across draws for one geography."""
    return j_samples[:, geo_index, variable_index[target], variable_index[source]]


def derive_observed_change_heatmap(data: dict[str, np.ndarray]) -> pd.DataFrame:
    geos = [str(x) for x in data["geo_ids"]]
    variables = [str(x) for x in data["sem_v_names"]]
    years = [int(x) for x in data["sem_obs_years"]]
    obs = np.asarray(data["sem_obs"], dtype=float)

    start_idx = 0
    end_idx = len(years) - 1
    rows = []
    for state in ROBUST_STATES:
        geo_idx = geos.index(state)
        for var_idx, variable in enumerate(variables):
            start_pct = 100 * obs[geo_idx, var_idx, start_idx]
            end_pct = 100 * obs[geo_idx, var_idx, end_idx]
            rows.append(
                {
                    "state": state,
                    "region": ROBUST_STATE_REGIONS[state],
                    "variable": variable,
                    "variable_label": VARIABLE_LABELS.get(variable, variable),
                    "start_year": years[start_idx],
                    "end_year": years[end_idx],
                    "start_pct": start_pct,
                    "end_pct": end_pct,
                    "change_pp": end_pct - start_pct,
                }
            )
    out = pd.DataFrame(rows)
    order = {state: i for i, state in enumerate(ROBUST_STATES)}
    out["state_order_rank"] = out["state"].map(order)
    return out


def derive_pathway_heterogeneity(data: dict[str, np.ndarray]) -> pd.DataFrame:
    geos = [str(x) for x in data["geo_ids"]]
    variables = [str(x) for x in data["sem_v_names"]]
    var_idx = {v: i for i, v in enumerate(variables)}
    j_samples = np.asarray(data["sem_J_samples"], dtype=float)

    rows = []
    for state in ROBUST_STATES:
        geo_idx = geos.index(state)
        for key, source, target, pathway, family in CORE_PATHWAYS:
            vals = j_values_for_path(j_samples, geo_idx, var_idx, source, target)
            qs = quantile_summary(vals)
            rows.append(
                {
                    "state": state,
                    "pathway_key": key,
                    "pathway": pathway,
                    "pathway_family": family,
                    "p025": qs["p025"],
                    "p50": qs["p50"],
                    "p975": qs["p975"],
                    "zero_share": float(np.mean(np.isclose(vals, 0.0))),
                }
            )
    return pd.DataFrame(rows)


def derive_structural_scores(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Derive Figure 2 structural-position scores directly from J draws.

    Score definitions:
    - proximal_score: pathway strength from AHS/care-facing stigma dynamics into
      prevention/disclosure, computed per draw as
        abs(AHS→PrEP) + abs(AHS→testing) + abs(AHS→disclosure)
        + max(0, -SawHCP→AHS)
      The last term treats protective care-stigma feedback as proximal leverage.
    - cross_score: upstream/broader stigma reinforcement into AHS, computed as
        max(0, GSS→AHS) + max(0, family→AHS) + max(0, GSS→family)
        + max(0, family→GSS)
    """
    geos = [str(x) for x in data["geo_ids"]]
    variables = [str(x) for x in data["sem_v_names"]]
    var_idx = {v: i for i, v in enumerate(variables)}
    j = np.asarray(data["sem_J_samples"], dtype=float)

    rows = []
    for state in ROBUST_STATES:
        gi = geos.index(state)
        ahs_prep = j_values_for_path(j, gi, var_idx, "stigma_ahs", "prep_used")
        ahs_test = j_values_for_path(j, gi, var_idx, "stigma_ahs", "hivtest12")
        ahs_out = j_values_for_path(j, gi, var_idx, "stigma_ahs", "out_gid")
        hcp_ahs = j_values_for_path(j, gi, var_idx, "seehcp", "stigma_ahs")
        gss_ahs = j_values_for_path(j, gi, var_idx, "stigma_gss", "stigma_ahs")
        fam_ahs = j_values_for_path(j, gi, var_idx, "stigma_family", "stigma_ahs")
        gss_fam = j_values_for_path(j, gi, var_idx, "stigma_gss", "stigma_family")
        fam_gss = j_values_for_path(j, gi, var_idx, "stigma_family", "stigma_gss")

        proximal = np.abs(ahs_prep) + np.abs(ahs_test) + np.abs(ahs_out) + np.maximum(0, -hcp_ahs)
        cross = np.maximum(0, gss_ahs) + np.maximum(0, fam_ahs) + np.maximum(0, gss_fam) + np.maximum(0, fam_gss)

        rows.append(
            {
                "state": state,
                "structure_only_primary": "",
                "structure_only_primary_prob": np.nan,
                "structure_only_certainty": "",
                "cross_context_reinforced": np.nan,
                "ahs_centered": np.nan,
                "broader_stigma_pressured": np.nan,
                "weak_or_decoupled": np.nan,
                "proximal_score_p50": float(np.nanmedian(proximal)),
                "cross_score_p50": float(np.nanmedian(cross)),
                "prevention_momentum_score": np.nan,
                "trend_overlay": "",
                "region": ROBUST_STATE_REGIONS[state],
                "x": float(np.nanmedian(proximal)),
                "y": float(np.nanmedian(cross)),
                "certainty_size": 120.0,
            }
        )
    return pd.DataFrame(rows)


def derive_scenario_response(scenario_draws_path: Path) -> pd.DataFrame:
    draws = pd.read_csv(scenario_draws_path)
    keep_scenarios = set(SCENARIO_LABEL_SHORT)
    sub = draws[
        draws["unit_id"].isin(ROBUST_STATES)
        & draws["scenario"].isin(keep_scenarios)
        & (draws["variable"] == "incidence")
    ].copy()
    if sub.empty:
        raise ValueError(f"No robust-state incidence scenario draws found in {scenario_draws_path}")

    rows = []
    for (state, scenario), g in sub.groupby(["unit_id", "scenario"], sort=False):
        qs = quantile_summary(g["pct_change"])
        rows.append(
            {
                "state": state,
                "scenario": scenario,
                "scenario_label_short": SCENARIO_LABEL_SHORT[scenario],
                "scenario_logic": SCENARIO_LOGIC[scenario],
                **qs,
            }
        )
    return pd.DataFrame(rows)


def derive_structural_alignment_3d(
    structural_scores: pd.DataFrame,
    scenario_response: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = scenario_response[scenario_response["scenario"] == "s7_combined_stigma_package"][
        ["state", "p50"]
    ].rename(columns={"p50": "combined_incidence_pct_change_p50"})
    ahs = scenario_response[scenario_response["scenario"] == "s1_reduce_ahs"][["state", "p50"]].rename(
        columns={"p50": "ahs_incidence_pct_change_p50"}
    )
    df = structural_scores[["state", "proximal_score_p50", "cross_score_p50", "region"]].merge(
        combined, on="state"
    ).merge(ahs, on="state")
    df["scenario_translation_magnitude"] = df["combined_incidence_pct_change_p50"].abs()
    df["direct_ahs_magnitude"] = df["ahs_incidence_pct_change_p50"].abs()

    prox_thr = df["proximal_score_p50"].median()
    cross_thr = df["cross_score_p50"].median()
    trans_thr = df["scenario_translation_magnitude"].median()

    def classify(row: pd.Series) -> str:
        high_prox = row["proximal_score_p50"] >= prox_thr
        high_cross = row["cross_score_p50"] >= cross_thr
        high_trans = row["scenario_translation_magnitude"] >= trans_thr
        if not high_trans:
            return "low detectable aggregate stigma leverage"
        if high_prox and high_cross:
            return "layered stigma action"
        if high_prox and not high_cross:
            return "proximal healthcare stigma action"
        if high_cross and not high_prox:
            return "pressure / low translation"
        return "low detectable aggregate stigma leverage"

    df["integrated_policy_profile_primary"] = df.apply(classify, axis=1)
    df["translation_primary"] = np.where(
        df["scenario_translation_magnitude"] >= trans_thr,
        "high aggregate response",
        "low aggregate response",
    )
    df["structure_primary"] = np.where(
        df["proximal_score_p50"] >= prox_thr,
        "high proximal AHS leverage",
        "low proximal AHS leverage",
    )
    df["bubble_size"] = 80 + 12 * df["scenario_translation_magnitude"]

    table_rows = []
    descriptions = {
        "layered stigma action": (
            "High proximal AHS leverage.",
            "High broader-stigma reinforcement.",
            "High aggregate scenario response.",
            "Combine healthcare stigma reduction, upstream stigma mitigation, and access supports.",
        ),
        "proximal healthcare stigma action": (
            "High proximal AHS leverage.",
            "Lower broader-stigma reinforcement.",
            "High aggregate scenario response.",
            "Prioritize affirming care, stigma-safe navigation, provider quality, and clinical trust.",
        ),
        "pressure / low translation": (
            "Lower proximal AHS leverage.",
            "High broader-stigma reinforcement.",
            "Stigma pressure is visible but weakly translated into aggregate scenario response.",
            "Use stigma-bypass and privacy-protective access; monitor subgroup inequities.",
        ),
        "low detectable aggregate stigma leverage": (
            "Lower proximal AHS leverage.",
            "Lower or mixed broader-stigma reinforcement.",
            "Lower aggregate scenario response.",
            "Prioritize targeted subgroup analysis and non-stigma access bottlenecks.",
        ),
    }
    for profile, g in df.groupby("integrated_policy_profile_primary"):
        a, b, c, d = descriptions[profile]
        table_rows.append(
            {
                "profile": profile.title(),
                "states": ", ".join(sorted(g["state"])),
                "Proximal AHS leverage": a,
                "Cross-context reinforcement": b,
                "Scenario translation": c,
                "Intervention logic": d,
            }
        )
    return df, pd.DataFrame(table_rows)


def summarize_j_draw_file(draws: pd.DataFrame, pathways: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for pop, pop_df in draws.groupby("population_label", sort=False):
        draw_ids = np.sort(pop_df["draw"].unique())
        for source, target, pathway in pathways:
            vals = (
                pd.DataFrame({"draw": draw_ids})
                .merge(
                    pop_df.loc[
                        (pop_df["source"] == source) & (pop_df["target"] == target),
                        ["draw", "J"],
                    ],
                    on="draw",
                    how="left",
                )["J"]
                .fillna(0.0)
            )
            qs = quantile_summary(vals)
            rows.append(
                {
                    "population_label": pop,
                    "pathway": pathway,
                    **qs,
                    "p_nonzero": float(np.mean(~np.isclose(vals, 0.0))),
                    "p_positive": float(np.mean(vals > 0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_state_j_from_npz(
    data: dict[str, np.ndarray],
    state: str,
    group_label: str,
    pathways: list[tuple[str, str, str]],
) -> pd.DataFrame:
    geos = [str(x) for x in data["geo_ids"]]
    variables = [str(x) for x in data["sem_v_names"]]
    var_idx = {v: i for i, v in enumerate(variables)}
    j = np.asarray(data["sem_J_samples"], dtype=float)
    gi = geos.index(state)

    rows = []
    for source, target, pathway in pathways:
        vals = j_values_for_path(j, gi, var_idx, source, target)
        qs = quantile_summary(vals)
        rows.append(
            {
                "group": group_label,
                "pathway": pathway,
                **qs,
                "p_nonzero": float(np.mean(~np.isclose(vals, 0.0))),
                "p_positive": float(np.mean(vals > 0)),
            }
        )
    return pd.DataFrame(rows)


def derive_priority_population_inputs(
    data: dict[str, np.ndarray],
    national_priority_j_path: Path,
    ca_bridge_j_path: Path,
    state_pathway_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    national_draws = pd.read_csv(national_priority_j_path)
    ca_draws = pd.read_csv(ca_bridge_j_path)

    # The current production artifact contains the six national demographic
    # groups and the California bridge in one jointly bootstrapped file.
    if "population_id" in national_draws.columns:
        national_draws = national_draws[
            ~national_draws["population_id"].astype(str).eq("ca_hispanic_latino")
        ].copy()
    if "population_id" in ca_draws.columns:
        ca_draws = ca_draws[
            ca_draws["population_id"].astype(str).eq("ca_hispanic_latino")
        ].copy()

    panel_a = summarize_j_draw_file(national_draws, FIGURE5_PATHWAYS)

    state_ref_rows = []
    state_lookup = {
        "AHS → disclosure": "stigma_ahs_to_out_gid",
        "AHS → PrEP use": "stigma_ahs_to_prep_used",
        "Disclosure → PrEP use": "out_gid_to_prep_used",
        "Disclosure → HIV testing": "out_gid_to_hivtest12",
        "Saw HCP → AHS": "seehcp_to_stigma_ahs",
        "Saw HCP → HIV testing": "seehcp_to_hivtest12",
    }
    for pathway, key in state_lookup.items():
        vals = state_pathway_summary.loc[state_pathway_summary["pathway_key"] == key, "p50"]
        state_ref_rows.append(
            {
                "pathway": pathway,
                "state_q25": float(vals.quantile(0.25)),
                "state_median": float(vals.median()),
                "state_q75": float(vals.quantile(0.75)),
            }
        )
    state_ref = pd.DataFrame(state_ref_rows)
    panel_a = panel_a.merge(state_ref, on="pathway", how="left")

    ca_hispanic = summarize_j_draw_file(ca_draws, FIGURE5_PATHWAYS).rename(
        columns={"population_label": "group"}
    )
    ca_hispanic["group"] = ca_hispanic["group"].replace(
        {"California Hispanic/Latino": "California Hispanic/Latino", "ca_hispanic_latino": "California Hispanic/Latino"}
    )
    ca_aggregate = summarize_state_j_from_npz(
        data, state="CA", group_label="California aggregate", pathways=FIGURE5_PATHWAYS
    )
    ca_summary = pd.concat([ca_aggregate, ca_hispanic], ignore_index=True)
    return panel_a, ca_summary, state_ref


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def prepare_inputs(
    repo_root: Path,
    legacy_root: Path,
    output_dir: Path,
    standardized_input: Path,
    scenario_draws: Path,
    national_priority_j: Path,
    ca_bridge_j: Path,
    overwrite: bool,
) -> Path:
    data = load_npz(standardized_input)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = derive_observed_change_heatmap(data)
    pathway_summary = derive_pathway_heterogeneity(data)
    structural_scores = derive_structural_scores(data)
    scenario_response = derive_scenario_response(scenario_draws)
    alignment_3d, framework_table = derive_structural_alignment_3d(structural_scores, scenario_response)
    fig5_a, fig5_b, fig5_ref = derive_priority_population_inputs(
        data, national_priority_j, ca_bridge_j, pathway_summary
    )

    outputs: list[tuple[str, pd.DataFrame, str]] = [
        ("figure1_observed_change_heatmap_data.csv", observed, "Derived from standardized_input_v9.npz: sem_obs end minus start among robust states."),
        ("figure2_structural_position_scores.csv", structural_scores, "Derived from standardized_input_v9.npz: sem_J_samples named pathway scores."),
        ("figure2_pathway_heterogeneity_data.csv", pathway_summary, "Derived from standardized_input_v9.npz: sem_J_samples pathway quantiles."),
        ("figure3_scenario_response_long.csv", scenario_response, "Derived from draw-level scenario endpoint pct_change values."),
        ("figure4_structural_alignment_3d_data.csv", alignment_3d, "Derived from J-score summaries plus scenario-response summaries."),
        ("figure4_structural_alignment_framework_table.csv", framework_table, "Derived from threshold-based profile labels in this script."),
        ("figure5_priority_population_national_panel_data.csv", fig5_a, "Derived from national priority-population J draw file."),
        ("figure5_california_bridge_panel_data.csv", fig5_b, "Derived from corrected California Hispanic/Latino bridge J draw file."),
        ("figure5_state_reference_panel_data.csv", fig5_ref, "Derived from robust-state J summaries computed from standardized_input_v9.npz."),
    ]

    manifest_rows = []
    for filename, df, note in outputs:
        out_path = output_dir / filename
        write_csv(df, out_path, overwrite=overwrite)
        manifest_rows.append(
            {
                "output_name": filename,
                "prepared_path": str(out_path),
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "note": note,
            }
        )

    manifest_path = output_dir / "paper_figure_input_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(
        description="Derive canonical manuscript figure input CSVs from model outputs."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--legacy-root", type=Path, default=repo_root.parent / "EEE-SD-Model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output/results_refinement/final_figure_inputs",
    )
    parser.add_argument(
        "--standardized-input",
        type=Path,
        default=None,
        help="Standardized NPZ containing sem_obs, sem_J_samples, and SEM metadata.",
    )
    parser.add_argument(
        "--scenario-draws",
        type=Path,
        default=None,
        help="Draw-level epidemiologic scenario endpoint file.",
    )
    parser.add_argument(
        "--national-priority-j",
        type=Path,
        default=None,
        help="Draw-level J estimates for national priority populations.",
    )
    parser.add_argument(
        "--ca-bridge-j",
        type=Path,
        default=None,
        help="Corrected draw-level J estimates for California Hispanic/Latino bridge case.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    standardized_input = (
        args.standardized_input.resolve()
        if args.standardized_input is not None
        else repo_root
        / "output/amis_joint_bootstrap_direct_v2_n500/standardized_input_v9_direct_bootstrap.npz"
    )
    scenario_draws = (
        args.scenario_draws.resolve()
        if args.scenario_draws is not None
        else repo_root
        / "output/amis_joint_bootstrap_direct_v2_n500/sem_scenarios_direct_bootstrap_n500/scenario_endpoint_epi_draws.csv"
    )
    national_priority_j = (
        args.national_priority_j.resolve()
        if args.national_priority_j is not None
        else repo_root
        / "output/population_joint_bootstrap_n500/fits_state_drift/population_J_draws_n500_state_drift.csv"
    )
    ca_bridge_j = (
        args.ca_bridge_j.resolve()
        if args.ca_bridge_j is not None
        else repo_root
        / "output/population_joint_bootstrap_n500/fits_state_drift/population_J_draws_n500_state_drift.csv"
    )
    manifest = prepare_inputs(
        repo_root=repo_root,
        legacy_root=args.legacy_root.resolve(),
        output_dir=args.output_dir.resolve(),
        standardized_input=standardized_input,
        scenario_draws=scenario_draws,
        national_priority_j=national_priority_j,
        ca_bridge_j=ca_bridge_j,
        overwrite=bool(args.overwrite),
    )
    print(f"Derived figure inputs: {manifest}")


if __name__ == "__main__":
    main()
