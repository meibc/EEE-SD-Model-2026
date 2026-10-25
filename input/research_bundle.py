"""Research bundle: the epidemiological bundle joined to a state-model fit.

The bundle (read by ``input.standardized_runtime``) carries, for the fitted
states only: the epidemiological posterior and CDC inputs (from the base bundle,
pipeline/hierarchical_epi_fit.py), the fitted transition matrices and drifts for
each draw, each unit's state anchors (mean logit per measure) for each draw, each
draw's AHS-PrEP prevalence ratio from the evidence target, the observed point
estimates, and each draw's own observed inputs.
SEM fields in the base bundle, if any, are replaced from the fit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

GEO_AXIS0 = {"geo_ids", "sem_obs", "sem_pred", "sem_fit_J_last", "sem_fit_drift",
             "cdc_raw", "cdc_raw_native", "cdc_kappa_prep", "cdc_risk0"}
GEO_AXIS1 = {"sem_J_samples", "sem_drift_samples", "sem_anchor_logits", "sem_ahs_prep_apr", "cdc_beta", "cdc_alpha",
             "cdc_kdx", "cdc_U0", "cdc_post_multiplier", "sem_obs_samples"}
GLOBAL_KEYS = {"schema_version", "model_years", "cdc_native_years", "sem_obs_years",
               "sem_pred_years", "sem_v_names", "cdc_raw_names"}


def observed_point_estimates(workbook_path: Path, states, v_names, years) -> np.ndarray:
    """State x measure x period point estimates from the workbook trends sheet."""
    trends = pd.read_excel(workbook_path, sheet_name="trends")
    wide = trends.pivot_table(index=["Geography", "Indicator"], columns="Year", values="Value")
    return np.array([[[wide.loc[(s, v), y] for y in years] for v in v_names] for s in states], dtype=float)


def observed_draws(input_draws_path: Path, states, v_names, years, draw_indices) -> np.ndarray:
    """Draw x state x measure x period observed inputs for the fitted draws."""
    with np.load(input_draws_path, allow_pickle=True) as d:
        if d["v_names"].astype(str).tolist() != list(v_names) or not np.array_equal(d["ts"], years):
            raise ValueError("Input-draw variables/years do not match the fit")
        units = d["unit_order"].astype(str).tolist()
        draws = np.asarray(d["Xbars_samples"], dtype=float)[np.asarray(draw_indices, dtype=int)]
    # draw x variable x period x unit -> draw x state x variable x period
    return np.transpose(draws[:, :, :, [units.index(s) for s in states]], (0, 3, 1, 2))


def build_research_bundle(base_path: Path, fit_path: Path, output_path: Path, workbook_path: Path,
                          input_draws_path: Path, require_success: bool = True) -> dict:
    """Subset the base bundle to the fitted states, attach the fit, and write it.

    Returns a manifest (also written next to the bundle as JSON).
    """
    with np.load(base_path, allow_pickle=True) as d:
        arrays = {k: d[k] for k in d.files}
    with np.load(fit_path, allow_pickle=True) as d:
        fit = {k: d[k] for k in d.files}
    states = fit["selected_states"].astype(str).tolist()
    geos = arrays["geo_ids"].astype(str).tolist()
    units = fit["unit_order"].astype(str).tolist()
    v_names = fit["v_names"].astype(str).tolist()
    if len(states) != len(set(states)):
        raise ValueError("Duplicate selected states")
    unknown = set(arrays) - GEO_AXIS0 - GEO_AXIS1 - GLOBAL_KEYS
    if unknown:
        raise ValueError(f"Unrecognized bundle fields: {sorted(unknown)}")
    ix = [geos.index(s) for s in states]
    fx = [units.index(s) for s in states]
    if require_success and not np.all(fit["success_stack"][:, fx]):
        raise ValueError("State fits contain optimizer failures")

    for key in GEO_AXIS0 & arrays.keys():
        arrays[key] = arrays[key][ix]
    for key in GEO_AXIS1 & arrays.keys():
        arrays[key] = arrays[key][:, ix]
    arrays["sem_v_names"] = np.array(v_names)
    arrays["sem_obs_years"] = arrays["sem_pred_years"] = np.asarray(fit["ts"], dtype=int)
    arrays["sem_J_samples"] = fit["Jmeans_stack"][:, fx]
    arrays["sem_drift_samples"] = fit["drift_stack"][:, fx]
    arrays["sem_anchor_logits"] = fit["anchor_logits"][:, fx]                 # draw x state x measure
    if "ahs_prep_apr" in fit:
        arrays["sem_ahs_prep_apr"] = fit["ahs_prep_apr"][:, fx]               # draw x state
    arrays["sem_fit_J_last"] = np.nanmedian(arrays["sem_J_samples"], axis=0)
    arrays["sem_fit_drift"] = np.nanmedian(arrays["sem_drift_samples"], axis=0)
    anchors = np.nanmedian(arrays["sem_anchor_logits"], axis=0)                # state x measure
    arrays["sem_obs"] = observed_point_estimates(workbook_path, states, v_names, fit["ts"])
    arrays["sem_obs_samples"] = observed_draws(input_draws_path, states, v_names, fit["ts"],
                                               fit["source_draw_indices"])

    # Cached trajectories under the median fitted system, from the first observed period.
    arrays["sem_pred"] = np.empty_like(arrays["sem_obs"])
    for g in range(len(states)):
        x = logit(np.clip(arrays["sem_obs"][g, :, 0], 1e-6, 1 - 1e-6)) - anchors[g]
        for t in range(len(arrays["sem_pred_years"])):
            arrays["sem_pred"][g, :, t] = expit(x + anchors[g])
            x = arrays["sem_fit_J_last"][g] @ x + arrays["sem_fit_drift"][g]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    manifest = {
        "states": states,
        "sem_draws": int(arrays["sem_J_samples"].shape[0]),
        "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
