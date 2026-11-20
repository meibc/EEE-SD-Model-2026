#!/usr/bin/env python3
"""Website input file (StigmaScope runtime, EEE-Model-Dashboard-Integration) from the primary model without the
PrEP -> AHS feedback (config/specifications.py, primary_fb_0), fitted to every state with CDC data.

Same keys, shapes and types as the website's current file (``--template``, standardized_input_v9.npz), so its
loader reads it unchanged, plus two per-state keys for the population-PrEP step the runtime adds:

    sem_prep_ceiling   the state's PrEP ceiling C (scripts/2_evidence/04_prep_diffusion.R)
    sem_prep_ratio     the state's AHS-PrEP prevalence ratio r (median over draws)

In the dynamic arrays the PrEP row is P0 / C (P0: PrEP among men without AHS), as in the fit. The runtime turns
it into population PrEP with  P = C x row x (1 - AHS x (1 - r)),  which also returns the observed years to the
observed PrEP. The website centres every state at one reference level (sem_reference_probs, kept from the
template); the fit centres each state at its own anchors a, so the drift is restated for the template's
reference: x' = x + (a - ref) gives x'(t+1) = J x'(t) + u + (I - J)(a - ref), the same trajectories.

Run from the repository root (after scripts/3_model/03 for primary_fb_0 with --selected-states
results/website/website_states.csv --output-dir results/website):
    python scripts/6_dashboard/build_website_npz.py --template <website repo>/standardized_input_v9.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifacts.research_bundle import build_research_bundle  # noqa: E402
from models.dynamic.mixture import prep_without_ahs  # noqa: E402

WEBSITE = ROOT / "results/website"


def website_logit(p: np.ndarray) -> np.ndarray:
    """The website runtime's logit (prediction/transforms.py)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--fit", type=Path, default=WEBSITE / "fits/primary_fb_0.npz")
    parser.add_argument("--input-draws", type=Path, default=ROOT / "results/02_inputs/state_inputs.npz")
    parser.add_argument("--base", type=Path, default=ROOT / "results/02_inputs/epi_fit/epi_bundle.npz")
    parser.add_argument("--output", type=Path, default=WEBSITE / "standardized_input_v10.npz")
    args = parser.parse_args()

    research = WEBSITE / "bundles" / args.fit.name
    build_research_bundle(args.base, args.fit, research, args.input_draws)
    b = dict(np.load(research, allow_pickle=True))
    t = dict(np.load(args.template, allow_pickle=True))
    v = b["sem_v_names"].astype(str).tolist()
    if v != t["sem_v_names"].astype(str).tolist():
        raise ValueError("Measures differ from the template")
    A, P = v.index("stigma_ahs"), v.index("prep_used")

    # Template's state order, for the states in the fit
    fitted = b["geo_ids"].astype(str).tolist()
    geos = [g for g in t["geo_ids"].astype(str) if g in fitted]
    dropped = sorted(set(t["geo_ids"].astype(str)) - set(fitted))
    ix = [fitted.index(g) for g in geos]
    print(f"{len(geos)} states; not in the fit (no CDC data for the epidemic model): {dropped}")

    ref = website_logit(np.asarray(t["sem_reference_probs"], dtype=float))
    J = np.asarray(b["sem_J_samples"], dtype=float)[:, ix]                     # draw x state x m x m
    u = np.asarray(b["sem_drift_samples"], dtype=float)[:, ix]
    a = np.asarray(b["sem_anchor_logits"], dtype=float)[:, ix]
    I = np.eye(len(v))
    drift = u + np.einsum("dsij,dsj->dsi", I - J, a - ref)
    J_pt = np.asarray(b["sem_fit_J_last"], dtype=float)[ix]
    a_pt = np.nanmedian(a, axis=0)
    drift_pt = np.asarray(b["sem_fit_drift"], dtype=float)[ix] + np.einsum("sij,sj->si", I - J_pt, a_pt - ref)

    ceiling = np.asarray(b["sem_mixture_prep_ceiling"], dtype=float)[ix]
    ratio = np.median(np.asarray(b["sem_ahs_prep_apr"], dtype=float)[:, ix], axis=0)
    obs = np.asarray(b["sem_obs"], dtype=float)[ix].copy()                    # state x measure x period
    if np.isnan(obs[:, :, 0]).any():
        raise ValueError("Missing starting values")
    for g in range(len(geos)):
        obs[g, P] = prep_without_ahs(obs[g, P], obs[g, A], ratio[g]) / ceiling[g]
    outside = (obs[:, :, 0] < 0.01) | (obs[:, :, 0] > 0.99)                    # the fit clips logits at [0.01, 0.99]
    if outside.any():
        raise ValueError(f"Starting values outside [0.01, 0.99]: {np.argwhere(outside)}")

    # Cached trajectories as the runtime computes them: from the first observed period, point estimates
    pred = np.empty_like(obs)
    for g in range(len(geos)):
        x = website_logit(obs[g, :, 0]) - ref
        for k in range(obs.shape[2]):
            pred[g, :, k] = 1 / (1 + np.exp(-(x + ref)))
            x = J_pt[g] @ x + drift_pt[g]

    obs[np.isnan(obs)] = 1e-6       # cells without survey data, as in the template (not used by the projection)
    pred[np.isnan(pred)] = 1e-6
    out = {}
    for key, value in t.items():
        if key.startswith("cdc_") and key not in ("cdc_raw_names", "cdc_native_years"):
            src = np.asarray(b[key])
            out[key] = (src[ix] if src.shape[0] == len(fitted) else src[:, ix]).astype(value.dtype)
        elif key in ("cdc_raw_names", "cdc_native_years", "model_years", "sem_obs_years", "sem_pred_years"):
            out[key] = np.asarray(b[key]).astype(value.dtype)
        elif key in ("schema_version", "sem_reference_probs", "sem_v_names"):
            out[key] = value
    out.update(geo_ids=np.array(geos, dtype=object), sem_J_samples=J, sem_drift_samples=drift,
               sem_fit_J_last=J_pt, sem_fit_drift=drift_pt, sem_obs=obs, sem_pred=pred,
               sem_prep_ceiling=ceiling, sem_prep_ratio=ratio)
    missing = set(t) - set(out)
    if missing:
        raise ValueError(f"Template keys not filled: {sorted(missing)}")
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; versions are not overwritten")
    np.savez_compressed(args.output, **out)
    for key in t:
        if np.asarray(out[key]).shape[1:] != np.asarray(t[key]).shape[1:] and key not in ("cdc_raw_native",):
            print(f"  note: {key} {np.asarray(t[key]).shape} -> {np.asarray(out[key]).shape}")
    print(f"website file: {args.output}")


if __name__ == "__main__":
    main()
