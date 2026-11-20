#!/usr/bin/env python3
"""Interactive model dashboard (exploratory): the primary model (config/specifications.py, primary_fb_*: logistic PrEP
diffusion toward each state's willingness ceiling, AHS through the mixture) run in the browser for each primary
state and the primary states combined.

Exports, for each primary state, a subset of paired draws (the first draws of the scenario runner's pairing,
pipeline/projection.py, seed 123): from each of the three main fits (feedback -0.075, 0, +0.15) the transition
matrix, drift, anchors and the feedback b;
the draw's observed 2018-2024 inputs and AHS-PrEP prevalence ratio; the state's PrEP ceiling; the paired epidemic-model parameters; the
CDC inputs; and the observed trends (median and 95% interval across all input draws). Aggregates and model
parameters only. The model itself is scripts/6_dashboard/model.js (checked against the Python model by
scripts/6_dashboard/validate.py); the page is scripts/6_dashboard/template.html.

Output: results/dashboard/model_dashboard.html (a single file; results/ is not committed).
Run from the repository root: python scripts/6_dashboard/build_dashboard.py [--draws 50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifacts.standardized_runtime import StandardizedBundle, StandardizedDynamicLoader, StandardizedEpiLoader  # noqa: E402
from config.projection import SEED, TARGET_END_YEAR, N_SAMPLES, CUMULATIVE_START_YEAR  # noqa: E402
from core.years import build_model_years, extend_to_end_year  # noqa: E402
from reporting import sources  # noqa: E402
from reporting.summaries import primary_states  # noqa: E402

HERE = Path(__file__).resolve().parent
SPEC = sources.PRIMARY_FITS / "bundles"
FITS = {"primary_fb_neg075": -0.075, "primary_fb_0": 0.0, "primary_fb_pos15": 0.15}   # the main feedback set
OUTPUT = sources.RESULTS / "dashboard/model_dashboard.html"
LABELS = {"stigma_ahs": "Healthcare stigma (avoided or feared care)", "stigma_gss": "General social stigma",
          "stigma_family": "Family stigma", "out_gid": "Out to a healthcare provider", "seehcp": "Saw a provider (past year)",
          "risk_behavior": "Condomless sex, partner HIV+ or status unknown", "prep_used": "Used PrEP (past year)", "hivtest12": "HIV test (past year)"}


def paired_draws(n_dynamic: int, n_epi: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The first k pairs of the scenario runner's pairing (pipeline/projection.py, ProjectionRunner.run)."""
    rng = np.random.default_rng(SEED)
    d = rng.choice(n_dynamic, size=N_SAMPLES, replace=N_SAMPLES > n_dynamic)
    e = rng.choice(n_epi, size=N_SAMPLES, replace=N_SAMPLES > n_epi)
    return d[:k], e[:k]


def r6(x) -> list:
    return np.round(np.asarray(x, dtype=float), 6).tolist()


def export(k: int) -> dict:
    loaders = {name: StandardizedDynamicLoader(StandardizedBundle(SPEC / f"{name}.npz")) for name in FITS}
    first = StandardizedBundle(SPEC / "primary_fb_0.npz")
    dyn0, epi = loaders["primary_fb_0"], StandardizedEpiLoader(first)
    units = first.build_units()
    v = list(dyn0.v_names)
    dyn_years = extend_to_end_year(dyn0.ts, target_end_year=TARGET_END_YEAR, step=2)
    model_years = build_model_years(epi.years, target_end_year=TARGET_END_YEAR)
    d_idx, e_idx = paired_draws(dyn0.n_samples, epi.n_samples, k)
    msm = pd.read_csv(sources.MSM_POPULATION).set_index("state")["median"]
    raw = np.load(SPEC / "primary_fb_0.npz", allow_pickle=True)
    geo = raw["geo_ids"].astype(str).tolist()
    states = {}
    for s in primary_states():
        unit = units[s]
        n_elig = np.interp(model_years, unit.cdc_years, np.asarray(unit.get_cdc("PrEP Eligible"), float)) \
            if len(unit.get_cdc("PrEP Eligible")) != len(model_years) else np.asarray(unit.get_cdc("PrEP Eligible"), float)
        prev = np.asarray(unit.get_cdc("Estimated HIV prevalence (MSM)"), float)
        vs = np.asarray(unit.get_cdc("HIV viral suppression"), float)
        no_vs = np.maximum(0.0, prev - vs)
        if len(no_vs) != len(model_years):
            no_vs = np.interp(model_years, unit.cdc_years, no_vs)
        draws = []
        for d, e in zip(d_idx, e_idx):
            p0 = dyn0.load_sample(int(d), s)
            ratio = dyn0.ahs_prep_apr(int(d), s)
            fits = {}
            for name, loader in loaders.items():
                p = loader.load_sample(int(d), s)
                if not np.allclose(p.observed, p0.observed) or not np.isclose(loader.ahs_prep_apr(int(d), s), ratio):
                    raise ValueError(f"Fits are not paired on draw {d} ({name}, {s})")
                fits[f"{FITS[name]:g}"] = {"J": r6(np.asarray(p.J).ravel()), "u": r6(p.drift), "ref": r6(p.reference_logits),
                                           "b": float(loader.mixture_prep_to_ahs)}
            ep = epi.load_sample(int(e), s)
            draws.append({"d": int(d), "e": int(e), "obs": r6(p0.observed), "r": round(float(ratio), 6), "fits": fits,
                          "epi": {"beta": ep.beta, "alpha": ep.alpha, "kdx": ep.kdx, "U0": ep.U0, "post": ep.post_multiplier}})
        obs_all = 100 * np.asarray(raw["sem_obs_samples"][:, geo.index(s)], float)        # draw x measure x period
        q = np.quantile(obs_all, [0.025, 0.5, 0.975], axis=0)
        ceilings = {loader.mixture_prep_ceiling(s) for loader in loaders.values()}
        if len(ceilings) != 1:
            raise ValueError(f"Fits disagree on the PrEP ceiling of {s}: {ceilings}")
        states[s] = {"msm": float(msm[s]), "ceiling": ceilings.pop(), "last_amis_year": int(unit.amis_years[-1]), "risk0": float(epi.load_sample(0, s).risk0),
                     "n_elig": r6(n_elig), "no_vs": r6(no_vs), "observed": {"low": r6(q[0]), "median": r6(q[1]), "high": r6(q[2])},
                     "draws": draws}
    # Primary states combined: MSM-weighted within each input draw, then quantiles across draws
    sx = [geo.index(s) for s in primary_states()]
    w = np.array([msm[s] for s in primary_states()], float)
    pooled = 100 * np.einsum("dsmt,s->dmt", np.asarray(raw["sem_obs_samples"][:, sx], float), w / w.sum())
    pq = np.quantile(pooled, [0.025, 0.5, 0.975], axis=0)
    return {"observed_pooled": {"low": r6(pq[0]), "median": r6(pq[1]), "high": r6(pq[2])}, "v_names": v, "labels": [LABELS[x] for x in v], "dyn_years": dyn_years.tolist(), "model_years": model_years.tolist(),
            "obs_years": [int(y) for y in dyn0.ts], "feedback_values": list(FITS.values()),
            "cumulative_start_year": CUMULATIVE_START_YEAR,
            "n_draws": k, "states": states}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--data-only", type=Path, default=None, help="Write only the exported data (JSON) to this path")
    args = parser.parse_args()
    data = export(args.draws)
    if args.data_only:
        args.data_only.write_text(json.dumps(data))
        print(f"data: {args.data_only}")
        return
    page = (HERE / "template.html").read_text()
    page = page.replace("/*MODEL_JS*/", (HERE / "model.js").read_text()).replace("/*DATA_JSON*/", json.dumps(data))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page)
    print(f"dashboard: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
