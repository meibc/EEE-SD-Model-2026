#!/usr/bin/env python3
"""Refit national demographic and California bridge models to joint bootstrap draws."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = Path(__file__).resolve().parents[1]
NATIONAL_INPUT_ROOT = Path(
    "/Users/meibinchen/Documents/Codex/2026-08-13/d/outputs/"
    "priority_sem_inputs_age_prep_primary_six_marginal_annual_smoothed_logit3yr"
)
CA_INPUT_DIR = ROOT / "outputs/ca_hispanic_latino_sem_inputs_corrected"
DEFAULT_BOOTSTRAP = ROOT / "output/population_joint_bootstrap_n500/population_joint_bootstrap_draws_n500.csv.gz"
DEFAULT_OUTPUT = ROOT / "output/population_joint_bootstrap_n500/fits_state_drift"

V_NAMES = [
    "stigma_ahs", "stigma_gss", "stigma_family", "out_gid",
    "seehcp", "risk_behavior", "prep_used", "hivtest12",
]
POPULATIONS = [
    "black_nh", "hispanic_latino", "white_nh",
    "age_under_24", "age_25_34", "age_35_plus",
    "ca_hispanic_latino",
]

_INPUTS = None
_CONFIG = None
_DRIFT_MODE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-path", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--expected-draws", type=int, default=500)
    parser.add_argument("--drift-mode", choices=["current", "zero"], default="current")
    parser.add_argument("--drift-tau", type=float, default=0.35)
    return parser.parse_args()


def input_dirs() -> dict[str, Path]:
    result = {
        pop: NATIONAL_INPUT_ROOT / f"{pop}_sem_inputs"
        for pop in POPULATIONS if pop != "ca_hispanic_latino"
    }
    result["ca_hispanic_latino"] = CA_INPUT_DIR
    return result


def init_worker(maxiter: int, drift_mode: str, drift_tau: float):
    global _INPUTS, _CONFIG, _DRIFT_MODE
    os.chdir(MODEL_REPO)
    if str(MODEL_REPO) not in sys.path:
        sys.path.insert(0, str(MODEL_REPO))
    from pipeline.priority_population_sem import load_priority_population_inputs
    from pipeline.priority_population_sem_mc import PriorityPopulationMCConfig

    _INPUTS = {pop: load_priority_population_inputs(path) for pop, path in input_dirs().items()}
    _CONFIG = PriorityPopulationMCConfig(
        summary_path=Path("unused.csv"),
        input_base_dir=Path("unused"),
        drift_mode=drift_mode,
        drift_tau=drift_tau,
        maxiter=maxiter,
    )
    _DRIFT_MODE = drift_mode


def fit_task(task):
    pop, label, draw, Y = task
    from pipeline.priority_population_sem_mc import _fit_one_sample

    inputs = _INPUTS[pop]
    _, _, _, J, c, diagnostics = _fit_one_sample(_CONFIG, inputs, Y)
    success = bool(diagnostics.get("success", False)) if isinstance(diagnostics, dict) else False
    final_loss = diagnostics.get("final_loss", np.nan) if isinstance(diagnostics, dict) else np.nan
    return pop, label, draw, J, c, success, final_loss


def make_tasks(data: pd.DataFrame, expected_draws: int):
    draw_ids = sorted(data["draw"].unique())
    if len(draw_ids) != expected_draws:
        raise ValueError(f"Expected {expected_draws} draws; found {len(draw_ids)}")
    labels = data[["population_id", "population_label"]].drop_duplicates().set_index("population_id")["population_label"]
    tasks = []
    for pop in POPULATIONS:
        dpop = data[data["population_id"].eq(pop)]
        years = sorted(dpop["Year"].unique())
        expected_years = [2018, 2020, 2022, 2024] if pop == "ca_hispanic_latino" else [2017, 2018, 2019, 2021, 2022, 2023, 2024]
        if years != expected_years:
            raise ValueError(f"{pop}: years {years} do not match {expected_years}")
        for draw in draw_ids:
            cell = dpop[dpop["draw"].eq(draw)].pivot(index="variable", columns="Year", values="value")
            cell = cell.loc[V_NAMES, expected_years]
            if cell.isna().any().any():
                raise ValueError(f"{pop} draw {draw}: incomplete trajectory")
            tasks.append((pop, labels.loc[pop], int(draw), cell.to_numpy(dtype=float)))
    return tasks


def main():
    args = parse_args()
    t0 = time.time()
    data = pd.read_csv(args.bootstrap_path)
    tasks = make_tasks(data, args.expected_draws)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fit_rows, j_rows, drift_rows = [], [], []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.maxiter, args.drift_mode, args.drift_tau),
    ) as pool:
        for idx, result in enumerate(pool.map(fit_task, tasks, chunksize=4), start=1):
            pop, label, draw, J, c, success, final_loss = result
            fit_rows.append({
                "population_id": pop, "population_label": label, "draw": draw,
                "success": success, "final_loss": final_loss,
                "drift_norm": float(np.linalg.norm(c)),
                "J_abs_mean": float(np.mean(np.abs(J))),
            })
            for i, target in enumerate(V_NAMES):
                drift_rows.append({
                    "population_id": pop, "population_label": label, "draw": draw,
                    "variable": target, "drift": float(c[i]),
                })
                for j, source in enumerate(V_NAMES):
                    if i == j or abs(J[i, j]) > 0:
                        j_rows.append({
                            "population_id": pop, "population_label": label, "draw": draw,
                            "target": target, "source": source, "J": float(J[i, j]),
                        })
            if idx % 100 == 0 or idx == len(tasks):
                print(f"completed {idx}/{len(tasks)} fits ({time.time() - t0:.1f}s)", flush=True)

    fits = pd.DataFrame(fit_rows)
    jdraws = pd.DataFrame(j_rows)
    drifts = pd.DataFrame(drift_rows)
    suffix = "state_drift" if args.drift_mode == "current" else f"zero_drift_tau{args.drift_tau:.2f}".replace(".", "p")
    fits.to_csv(args.output_dir / f"population_fit_audit_n500_{suffix}.csv", index=False)
    jdraws.to_csv(args.output_dir / f"population_J_draws_n500_{suffix}.csv", index=False)
    drifts.to_csv(args.output_dir / f"population_drift_draws_n500_{suffix}.csv", index=False)

    summary = pd.DataFrame({
        "metric": ["n_draws", "n_populations", "n_fits", "success_rate", "elapsed_seconds", "drift_mode"],
        "value": [args.expected_draws, len(POPULATIONS), len(fits), fits["success"].mean(), time.time() - t0, args.drift_mode],
    })
    summary.to_csv(args.output_dir / f"population_run_summary_n500_{suffix}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
