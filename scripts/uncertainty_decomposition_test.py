from __future__ import annotations

"""
Uncertainty decomposition test for combined SEM+CDC sampling.

Compares incidence trajectories under:
1) SEM-only uncertainty (CDC fixed at point estimates)
2) CDC-only uncertainty (SEM fixed at one trajectory)
3) Combined SEM+CDC uncertainty

Usage:
  source .venv313/bin/activate
  python scripts/uncertainty_decomposition_test.py --unit NY --n-samples 1000
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.run import RunConfig
from data.params_cdc import CDCParamsLoader
from data.params_sem import SEMParamsLoader
from data.unit import Unit
from models.epi.prediction.predictor import CDCPredictor
from models.shared.alignment import build_cdc_inputs_from_sem, build_model_years, extend_years, extend_to_end_year
from models.sbm.runner import SBRunner
from config.sem import SEMConfig
from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig
from pipeline.loaders import load_fit_results
from pipeline.results import CDCInputs
from pipeline.joint_simulation import UncertaintyRunner


def _summarize(arr: np.ndarray) -> dict[str, np.ndarray]:
    # arr: (S, T)
    return {
        "q025": np.nanquantile(arr, 0.025, axis=0),
        "q500": np.nanquantile(arr, 0.5, axis=0),
        "q975": np.nanquantile(arr, 0.975, axis=0),
        "mean": np.nanmean(arr, axis=0),
    }


def _print_peak(label: str, years: np.ndarray, stat: dict[str, np.ndarray]) -> None:
    i_peak = int(np.nanargmax(stat["mean"]))
    print(
        f"{label:10s} peak_mean={stat['mean'][i_peak]:.1f} "
        f"year={int(years[i_peak])} "
        f"start_mean={stat['mean'][0]:.1f} "
        f"end_mean={stat['mean'][-1]:.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="NY", help="State/unit id (default NY)")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--fixed-sem-source",
        choices=("deterministic", "sample"),
        default="deterministic",
        help="Source for fixed SEM trajectory in CDC-only branch.",
    )
    ap.add_argument("--save-fig", default=None, help="Optional path to save plot. If omitted, plot is only shown.")
    args = ap.parse_args()

    options = RunConfig()

    # Build/load SEM output in load mode behavior
    base = SEMConfig()
    opt = OptimConfig()
    shrink = ShrinkageConfig()
    runner = SBRunner(base, opt, shrink)
    sem_pickle_path = options.output_dir / options.sem_pickle_name
    fit_results = load_fit_results(sem_pickle_path)
    sem_output = runner.run(fit=False, predict=options.run_predict, fit_results=fit_results)

    units = Unit.to_dict(sem_output.inputs.units)
    if args.unit not in units:
        raise ValueError(f"Unit {args.unit!r} not found in SEM units.")

    cdc_loader = CDCParamsLoader(options.cdc_posterior_path, options.cdc_trans_path)
    sem_loader = SEMParamsLoader(options.sem_params_path)

    target_end_year = (
        int(options.target_end_year)
        if options.target_end_year is not None
        else int(cdc_loader.years[-1]) + int(options.forecast_years_ahead)
    )
    model_years = build_model_years(cdc_loader.years, target_end_year)
    sem_years_base = extend_to_end_year(
        sem_loader.ts,
        target_end_year=int(model_years[-1]) if model_years.size > 0 else None,
    )

    # Reuse runner for SEM trajectory generation
    ur = UncertaintyRunner(
        sem_loader=sem_loader,
        cdc_params_loader=cdc_loader,
        units=units,
        model_years=model_years,
        hivtest_var=options.joint.hivtest_var,
        prep_var=options.joint.prep_var,
        risk_var=options.joint.risk_var,
        n_elig_var=options.joint.n_elig_var,
    )

    rng = np.random.default_rng(args.seed)
    idx_sem = rng.choice(ur.S_sem, size=args.n_samples, replace=True)
    idx_cdc = rng.choice(ur.S_cdc, size=args.n_samples, replace=True)

    # Fixed references
    fixed_sem_idx = int(idx_sem[0])
    fixed_cdc_params = cdc_loader.load_point_estimates(args.unit)

    inc_sem_only = np.zeros((args.n_samples, len(model_years)), dtype=float)
    inc_cdc_only = np.zeros((args.n_samples, len(model_years)), dtype=float)
    inc_combined = np.zeros((args.n_samples, len(model_years)), dtype=float)

    # Build one fixed SEM trajectory for CDC-only decomposition
    # Default: use deterministic SEM trajectory to align with baseline model runs.
    if args.fixed_sem_source == "deterministic":
        if sem_output.predictions is None or args.unit not in sem_output.predictions.results:
            raise ValueError(
                f"Deterministic SEM trajectory unavailable for unit={args.unit}. "
                "Use --fixed-sem-source sample or ensure run_predict=True with saved predictions."
            )
        sem_traj_fixed = np.asarray(
            sem_output.predictions.results[args.unit].Ypred_trajectory,
            dtype=float,
        )
        sem_years_fixed = extend_years(np.asarray(sem_output.predictions.ts, dtype=int), sem_traj_fixed.shape[1])
    else:
        sem_traj_fixed = ur._build_sem_trajectory(args.unit, fixed_sem_idx)
        sem_years_fixed = extend_years(sem_years_base, sem_traj_fixed.shape[1])
    tau_f, prep_f, nelig_f, risk_f = build_cdc_inputs_from_sem(
        sem_traj=sem_traj_fixed,
        unit=units[args.unit],
        hivtest_idx=ur._hivtest_idx,
        prep_idx=ur._prep_idx,
        risk_idx=ur._risk_idx,
        sem_years=sem_years_fixed,
        model_years=model_years,
        n_elig_var=options.joint.n_elig_var,
    )
    cdc_inputs_fixed_sem = CDCInputs(
        years=model_years,
        tau=tau_f,
        prep_on=prep_f,
        N_elig=nelig_f,
        risk_behavior=risk_f,
    )

    for i in range(args.n_samples):
        # SEM-only (vary SEM idx, fixed CDC point estimate)
        sem_traj_i = ur._build_sem_trajectory(args.unit, int(idx_sem[i]))
        sem_years_i = extend_years(sem_years_base, sem_traj_i.shape[1])
        tau_i, prep_i, nelig_i, risk_i = build_cdc_inputs_from_sem(
            sem_traj=sem_traj_i,
            unit=units[args.unit],
            hivtest_idx=ur._hivtest_idx,
            prep_idx=ur._prep_idx,
            risk_idx=ur._risk_idx,
            sem_years=sem_years_i,
            model_years=model_years,
            n_elig_var=options.joint.n_elig_var,
        )
        cdc_inputs_i = CDCInputs(
            years=model_years,
            tau=tau_i,
            prep_on=prep_i,
            N_elig=nelig_i,
            risk_behavior=risk_i,
        )
        out_sem_only = CDCPredictor(fixed_cdc_params).predict(cdc_inputs_i, args.unit)
        inc_sem_only[i, :] = out_sem_only.incidence

        # CDC-only (vary CDC idx, fixed SEM trajectory)
        cdc_params_i = cdc_loader.load_sample(int(idx_cdc[i]), args.unit)
        out_cdc_only = CDCPredictor(cdc_params_i).predict(cdc_inputs_fixed_sem, args.unit)
        inc_cdc_only[i, :] = out_cdc_only.incidence

        # Combined (vary both)
        out_combined = CDCPredictor(cdc_params_i).predict(cdc_inputs_i, args.unit)
        inc_combined[i, :] = out_combined.incidence

    s_sem = _summarize(inc_sem_only)
    s_cdc = _summarize(inc_cdc_only)
    s_comb = _summarize(inc_combined)

    # ------------------------------------------------------------------
    # Sanity dump: confirm this test uses the expected files/inputs.
    # ------------------------------------------------------------------
    print("\n=== SANITY CHECK ===")
    print(f"unit: {args.unit}")
    print(f"fixed_sem_source: {args.fixed_sem_source}")
    print(f"sem_pickle: {options.output_dir / options.sem_pickle_name}")
    print(f"sem_params_path: {options.sem_params_path}")
    print(f"cdc_posterior_path: {options.cdc_posterior_path}")
    print(f"cdc_trans_path: {options.cdc_trans_path}")
    print(f"years(model): {model_years.tolist()}")
    print("fixed-SEM CDC input (first 6 years):")
    print(f"  N_elig:        {np.asarray(nelig_f)[:6]}")
    print(f"  prep_on:       {np.asarray(prep_f)[:6]}")
    print(f"  risk_behavior: {np.asarray(risk_f)[:6]}")
    print(f"  tau:           {np.asarray(tau_f)[:6]}")

    # ------------------------------------------------------------------
    # SEM consistency check:
    # deterministic trajectory from output.pkl vs SEM posterior median
    # from sem_mc_samples_v2.npz for key connector variables.
    # ------------------------------------------------------------------
    if sem_output.predictions is not None and args.unit in sem_output.predictions.results:
        det_sem = np.asarray(sem_output.predictions.results[args.unit].Ypred_trajectory, dtype=float)  # (m, T_det)
        det_years = extend_years(np.asarray(sem_output.predictions.ts, dtype=int), det_sem.shape[1])
        # SEMParamsLoader stores posterior J draws, not trajectories.
        # Build trajectories from each posterior J draw and take median over draws.
        sem_traj_stack = []
        for s in range(sem_loader.n_samples):
            sem_traj_stack.append(ur._build_sem_trajectory(args.unit, s))  # (m, T)
        sem_stack = np.asarray(sem_traj_stack, dtype=float)  # (S, m, T)
        mc_med = np.median(sem_stack, axis=0)  # (m, T_mc)
        mc_years = extend_years(np.asarray(sem_loader.ts, dtype=int), mc_med.shape[1])

        # common years only
        common_years = sorted(set(det_years.astype(int)).intersection(set(mc_years.astype(int))))
        if common_years:
            print("\nSEM consistency (deterministic vs SEM posterior median):")
            key_vars = [options.joint.hivtest_var, options.joint.prep_var, options.joint.risk_var]
            for var in key_vars:
                if var not in sem_output.inputs.v_names or var not in sem_loader.v_names:
                    continue
                i_det = sem_output.inputs.v_names.index(var)
                i_mc = sem_loader.v_names.index(var)
                det_vals = []
                mc_vals = []
                for y in common_years:
                    d_idx = int(np.where(det_years.astype(int) == y)[0][0])
                    m_idx = int(np.where(mc_years.astype(int) == y)[0][0])
                    det_vals.append(float(det_sem[i_det, d_idx]))
                    mc_vals.append(float(mc_med[i_mc, m_idx]))
                det_vals = np.asarray(det_vals, dtype=float)
                mc_vals = np.asarray(mc_vals, dtype=float)
                diff = det_vals - mc_vals
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff**2)))
                print(f"  var={var} | years={common_years[0]}-{common_years[-1]} | MAE={mae:.6f} RMSE={rmse:.6f}")
                print(f"    det first6: {det_vals[:6]}")
                print(f"    mcM first6: {mc_vals[:6]}")
                print(f"    dif first6: {diff[:6]}")
        else:
            print("\nSEM consistency: no overlapping years between deterministic and SEM MC trajectories.")
    else:
        print("\nSEM consistency: deterministic SEM predictions unavailable in sem_output.")

    # Reference trajectories under point estimates for the fixed SEM inputs
    point_params = cdc_loader.load_point_estimates(args.unit)
    point_out = CDCPredictor(point_params).predict(cdc_inputs_fixed_sem, args.unit)
    print("point-estimate trajectories (fixed SEM inputs, first 6):")
    print(f"  Inc: {np.asarray(point_out.incidence)[:6]}")
    print(f"  Dx:  {np.asarray(point_out.diagnosed)[:6]}")
    print(f"  U:   {np.asarray(point_out.undiagnosed)[:6]}")

    print(f"\nUnit={args.unit}, n_samples={args.n_samples}")
    _print_peak("SEM-only", model_years, s_sem)
    _print_peak("CDC-only", model_years, s_cdc)
    _print_peak("Combined", model_years, s_comb)
    print("means first 6:")
    print(f"  SEM-only: {s_sem['mean'][:6]}")
    print(f"  CDC-only: {s_cdc['mean'][:6]}")
    print(f"  Combined: {s_comb['mean'][:6]}")

    years = np.asarray(model_years, dtype=int)
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.fill_between(years, s_sem["q025"], s_sem["q975"], alpha=0.20, color="#1f77b4", label="SEM-only 95% CI")
    ax.plot(years, s_sem["q500"], color="#1f77b4", lw=2.2, label="SEM-only median")

    ax.fill_between(years, s_cdc["q025"], s_cdc["q975"], alpha=0.20, color="#2ca02c", label="CDC-only 95% CI")
    ax.plot(years, s_cdc["q500"], color="#2ca02c", lw=2.2, label="CDC-only median")

    ax.fill_between(years, s_comb["q025"], s_comb["q975"], alpha=0.16, color="#d62728", label="Combined 95% CI")
    ax.plot(years, s_comb["q500"], color="#d62728", lw=2.8, label="Combined median")

    ax.set_title(f"{args.unit}: Incidence uncertainty decomposition")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    plt.show()
    if args.save_fig:
        out_path = Path(args.save_fig)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"Saved decomposition plot: {out_path}")


if __name__ == "__main__":
    main()
