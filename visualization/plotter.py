from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from config.plotting import (
    BASELINE_COLOR,
    CDC_RAW_MAP,
    EPI_PLOT_VARS,
    FORECAST_BG,
    INTERVENTION_COLOR,
    OBSERVED_COLOR,
    SEM_PLOT_VARS,
)
from pipeline.results import JointOutput, RunOutput, UncertaintyOutput


def _extend_years(years: Iterable[float | int], target_len: int) -> np.ndarray:
    ys = np.asarray(list(years), dtype=float).ravel()
    if ys.size == target_len:
        return ys
    if ys.size == 0:
        return np.arange(target_len, dtype=float)
    if ys.size > target_len:
        return ys[:target_len]

    if ys.size >= 2:
        step = float(ys[-1] - ys[-2])
        if step == 0:
            step = 1.0
    else:
        step = 1.0

    extra = ys[-1] + step * np.arange(1, target_len - ys.size + 1, dtype=float)
    return np.concatenate([ys, extra])


def _add_forecast_background(ax, years: np.ndarray, forecast_start: float) -> None:
    years = np.asarray(years, dtype=float)
    if years.size == 0:
        return
    ymax = float(np.max(years))
    if ymax > forecast_start:
        ax.axvspan(forecast_start, ymax, color=FORECAST_BG, alpha=0.35, zorder=0)
    ax.axvline(forecast_start, ls="--", color="#9a9a9a", lw=1.2, zorder=2)


def _style_sem_axis(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def plot_state_outputs(
    sem_output: RunOutput,
    joint_output: JointOutput,
    state_ids: list[str] | None = None,
    max_states: int = 5,
    hivtest_var: str = "hivtest12",
    prep_var: str = "prep_used",
) -> list[str]:
    """
    Show SEM and Epi outputs for selected states using interactive matplotlib windows.

    Returns the list of state IDs that were plotted.
    """
    unit_map = {u.id: u for u in sem_output.inputs.units}
    available_states = [
        uid
        for uid in joint_output.results
        if uid in unit_map and getattr(unit_map[uid], "kind", None) == "state"
    ]
    available_states = sorted(available_states)

    if not available_states:
        raise ValueError("No state-level joint outputs available to visualize.")

    if state_ids is None:
        selected = available_states[:max_states]
    else:
        requested = list(dict.fromkeys(state_ids))
        selected = [uid for uid in requested if uid in available_states][:max_states]
        if not selected:
            raise ValueError("None of the requested states are available in joint outputs.")

    sem_var_names = list(sem_output.predictions.v_names)
    sem_hist_years = np.asarray(sem_output.inputs.ts, dtype=float)
    sem_pred_base_years = np.asarray(joint_output.sem_years, dtype=float)

    sem_plot_vars = [
        v for v in SEM_PLOT_VARS if v in sem_var_names
    ]
    if not sem_plot_vars:
        raise ValueError("No requested SEM plotting variables found in SEM outputs.")

    epi_vars = EPI_PLOT_VARS

    for state_id in selected:
        unit = unit_map[state_id]
        result = joint_output.results[state_id]

        sem_traj = np.asarray(result.sem_trajectory, dtype=float)
        sem_pred_years = _extend_years(sem_pred_base_years, sem_traj.shape[1])
        fig_sem, axes_sem = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
        fig_sem.suptitle(f"{state_id}: SEM outputs")
        axes_sem_flat = axes_sem.ravel()
        for i, var in enumerate(sem_plot_vars):
            idx = sem_var_names.index(var)
            ax = axes_sem_flat[i]
            _add_forecast_background(ax, sem_pred_years, sem_hist_years[-1])
            ax.plot(
                sem_hist_years,
                np.asarray(unit.amis_values[idx], dtype=float),
                marker="o",
                linestyle="None",
                markersize=5,
                color=OBSERVED_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label="Observed",
            )
            ax.plot(
                sem_pred_years,
                sem_traj[idx],
                color=BASELINE_COLOR,
                linewidth=2.6,
                label="SEM trajectory",
            )
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            _style_sem_axis(ax)
            ax.grid(alpha=0.3)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.legend(
            handles=[
                Line2D([0], [0], color=BASELINE_COLOR, lw=2.6, label="SEM trajectory"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=OBSERVED_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=7,
                    label="Observed",
                ),
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=8,
        )
        fig_sem.tight_layout()
        plt.show()

        epi_years = np.asarray(result.cdc_output.years, dtype=float)
        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC Epi outputs")
        for ax, var in zip(axes_epi, epi_vars):
            vals = np.asarray(getattr(result.cdc_output, var), dtype=float)
            _add_forecast_background(ax, epi_years, sem_hist_years[-1])
            ax.plot(epi_years, vals, color=BASELINE_COLOR, linewidth=2.6, label="Epi")

            if unit.cdc_names is not None and unit.cdc_values is not None and unit.cdc_years is not None:
                raw_name = CDC_RAW_MAP[var]
                if raw_name in unit.cdc_names:
                    raw_idx = unit.cdc_names.index(raw_name)
                    raw_values = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                    raw_years = np.asarray(unit.cdc_years, dtype=float)
                    mask = ~np.isnan(raw_values)
                    if np.any(mask):
                        ax.scatter(
                            raw_years[mask],
                            raw_values[mask],
                            s=28,
                            marker="o",
                            color=OBSERVED_COLOR,
                            edgecolors="white",
                            linewidths=0.8,
                            label="CDC raw",
                        )

            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig_epi.tight_layout()
        plt.show()

    return selected


def plot_deterministic_comparison(
    sem_output: RunOutput,
    baseline: JointOutput,
    intervention: JointOutput,
    state_ids: list[str] | None = None,
    max_states: int = 5,
) -> list[str]:
    """
    Plot deterministic baseline vs intervention overlays for SEM and CDC outputs.
    """
    unit_map = {u.id: u for u in sem_output.inputs.units}
    available_states = [
        uid
        for uid in baseline.results
        if uid in intervention.results and uid in unit_map and getattr(unit_map[uid], "kind", None) == "state"
    ]
    available_states = sorted(available_states)
    if not available_states:
        raise ValueError("No shared state-level deterministic results found between baseline and intervention.")

    if state_ids is None:
        selected = available_states[:max_states]
    else:
        requested = list(dict.fromkeys(state_ids))
        selected = [uid for uid in requested if uid in available_states][:max_states]
        if not selected:
            raise ValueError("None of the requested states are available in both deterministic result sets.")

    sem_var_names = list(sem_output.inputs.v_names)
    sem_obs_years = np.asarray(sem_output.inputs.ts, dtype=float)
    sem_plot_vars = [v for v in SEM_PLOT_VARS if v in sem_var_names]
    epi_vars = EPI_PLOT_VARS

    for state_id in selected:
        unit = unit_map[state_id]
        b_res = baseline.results[state_id]
        i_res = intervention.results[state_id]

        b_sem = np.asarray(b_res.sem_trajectory, dtype=float)
        i_sem = np.asarray(i_res.sem_trajectory, dtype=float)
        sem_years = _extend_years(sem_obs_years, b_sem.shape[1])

        fig_sem, axes_sem = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
        fig_sem.suptitle(f"{state_id}: SEM baseline vs intervention")
        axes_sem_flat = axes_sem.ravel()
        for i, var in enumerate(sem_plot_vars):
            idx = sem_var_names.index(var)
            ax = axes_sem_flat[i]
            _add_forecast_background(ax, sem_years, sem_obs_years[-1])
            ax.plot(sem_years, b_sem[idx], color=BASELINE_COLOR, linewidth=2.6, label="Baseline")
            ax.plot(sem_years, i_sem[idx], color=INTERVENTION_COLOR, linewidth=2.6, label="Intervention")
            ax.scatter(
                sem_obs_years,
                np.asarray(unit.amis_values[idx], dtype=float),
                s=24,
                color=OBSERVED_COLOR,
                edgecolors="white",
                linewidths=0.8,
                label="Observed",
            )
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            _style_sem_axis(ax)
            ax.grid(alpha=0.3)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.legend(
            handles=[
                Line2D([0], [0], color=BASELINE_COLOR, lw=2.6, label="Baseline"),
                Line2D([0], [0], color=INTERVENTION_COLOR, lw=2.6, label="Intervention"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=OBSERVED_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=7,
                    label="Observed",
                ),
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=8,
        )
        fig_sem.tight_layout()
        plt.show()

        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC baseline vs intervention")
        years = np.asarray(b_res.cdc_output.years, dtype=float)
        for ax, var in zip(axes_epi, epi_vars):
            yb = np.asarray(getattr(b_res.cdc_output, var), dtype=float)
            yi = np.asarray(getattr(i_res.cdc_output, var), dtype=float)
            _add_forecast_background(ax, years, sem_obs_years[-1])
            ax.plot(years, yb, color=BASELINE_COLOR, linewidth=2.6, label="Baseline")
            ax.plot(years, yi, color=INTERVENTION_COLOR, linewidth=2.6, label="Intervention")

            raw_name = CDC_RAW_MAP[var]
            if unit.cdc_names is not None and unit.cdc_values is not None and unit.cdc_years is not None and raw_name in unit.cdc_names:
                raw_idx = unit.cdc_names.index(raw_name)
                raw_vals = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                raw_years = np.asarray(unit.cdc_years, dtype=float)
                mask = ~np.isnan(raw_vals)
                if np.any(mask):
                    ax.scatter(
                        raw_years[mask],
                        raw_vals[mask],
                        s=28,
                        marker="o",
                        color=OBSERVED_COLOR,
                        edgecolors="white",
                        linewidths=0.8,
                        label="CDC raw",
                    )

            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig_epi.tight_layout()
        plt.show()

    return selected


def plot_state_uncertainty_outputs(
    sem_output: RunOutput,
    uncertainty: UncertaintyOutput,
    state_ids: list[str] | None = None,
    max_states: int = 5,
    hivtest_var: str = "hivtest12",
    prep_var: str = "prep_used",
    q_low: float = 0.025,
    q_med: float = 0.5,
    q_high: float = 0.975,
) -> list[str]:
    """
    Show SEM and Epi uncertainty plots for selected states (interactive only).

    Returns the list of state IDs that were plotted.
    """
    unit_map = {u.id: u for u in sem_output.inputs.units}
    available_states = [
        uid
        for uid in uncertainty.results
        if uid in unit_map and getattr(unit_map[uid], "kind", None) == "state"
    ]
    available_states = sorted(available_states)

    if not available_states:
        raise ValueError("No state-level uncertainty outputs available to visualize.")

    if state_ids is None:
        selected = available_states[:max_states]
    else:
        requested = list(dict.fromkeys(state_ids))
        selected = [uid for uid in requested if uid in available_states][:max_states]
        if not selected:
            raise ValueError("None of the requested states are available in uncertainty outputs.")

    sem_var_names = list(sem_output.inputs.v_names)
    sem_obs_years = np.asarray(sem_output.inputs.ts, dtype=float)

    sem_plot_vars = [v for v in SEM_PLOT_VARS if v in sem_var_names]
    if not sem_plot_vars:
        raise ValueError("No requested SEM plotting variables found in SEM outputs.")

    epi_vars = EPI_PLOT_VARS

    for state_id in selected:
        unit = unit_map[state_id]
        u_res = uncertainty.results[state_id]

        sem_stack = np.asarray([s.sem_trajectory for s in u_res.samples], dtype=float)  # (S, m, T)
        sem_years = _extend_years(sem_obs_years, sem_stack.shape[2])
        sem_q = {
            var: np.quantile(
                sem_stack[:, sem_var_names.index(var), :],
                [q_low, q_med, q_high],
                axis=0,
            )
            for var in sem_plot_vars
        }

        epi_years = np.asarray(u_res.years, dtype=float)
        epi_q = {
            var: u_res.get_quantiles(var, q=[q_low, q_med, q_high]) for var in epi_vars
        }

        fig_sem, axes_sem = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
        fig_sem.suptitle(f"{state_id}: SEM uncertainty")
        axes_sem_flat = axes_sem.ravel()
        sem_colors = ["#1f5d9b", "#c8702f", "#2f7a4f", "#b34045", "#5b4f9b"]
        for i, var in enumerate(sem_plot_vars):
            idx = sem_var_names.index(var)
            color = sem_colors[i % len(sem_colors)]
            ax = axes_sem_flat[i]
            _add_forecast_background(ax, sem_years, sem_obs_years[-1])
            ax.fill_between(
                sem_years,
                sem_q[var][0],
                sem_q[var][2],
                alpha=0.12,
                color=color,
                label="95% CI",
            )
            ax.plot(sem_years, sem_q[var][1], color=color, linewidth=2.6, label="Median")
            ax.scatter(
                sem_obs_years,
                np.asarray(unit.amis_values[idx], dtype=float),
                color=OBSERVED_COLOR,
                edgecolors="white",
                linewidths=0.8,
                s=24,
                alpha=0.95,
                label="Observed",
            )
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            _style_sem_axis(ax)
            ax.grid(alpha=0.3)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.legend(
            handles=[
                Patch(facecolor="#1f5d9b", alpha=0.12, label="95% CI"),
                Line2D([0], [0], color="#1f5d9b", lw=2.6, label="Median"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=OBSERVED_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=7,
                    label="Observed",
                ),
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=8,
        )
        fig_sem.tight_layout()
        plt.show()

        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC Epi uncertainty")
        for ax, (var, color) in zip(
            axes_epi,
            [("diagnosed", "#2f7a4f"), ("incidence", "#b34045"), ("prep_on_count", "#5b4f9b")],
        ):
            qv = epi_q[var]
            _add_forecast_background(ax, epi_years, sem_obs_years[-1])
            ax.fill_between(
                epi_years,
                qv[q_low],
                qv[q_high],
                alpha=0.12,
                color=color,
                label="95% CI",
            )
            ax.plot(
                epi_years,
                qv[q_med],
                color=color,
                linewidth=2.6,
                label="Median",
            )

            raw_name = CDC_RAW_MAP.get(var)
            if (
                raw_name is not None
                and unit.cdc_names is not None
                and raw_name in unit.cdc_names
                and unit.cdc_values is not None
                and unit.cdc_years is not None
            ):
                raw_idx = unit.cdc_names.index(raw_name)
                raw_vals = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                raw_years = np.asarray(unit.cdc_years, dtype=float)
                mask = ~np.isnan(raw_vals)
                if np.any(mask):
                    ax.scatter(
                        raw_years[mask],
                        raw_vals[mask],
                        s=28,
                        color=OBSERVED_COLOR,
                        edgecolors="white",
                        linewidths=0.8,
                        marker="o",
                        alpha=0.95,
                        label="CDC raw",
                    )

            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        fig_epi.tight_layout()
        plt.show()

    return selected


def plot_uncertainty_comparison(
    sem_output: RunOutput,
    baseline: UncertaintyOutput,
    intervention: UncertaintyOutput,
    state_ids: list[str] | None = None,
    max_states: int = 5,
    q_low: float = 0.025,
    q_med: float = 0.5,
    q_high: float = 0.975,
) -> list[str]:
    """
    Plot baseline vs intervention uncertainty overlays for SEM and CDC outputs.
    """
    unit_map = {u.id: u for u in sem_output.inputs.units}
    available_states = [
        uid
        for uid in baseline.results
        if uid in intervention.results and uid in unit_map and getattr(unit_map[uid], "kind", None) == "state"
    ]
    available_states = sorted(available_states)
    if not available_states:
        raise ValueError("No shared state-level results found between baseline and intervention.")

    if state_ids is None:
        selected = available_states[:max_states]
    else:
        requested = list(dict.fromkeys(state_ids))
        selected = [uid for uid in requested if uid in available_states][:max_states]
        if not selected:
            raise ValueError("None of the requested states are available in both result sets.")

    sem_var_names = list(sem_output.inputs.v_names)
    sem_obs_years = np.asarray(sem_output.inputs.ts, dtype=float)
    sem_plot_vars = [v for v in SEM_PLOT_VARS if v in sem_var_names]
    epi_vars = EPI_PLOT_VARS

    for state_id in selected:
        unit = unit_map[state_id]
        b_res = baseline.results[state_id]
        i_res = intervention.results[state_id]

        b_sem = np.asarray([s.sem_trajectory for s in b_res.samples], dtype=float)
        i_sem = np.asarray([s.sem_trajectory for s in i_res.samples], dtype=float)
        sem_years = _extend_years(sem_obs_years, b_sem.shape[2])

        fig_sem, axes_sem = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
        fig_sem.suptitle(f"{state_id}: SEM baseline vs intervention")
        axes_sem_flat = axes_sem.ravel()
        for i, var in enumerate(sem_plot_vars):
            idx = sem_var_names.index(var)
            ax = axes_sem_flat[i]
            qb = np.quantile(b_sem[:, idx, :], [q_low, q_med, q_high], axis=0)
            qi = np.quantile(i_sem[:, idx, :], [q_low, q_med, q_high], axis=0)
            _add_forecast_background(ax, sem_years, sem_obs_years[-1])
            ax.fill_between(sem_years, qb[0], qb[2], alpha=0.12, color=BASELINE_COLOR, label="Baseline 95% CI")
            ax.plot(sem_years, qb[1], color=BASELINE_COLOR, linewidth=2.6, label="Baseline median")
            ax.fill_between(sem_years, qi[0], qi[2], alpha=0.12, color=INTERVENTION_COLOR, label="Intervention 95% CI")
            ax.plot(sem_years, qi[1], color=INTERVENTION_COLOR, linewidth=2.6, label="Intervention median")
            ax.scatter(
                sem_obs_years,
                np.asarray(unit.amis_values[idx], dtype=float),
                s=24,
                color=OBSERVED_COLOR,
                edgecolors="white",
                linewidths=0.8,
                label="Observed",
            )
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            _style_sem_axis(ax)
            ax.grid(alpha=0.3)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.legend(
            handles=[
                Patch(facecolor=BASELINE_COLOR, alpha=0.12, label="Baseline 95% CI"),
                Line2D([0], [0], color=BASELINE_COLOR, lw=2.6, label="Baseline median"),
                Patch(facecolor=INTERVENTION_COLOR, alpha=0.12, label="Intervention 95% CI"),
                Line2D([0], [0], color=INTERVENTION_COLOR, lw=2.6, label="Intervention median"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=OBSERVED_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=7,
                    label="Observed",
                ),
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=8,
        )
        fig_sem.tight_layout()
        plt.show()

        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC baseline vs intervention")
        years = np.asarray(b_res.years, dtype=float)
        for ax, var in zip(axes_epi, epi_vars):
            qb = b_res.get_quantiles(var, q=[q_low, q_med, q_high])
            qi = i_res.get_quantiles(var, q=[q_low, q_med, q_high])
            _add_forecast_background(ax, years, sem_obs_years[-1])
            ax.fill_between(years, qb[q_low], qb[q_high], alpha=0.12, color=BASELINE_COLOR, label="Baseline 95% CI")
            ax.plot(years, qb[q_med], color=BASELINE_COLOR, linewidth=2.6, label="Baseline median")
            ax.fill_between(years, qi[q_low], qi[q_high], alpha=0.12, color=INTERVENTION_COLOR, label="Intervention 95% CI")
            ax.plot(years, qi[q_med], color=INTERVENTION_COLOR, linewidth=2.6, label="Intervention median")

            raw_name = CDC_RAW_MAP[var]
            if unit.cdc_names is not None and unit.cdc_values is not None and unit.cdc_years is not None and raw_name in unit.cdc_names:
                raw_idx = unit.cdc_names.index(raw_name)
                raw_vals = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                raw_years = np.asarray(unit.cdc_years, dtype=float)
                mask = ~np.isnan(raw_vals)
                if np.any(mask):
                    ax.scatter(
                        raw_years[mask],
                        raw_vals[mask],
                        s=28,
                        marker="o",
                        color=OBSERVED_COLOR,
                        edgecolors="white",
                        linewidths=0.8,
                        label="CDC raw",
                    )

            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig_epi.tight_layout()
        plt.show()

    return selected


def plot_sem_j_violin(
    sem_loader,
    state_ids: list[str] | None = None,
    max_states: int = 5,
    nonzero_tol: float = 1e-12,
) -> list[str]:
    """
    Plot per-state violin plots for each non-zero J matrix entry across SEM samples.
    """
    geo_names = list(sem_loader.geo_names)
    if state_ids is None:
        selected = geo_names[:max_states]
    else:
        requested = list(dict.fromkeys(state_ids))
        selected = [uid for uid in requested if uid in geo_names][:max_states]
        if not selected:
            raise ValueError("None of the requested states are available in SEM samples.")

    j_samples = np.asarray(sem_loader.J_samples, dtype=float)  # (S, G, m, m)
    v_names = list(sem_loader.v_names)

    plotted = []
    for state_id in selected:
        g = geo_names.index(state_id)
        jg = j_samples[:, g, :, :]  # (S, m, m)
        nz_mask = np.any(np.abs(jg) > nonzero_tol, axis=0)
        nz_entries = np.argwhere(nz_mask)
        if nz_entries.size == 0:
            continue

        violins = [jg[:, i, j] for i, j in nz_entries]
        labels = [f"{v_names[i]}<-{v_names[j]}" for i, j in nz_entries]

        fig_w = min(26, max(12, 0.45 * len(labels) + 6))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5))
        parts = ax.violinplot(
            violins,
            showmeans=True,
            showmedians=False,
            showextrema=False,
            widths=0.9,
        )
        for body in parts["bodies"]:
            body.set_alpha(0.45)
            body.set_facecolor("tab:blue")
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.2)

        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
        ax.set_title(f"{state_id}: SEM J non-zero entry distributions")
        ax.set_ylabel("J value")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        plt.show()
        plotted.append(state_id)

    return plotted
