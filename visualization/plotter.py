from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from engine.results import JointOutput, RunOutput, UncertaintyResult


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
        "stigma_ahs",
        "stigma_gss",
        "stigma_family",
        "out_gid",
        "seehcp",
        "risk_behavior",
        "hivtest12",
        "prep_used",
    ]
    sem_plot_vars = [v for v in sem_plot_vars if v in sem_var_names]
    if not sem_plot_vars:
        raise ValueError("No requested SEM plotting variables found in SEM outputs.")

    cdc_raw_map = {
        "prep_on_count": "PrEP",
        "incidence": "Estimated HIV incidence (MSM)",
        "diagnosed": "HIV diagnoses",
    }
    epi_vars = ["diagnosed", "incidence", "prep_on_count"]

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
            ax.plot(
                sem_hist_years,
                np.asarray(unit.amis_values[idx], dtype=float),
                marker="o",
                linewidth=1.5,
                label="Observed",
            )
            ax.plot(
                sem_pred_years,
                sem_traj[idx],
                linewidth=2.0,
                label="SEM trajectory",
            )
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.tight_layout()
        plt.show()

        epi_years = np.asarray(result.cdc_output.years, dtype=float)
        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC Epi outputs")
        for ax, var in zip(axes_epi, epi_vars):
            vals = np.asarray(getattr(result.cdc_output, var), dtype=float)
            ax.plot(epi_years, vals, linewidth=2.0, label="Epi")

            if unit.cdc_names is not None and unit.cdc_values is not None and unit.cdc_years is not None:
                raw_name = cdc_raw_map[var]
                if raw_name in unit.cdc_names:
                    raw_idx = unit.cdc_names.index(raw_name)
                    raw_values = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                    raw_years = np.asarray(unit.cdc_years, dtype=float)
                    mask = ~np.isnan(raw_values)
                    if np.any(mask):
                        ax.scatter(
                            raw_years[mask],
                            raw_values[mask],
                            s=24,
                            marker="o",
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
    sem_plot_vars = [
        "stigma_ahs",
        "stigma_gss",
        "stigma_family",
        "out_gid",
        "seehcp",
        "risk_behavior",
        "hivtest12",
        "prep_used",
    ]
    sem_plot_vars = [v for v in sem_plot_vars if v in sem_var_names]
    epi_vars = ["diagnosed", "incidence", "prep_on_count"]
    cdc_raw_map = {
        "prep_on_count": "PrEP",
        "incidence": "Estimated HIV incidence (MSM)",
        "diagnosed": "HIV diagnoses",
    }

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
            ax.plot(sem_years, b_sem[idx], color="gray", linewidth=2.0, label="Baseline")
            ax.plot(sem_years, i_sem[idx], color="tab:blue", linewidth=2.0, label="Intervention")
            ax.scatter(sem_obs_years, np.asarray(unit.amis_values[idx], dtype=float), s=16, color="black", label="Observed")
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.tight_layout()
        plt.show()

        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC baseline vs intervention")
        years = np.asarray(b_res.cdc_output.years, dtype=float)
        for ax, var in zip(axes_epi, epi_vars):
            yb = np.asarray(getattr(b_res.cdc_output, var), dtype=float)
            yi = np.asarray(getattr(i_res.cdc_output, var), dtype=float)
            ax.plot(years, yb, color="gray", linewidth=2.0, label="Baseline")
            ax.plot(years, yi, color="tab:red", linewidth=2.0, label="Intervention")

            raw_name = cdc_raw_map[var]
            if unit.cdc_names is not None and unit.cdc_values is not None and unit.cdc_years is not None and raw_name in unit.cdc_names:
                raw_idx = unit.cdc_names.index(raw_name)
                raw_vals = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                raw_years = np.asarray(unit.cdc_years, dtype=float)
                mask = ~np.isnan(raw_vals)
                if np.any(mask):
                    ax.scatter(raw_years[mask], raw_vals[mask], s=24, marker="o", color="black", label="CDC raw")

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
    uncertainty: dict[str, UncertaintyResult],
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
        for uid in uncertainty
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

    sem_plot_vars = [
        "stigma_ahs",
        "stigma_gss",
        "stigma_family",
        "out_gid",
        "seehcp",
        "risk_behavior",
        "hivtest12",
        "prep_used",
    ]
    sem_plot_vars = [v for v in sem_plot_vars if v in sem_var_names]
    if not sem_plot_vars:
        raise ValueError("No requested SEM plotting variables found in SEM outputs.")

    cdc_raw_map = {
        "prep_on_count": "PrEP",
        "incidence": "Estimated HIV incidence (MSM)",
        "diagnosed": "HIV diagnoses",
    }
    epi_vars = ["diagnosed", "incidence", "prep_on_count"]

    for state_id in selected:
        unit = unit_map[state_id]
        u_res = uncertainty[state_id]

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
        sem_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, var in enumerate(sem_plot_vars):
            idx = sem_var_names.index(var)
            color = sem_colors[i % len(sem_colors)]
            ax = axes_sem_flat[i]
            ax.fill_between(
                sem_years,
                sem_q[var][0],
                sem_q[var][2],
                alpha=0.2,
                color=color,
                label="95% CI",
            )
            ax.plot(sem_years, sem_q[var][1], color=color, linewidth=2.0, label="Median")
            ax.scatter(
                sem_obs_years,
                np.asarray(unit.amis_values[idx], dtype=float),
                color=color,
                s=18,
                alpha=0.8,
                label="Observed",
            )
            if sem_obs_years.size > 0:
                ax.axvline(sem_obs_years[-1], linestyle="--", alpha=0.5, color="gray")
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.tight_layout()
        plt.show()

        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC Epi uncertainty")
        for ax, (var, color) in zip(
            axes_epi,
            [("diagnosed", "tab:green"), ("incidence", "tab:red"), ("prep_on_count", "tab:purple")],
        ):
            qv = epi_q[var]
            ax.fill_between(
                epi_years,
                qv[q_low],
                qv[q_high],
                alpha=0.15,
                color=color,
                label="95% CI",
            )
            ax.plot(
                epi_years,
                qv[q_med],
                color=color,
                linewidth=2.0,
                label="Median",
            )

            raw_name = cdc_raw_map.get(var)
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
                        s=24,
                        color=color,
                        marker="o",
                        alpha=0.9,
                        label="CDC raw",
                    )

            if sem_obs_years.size > 0:
                ax.axvline(sem_obs_years[-1], linestyle="--", alpha=0.5, color="gray")
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
    baseline: dict[str, UncertaintyResult],
    intervention: dict[str, UncertaintyResult],
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
        for uid in baseline
        if uid in intervention and uid in unit_map and getattr(unit_map[uid], "kind", None) == "state"
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
    sem_plot_vars = [
        "stigma_ahs",
        "stigma_gss",
        "stigma_family",
        "out_gid",
        "seehcp",
        "risk_behavior",
        "hivtest12",
        "prep_used",
    ]
    sem_plot_vars = [v for v in sem_plot_vars if v in sem_var_names]
    epi_vars = ["diagnosed", "incidence", "prep_on_count"]
    cdc_raw_map = {
        "prep_on_count": "PrEP",
        "incidence": "Estimated HIV incidence (MSM)",
        "diagnosed": "HIV diagnoses",
    }

    for state_id in selected:
        unit = unit_map[state_id]
        b_res = baseline[state_id]
        i_res = intervention[state_id]

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
            ax.fill_between(sem_years, qb[0], qb[2], alpha=0.15, color="gray", label="Baseline 95% CI")
            ax.plot(sem_years, qb[1], color="gray", linewidth=2.0, label="Baseline median")
            ax.fill_between(sem_years, qi[0], qi[2], alpha=0.2, color="tab:blue", label="Intervention 95% CI")
            ax.plot(sem_years, qi[1], color="tab:blue", linewidth=2.0, label="Intervention median")
            ax.scatter(sem_obs_years, np.asarray(unit.amis_values[idx], dtype=float), s=16, color="black", label="Observed")
            ax.set_title(var)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        for j in range(len(sem_plot_vars), len(axes_sem_flat)):
            axes_sem_flat[j].axis("off")
        fig_sem.tight_layout()
        plt.show()

        fig_epi, axes_epi = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        fig_epi.suptitle(f"{state_id}: CDC baseline vs intervention")
        years = np.asarray(b_res.years, dtype=float)
        for ax, var in zip(axes_epi, epi_vars):
            qb = b_res.get_quantiles(var, q=[q_low, q_med, q_high])
            qi = i_res.get_quantiles(var, q=[q_low, q_med, q_high])
            ax.fill_between(years, qb[q_low], qb[q_high], alpha=0.15, color="gray", label="Baseline 95% CI")
            ax.plot(years, qb[q_med], color="gray", linewidth=2.0, label="Baseline median")
            ax.fill_between(years, qi[q_low], qi[q_high], alpha=0.2, color="tab:red", label="Intervention 95% CI")
            ax.plot(years, qi[q_med], color="tab:red", linewidth=2.0, label="Intervention median")

            raw_name = cdc_raw_map[var]
            if unit.cdc_names is not None and unit.cdc_values is not None and unit.cdc_years is not None and raw_name in unit.cdc_names:
                raw_idx = unit.cdc_names.index(raw_name)
                raw_vals = np.asarray(unit.cdc_values[raw_idx], dtype=float)
                raw_years = np.asarray(unit.cdc_years, dtype=float)
                mask = ~np.isnan(raw_vals)
                if np.any(mask):
                    ax.scatter(raw_years[mask], raw_vals[mask], s=24, marker="o", color="black", label="CDC raw")

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
