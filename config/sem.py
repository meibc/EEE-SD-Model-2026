from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SEMConfig:
    """SEM model configuration."""

    # Data
    data_path: Path = Path("Factor Analysis v3.xlsx")

    # Variables
    v_names: list[str] = field(default_factory=lambda: [
        "stigma_ahs", "stigma_gss", "stigma_family",
        "out_gid", "seehcp", "risk_behavior", "prep_used",
        "hivtest12",
    ])

    # Centered-logit/variable-level SEM config. This single table contains
    # reference probabilities and optional exogenous drift priors.
    use_centered_logit: bool = True
    variable_config_path: Path = Path("config/sem_variable_config_v3.csv")

    # Edge-level structural priors for the J matrix. One row is one directional
    # relationship, using J[target, source] orientation.
    structural_prior_path: Path = Path("config/sem_structural_priors_v3.csv")

    # Simulation
    n_steps: int = 6
    n_forecast: int = 5

    def _variable_config(self) -> pd.DataFrame:
        return pd.read_csv(self.variable_config_path)

    def reference_probs(self) -> np.ndarray:
        if not self.use_centered_logit:
            return np.full(len(self.v_names), 0.5, dtype=float)
        df = self._variable_config()
        mapping = dict(zip(df["variable"].astype(str), df["reference_prob"].astype(float)))
        ref = np.array([mapping[v] for v in self.v_names], dtype=float)
        if not np.all((ref > 0) & (ref < 1)):
            raise ValueError("All centered-logit reference probabilities must be in (0, 1)")
        return ref

    def structural_prior_arrays(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return sign mask, prior mean, prior tau arrays from structural-prior CSV.

        Expected columns are source, target, sign, prior_mean, prior_tau. Missing
        or empty priors are allowed; sign still defines the hard structural mask.
        """
        df = pd.read_csv(self.structural_prior_path)
        m = len(self.v_names)
        idx = {v: i for i, v in enumerate(self.v_names)}
        sign = np.zeros((m, m), dtype=int)
        mean = np.full((m, m), np.nan, dtype=float)
        tau = np.full((m, m), np.nan, dtype=float)
        for row in df.to_dict(orient="records"):
            src = str(row["source"])
            tgt = str(row["target"])
            i = idx[tgt]
            j = idx[src]
            if i == j:
                continue
            sign_val = row.get("sign", np.nan)
            if pd.notna(sign_val) and str(sign_val).strip() != "":
                sign[i, j] = int(np.sign(float(sign_val)))
            elif pd.notna(row.get("prior_mean", np.nan)) and str(row.get("prior_mean", "")).strip() != "":
                sign[i, j] = int(np.sign(float(row["prior_mean"])))
            pm = row.get("prior_mean", np.nan)
            pt = row.get("prior_tau", np.nan)
            if pd.notna(pm) and str(pm).strip() != "":
                mean[i, j] = float(pm)
            if pd.notna(pt) and str(pt).strip() != "":
                tau[i, j] = float(pt)
        return sign, mean, tau

    def drift_prior_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return enabled mask, drift prior mean, drift prior tau arrays."""
        m = len(self.v_names)
        enabled = np.zeros(m, dtype=bool)
        mean = np.zeros(m, dtype=float)
        tau = np.full(m, np.nan, dtype=float)
        df = self._variable_config()
        idx = {v: i for i, v in enumerate(self.v_names)}
        for row in df.to_dict(orient="records"):
            var = str(row["variable"])
            k = idx[var]
            enabled[k] = bool(int(row["drift_enabled"]))
            pm = row.get("drift_prior_mean", np.nan)
            pt = row.get("drift_prior_tau", np.nan)
            if pd.notna(pm) and str(pm).strip() != "":
                mean[k] = float(pm)
            if pd.notna(pt) and str(pt).strip() != "":
                tau[k] = float(pt)
            if enabled[k] and (not np.isfinite(tau[k]) or tau[k] <= 0):
                raise ValueError(f"Enabled drift variable {var} must have positive drift_prior_tau")
        return enabled, mean, tau
