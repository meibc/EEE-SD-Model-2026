from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DynamicConfig:
    """Dynamic model configuration: measures, pathway and persistence priors, drift priors."""

    # Variables
    v_names: list[str] = field(default_factory=lambda: [
        "stigma_ahs", "stigma_gss", "stigma_family",
        "out_gid", "seehcp", "risk_behavior", "prep_used",
        "hivtest12",
    ])

    # Centered-logit/variable-level config. This single table contains
    # reference probabilities and optional exogenous drift priors.
    variable_config_path: Path = Path(__file__).resolve().parent / "dynamic_variables.csv"

    # Edge-level structural priors for the J matrix. One row is one directional
    # relationship, using J[target, source] orientation.
    structural_prior_path: Path = Path(__file__).resolve().parent / "dynamic_pathway_priors.csv"

    # Selected diagonal persistence priors. These regularize own-variable
    # persistence separately from directional structural pathways.
    diagonal_prior_path: Path = Path(__file__).resolve().parent / "dynamic_persistence_priors.csv"

    def _variable_config(self) -> pd.DataFrame:
        df = pd.read_csv(self.variable_config_path)
        names = df["variable"].astype(str)
        if names.duplicated().any():
            raise ValueError(f"Duplicate variables in {self.variable_config_path.name}: {sorted(set(names[names.duplicated()]))}")
        missing, extra = set(self.v_names) - set(names), set(names) - set(self.v_names)
        if missing or extra:
            raise ValueError(f"{self.variable_config_path.name} must list exactly the model's measures; "
                             f"missing {sorted(missing)}, extra {sorted(extra)}")
        return df

    def reference_probs(self) -> np.ndarray:
        df = self._variable_config()
        mapping = dict(zip(df["variable"].astype(str), df["reference_prob"].astype(float)))
        ref = np.array([mapping[v] for v in self.v_names], dtype=float)
        if not np.all((ref > 0) & (ref < 1)):
            raise ValueError("All centered-logit reference probabilities must be in (0, 1)")
        return ref

    def structural_prior_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sign mask, prior mean, prior tau arrays from structural-prior CSV.

        Expected columns are source, target, sign, prior_mean, prior_tau. Every row
        needs a sign: 1 or -1 constrains the pathway's direction, "both" permits it
        without a sign constraint (sign 0; see allowed_mask). Missing or empty priors
        are allowed.
        """
        sign, mean, tau, _ = self._structural_table()
        return sign, mean, tau

    def allowed_mask(self) -> np.ndarray:
        """Off-diagonal pathways that are estimated (signed or "both")."""
        return self._structural_table()[3]

    def _structural_table(self):
        df = pd.read_csv(self.structural_prior_path)
        m = len(self.v_names)
        idx = {v: i for i, v in enumerate(self.v_names)}
        sign = np.zeros((m, m), dtype=int)
        allowed = np.zeros((m, m), dtype=bool)
        mean = np.full((m, m), np.nan, dtype=float)
        tau = np.full((m, m), np.nan, dtype=float)
        pairs = list(zip(df["source"].astype(str), df["target"].astype(str)))
        duplicated = sorted({pair for pair in pairs if pairs.count(pair) > 1})
        if duplicated:
            raise ValueError(f"Duplicate pathways in {self.structural_prior_path.name}: {duplicated}")
        for row in df.to_dict(orient="records"):
            src = str(row["source"])
            tgt = str(row["target"])
            unknown = [v for v in (src, tgt) if v not in idx]
            if unknown:
                raise ValueError(f"Pathway {src} -> {tgt} in {self.structural_prior_path.name} uses unknown measure(s) {unknown}")
            i = idx[tgt]
            j = idx[src]
            if i == j:
                continue
            sign_val = str(row.get("sign", "")).strip().lower()
            if sign_val == "both":
                allowed[i, j] = True
            elif sign_val in ("1", "-1", "1.0", "-1.0"):
                sign[i, j] = int(float(sign_val))
                allowed[i, j] = True
            else:
                raise ValueError(f"Pathway {src} -> {tgt} needs a sign of 1, -1 or both (got {row.get('sign')!r})")
            pm = row.get("prior_mean", np.nan)
            pt = row.get("prior_tau", np.nan)
            if pd.notna(pm) and str(pm).strip() != "":
                mean[i, j] = float(pm)
            if pd.notna(pt) and str(pt).strip() != "":
                tau[i, j] = float(pt)

        if self.diagonal_prior_path.exists():
            diag_df = pd.read_csv(self.diagonal_prior_path)
            names = diag_df["variable"].astype(str)
            if names.duplicated().any() or not names.isin(list(idx)).all():
                raise ValueError(f"{self.diagonal_prior_path.name}: duplicate or unknown measures "
                                 f"{sorted(set(names[names.duplicated() | ~names.isin(list(idx))]))}")
            for row in diag_df.to_dict(orient="records"):
                var = str(row["variable"])
                k = idx[var]
                pm = row.get("prior_mean", np.nan)
                pt = row.get("prior_tau", np.nan)
                if pd.notna(pm) and str(pm).strip() != "":
                    mean[k, k] = float(pm)
                if pd.notna(pt) and str(pt).strip() != "":
                    tau[k, k] = float(pt)
        return sign, mean, tau, allowed

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
