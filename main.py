#!/usr/bin/env python3
from __future__ import annotations

"""EEE-SD-Model-2026 staged workflow entrypoint.

The legacy monolithic `run_pipeline` path has been retired. Use the numbered
scripts directly so each long-running stage is explicit and reproducible.
"""

STAGED_WORKFLOW = [
    "scripts/01_generate_sem_input_draws.py",
    "scripts/02_fit_sem_draws.py",
    "scripts/03_export_dashboard_standardized_v8.py",
    "scripts/04_run_sem_scenarios.py",
    "scripts/05_fit_hierarchical_epi_model.py",
]


def main() -> list[str]:
    print("Legacy monolithic pipeline retired. Run stages explicitly:")
    for stage in STAGED_WORKFLOW:
        print(f"  python {stage}")
    return STAGED_WORKFLOW


if __name__ == "__main__":
    main()
