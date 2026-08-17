#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dashboard_export import DashboardExportConfig, export_dashboard_standardized_v8


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dashboard-compatible v8 SEM parameter artifact")
    parser.add_argument("--base-standardized-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE-Model-Dashboard-Integration/standardized_input_v7.npz"))
    parser.add_argument("--sem-fits-path", type=Path, default=Path("output/sem_fits_centered_drift_v5.npz"))
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--output-path", type=Path, default=Path("output/standardized_input_v8.npz"))
    parser.add_argument("--schema-report-path", type=Path, default=Path("output/standardized_input_v8_schema.csv"))
    args = parser.parse_args()

    path = export_dashboard_standardized_v8(
        DashboardExportConfig(
            base_standardized_path=args.base_standardized_path,
            sem_fits_path=args.sem_fits_path,
            data_path=args.data_path,
            output_path=args.output_path,
            schema_report_path=args.schema_report_path,
        )
    )
    print(f"Wrote dashboard standardized v8 artifact: {path}")
    print(f"Wrote schema report: {args.schema_report_path}")


if __name__ == "__main__":
    main()
