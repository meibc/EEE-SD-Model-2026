from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.epi.params import EpiOutput  # noqa: E402
from pipeline.projection import (  # noqa: E402
    ProjectionOutput,
    ProjectionResult,
    ProjectionSample,
)
from pipeline.scenario_results import (  # noqa: E402
    draw_rows,
    paired_endpoints,
    summary_rows,
)


def _output(dynamic_values, incidence_values):
    samples = []
    for draw, (dynamic, incidence) in enumerate(zip(dynamic_values, incidence_values)):
        epi = EpiOutput(
            unit_id="CA",
            years=np.array([2024, 2025, 2026]),
            prep_on_count=np.array([10.0, 11.0, 12.0]) + draw,
            incidence=np.asarray(incidence, dtype=float),
            diagnosed=np.array([1.0, 1.0, 1.0]),
            undiagnosed=np.array([2.0, 2.0, 2.0]),
        )
        samples.append(ProjectionSample(draw, draw, np.asarray(dynamic, dtype=float), epi))
    result = ProjectionResult("CA", samples, np.array([2024, 2025, 2026]))
    return ProjectionOutput({"CA": result}, np.array([2024, 2025, 2026]), ["stigma_ahs", "prep_used"])


class ProjectionResultSummaryTests(unittest.TestCase):
    def setUp(self):
        self.dynamic_years = np.array([2024, 2026])
        self.baseline = _output(
            [np.array([[0.20, 0.30], [0.30, 0.40]]), np.array([[0.25, 0.35], [0.35, 0.45]])],
            [[100.0, 90.0, 80.0], [110.0, 100.0, 90.0]],
        )
        self.scenario = _output(
            [np.array([[0.20, 0.15], [0.30, 0.50]]), np.array([[0.25, 0.20], [0.35, 0.55]])],
            [[100.0, 85.0, 70.0], [110.0, 95.0, 80.0]],
        )

    def test_extracts_dynamic_and_cumulative_epidemic_endpoints(self):
        endpoints = paired_endpoints(self.baseline, self.scenario, self.dynamic_years, 2026, 2025)
        dynamic, epidemic = endpoints["CA"]
        np.testing.assert_allclose(dynamic["prep_used"][0], [0.40, 0.45])
        np.testing.assert_allclose(dynamic["prep_used"][1], [0.50, 0.55])
        np.testing.assert_allclose(epidemic["incidence_cumulative"][0], [170.0, 190.0])
        np.testing.assert_allclose(epidemic["incidence_cumulative"][1], [155.0, 175.0])

    def test_summary_and_draw_rows_preserve_paired_differences(self):
        endpoints = paired_endpoints(self.baseline, self.scenario, self.dynamic_years, 2026, 2025)
        labels = {"scenario": "halve_ahs", "year": 2026}
        dynamic_summary, _ = summary_rows(endpoints, labels)
        prep = next(row for row in dynamic_summary if row["variable"] == "prep_used")
        self.assertAlmostEqual(prep["delta_p50"], 0.10)

        dynamic_draws, _ = draw_rows(endpoints, labels)
        prep_draws = [row for row in dynamic_draws if row["variable"] == "prep_used"]
        self.assertEqual(len(prep_draws), 2)
        self.assertTrue(all(abs(row["delta"] - 0.10) < 1e-12 for row in prep_draws))


if __name__ == "__main__":
    unittest.main()
