# Deprecated

Retired code kept for reference. Not part of the paper pipeline; it is not maintained and may not run
against current outputs (several read scenario folders whose draw-level files were removed on 2026-10-26).
See docs/analysis_decisions.md for the decisions behind each retirement.

| Script | What it did | Why retired |
|---|---|---|
| scripts/12_fixed_dose_scenarios.py | Fixed-dose (percentage-point) stigma scenarios | Dose-response and fixed-dose scenarios dropped from the paper |
| scripts/13_structural_sensitivity.py | AHS -> PrEP coupling strength and route decomposition | Superseded by the evidence constraint and static translation; routes not identified |
| scripts/13b_ahs_counterfactuals.py | AHS history counterfactual and AHS-free PrEP | Superseded framing; durability prior-driven |
| scripts/13c_ahs_prep_individual_bounds.R | Model vs individual-evidence diagnostic | Evidence is now a constraint in the fit (scripts/08b, pipeline/state_fit.py) |
| scripts/13d_system_decomposition.py | Projected change split into own momentum/drift vs the rest of the system | Not in the paper outputs; cited only in the decision log (projected AHS rise via PrEP -> AHS) |
| models/structural.py, tests/test_structural.py | Post-fit analysis of J: impulse coupling, source counterfactuals, trajectory decomposition (was models/sbm/structural.py) | Used only by scripts 13, 13b and 13d; these scripts import it as models.sbm.structural |
| scripts/15_main_figures.py | Previous main figures and Table 1 | Replaced by scripts/results_*.py |
| scripts/16_supplementary_outputs.py | Previous supplement tables, data tables and figures | Replaced by scripts/results_supplement.py and the results scripts |
