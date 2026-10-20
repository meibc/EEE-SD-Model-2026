# Consolidation verification

- Current research export: 17 primary states and 500 fitted SEM draws.
- Reference probabilities copied from the completed empirical fifth-percentile fit.
- Baseline and three level-reduction scenarios: 10,240 numerical values matched exactly between the copied implementation and canonical implementation (GA and TN, eight samples each, seed 123).
- All 91 Python source files parsed successfully.
- Required configuration CSVs are visible to Git; generated bundles and rollback files are ignored.
- Local backward links from the older research output directories resolve successfully.
- Nine generated binary artifacts removed from the canonical Git index; files retained locally.
- Saved displays cleared from seven exploratory notebooks, with every cell source preserved; 173,962,490 bytes removed from working files.

No commits, remote pushes, or history rewrites were performed. Numerical comparisons cover migration behavior, not external model validation.

Remaining production work: assemble the complete upstream data run, remove residual machine-specific data paths, rebase paper figure inputs on the current scenario set, and validate the separate all-state dashboard release.

Full current-scenario run completed: three scenarios × 17 states × 500 samples, to 2036, with the three-step ramp. Export alignment/failure checks passed (three tests). Older Git objects packed from 1.01 GiB to 740.98 MiB without expiring unreachable objects.
