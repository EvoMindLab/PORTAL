# Changelog

## [2.0.0] - 2026-11-03

- License PORTAL-owned framework material under MIT, naming Danial Yazdani
  and PORTAL contributors; keep optimizer implementations and dependencies
  under separate terms, preserve upstream notices, and exclude the unlicensed
  author-derived dd-CMA-ES module from the public release.
- Replace the legacy MATLAB/Python split with a Python-only release; earlier
  versions remain available in Git history.
- Introduce PORTAL Studio's Generator, Experiment, and Education workspaces.
- Use a unified JSON format for single- and multi-block recipes.
- Add configurable suite-construction and Type 3 dataset-generation runners
  with parallel execution, progress reporting, and resume support.
- Release the 348-recipe catalog, S50, S20, disjoint Train30/Test30 suites,
  membership manifests, and small runnable examples.
- Use optimizer-performance terminology while accepting historical ISA-named
  inputs for compatibility.
- Refresh installation instructions and the user guide, and include
  self-contained software tests and third-party source credits.
