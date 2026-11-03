# PORTAL End-to-End Suite Runner

The repository-root `run_suite_pipeline.py` entry point executes the complete
recipe-curation workflow from canonical unified PORTAL JSON files to a
materialized benchmark suite. The Python implementation lives in this package;
the root file is intentionally only a stable command-line entry point.

## Input contract

`run.recipes_root` may be flat or nested. Directory names have no semantic
role. The runner recursively discovers JSON files and reads recipe identity,
family membership, dimension, and block count from each payload. Before any
expensive work, it rejects:

- malformed or legacy composite-envelope JSONs;
- missing or inconsistent recipe/family metadata;
- duplicate recipe identifiers, including case-only duplicates;
- recipe identifiers that are unsafe as portable filenames;
- mixed dimensions in one curation run; and
- any payload that cannot be loaded by PORTAL's canonical unified-instance
  loader.

The final check uses the same loader used by script-based evaluation and PORTAL
Studio. Therefore, the runner does not introduce a second recipe schema and
does not change GUI behavior.

## Running the pipeline

For a short installation check, use `examples/suite_smoke.toml` from the
repository root. The full `suite_pipeline.example.toml` schedules 26,796
optimizer runs and is not a quick-start workload. Raw paper measurement tables
are not bundled: the runner generates its own inputs and outputs locally.

Copy `suite_pipeline.example.toml`, edit the copy, and first inspect the plan:

```powershell
python run_suite_pipeline.py --config suite_pipeline.toml --dry-run
```

Then run it:

```powershell
python run_suite_pipeline.py --config suite_pipeline.toml
```

The normal stage order is:

```text
validate -> ela -> performance -> prepare -> deduplicate -> select -> materialize
```

When `train_test.enabled = true`, the following stages are appended:

```text
train_test -> materialize_train_test
```

Useful command-line controls are:

```text
--from-stage STAGE       start at a selected stage
--to-stage STAGE         stop after a selected stage
--force                  rerun every selected stage
--force-stage STAGE      rerun one stage; the option may be repeated
--dry-run                validate and display commands without writing outputs
```

## Configuration reference

All paths are resolved relative to the repository root unless absolute.
Unknown or misspelled TOML keys are rejected.

Use `[performance]` for algorithm-performance measurement and
`selection.performance_weight` for its distance weight. Older `[isa]` and
`selection.isa_weight` settings are accepted as compatibility aliases;
conflicting old/new settings are rejected. The old `isa` stage token is also
accepted, but plans, logs, and newly generated outputs use `performance`.
Low-level readers can load existing ISA-named CSVs without changing their bytes.

### Run

| Key | Meaning |
|---|---|
| `run.name` | Human-readable run identifier recorded in metadata. |
| `run.recipes_root` | Root containing unified recipe JSONs. |
| `run.output_root` | Isolated output directory for this run. It must be outside `recipes_root`. |
| `run.resume` | Permit stage-level resume when signatures and outputs still match. |

### ELA measurement

| Key | Meaning |
|---|---|
| `ela.sample_factor` | Improved-LHS sample count per dimension. |
| `ela.seed` | Common ELA sampling seed. |
| `ela.max_workers` | ELA worker processes. Zero selects the existing automatic default of 90% of available CPU cores. |
| `ela.heartbeat_percent` | Percentage interval for progress messages. |
| `ela.heartbeat_alive_seconds` | Emit an alive message after this many quiet seconds; zero disables quiet-time messages. |
| `ela.evaluation_device` | PORTAL evaluation backend: `auto`, `numpy`, `cpu`, `cuda`, or `mps`. |

### Performance measurement

| Key | Meaning |
|---|---|
| `performance.algorithms` | Ordered algorithm-module names used to construct the behavioral representation. |
| `performance.runs` | Independent runs per recipe and algorithm. |
| `performance.max_fe_per_dimension` | Function-evaluation budget per dimension. |
| `performance.checkpoints` | Strictly increasing positive-integer absolute FE checkpoints retained in the raw table. Fractional values are rejected before measurement. |
| `performance.base_seed` | Base in `seed = base + 100 * algorithm_id + run_id`. |
| `performance.max_parallel_runs` | Concurrent algorithm runs. Zero selects 90% of available CPU cores. |
| `performance.checkpoint_every` | Persist the resumable run table after this many newly completed runs. |
| `performance.heartbeat_percent` | Percentage interval for progress messages. |
| `performance.heartbeat_alive_seconds` | Emit an alive message after this many quiet seconds; zero disables it. |
| `performance.evaluation_device` | PORTAL evaluation backend. |
| `performance.resume` | Reuse compatible run-level rows from `performance_runs.partial.csv`. |

ELA and Performance use process-based parallelism. Each worker limits BLAS/OpenMP-style
numeric libraries to one thread, overriding inherited thread settings during
measurement and restoring the caller's settings afterwards. Explicit worker
values are capped at the number of available CPU cores; Performance also respects
Windows' limit of 61 process-executor workers.
ELA reuses the common unit sampling design but caches dispersion reference
statistics separately for each distinct recipe search domain.
The vendored `pflacco` source used for ELA is included in the ELA stage
fingerprint, so changing that implementation invalidates stale ELA outputs.

The curation measurements intentionally use the recipes directly (Type 0).
Types 1--3 serve repeated-run and augmentation studies and are not selectable
for the recipe-level curation representation.

### Feature preparation

| Key | Meaning |
|---|---|
| `features.log_epsilon` | Positive offset in the Performance log-error transform. |
| `features.iqr_floor` | Global ELA near-constant IQR threshold. |
| `features.correlation_threshold` | Absolute Pearson-correlation pruning threshold. |
| `features.mad_floor` | Lower bound used for robust MAD scaling. |
| `features.winsor_lower` | Lower ELA winsorization quantile. |
| `features.winsor_upper` | Upper ELA winsorization quantile. |

### De-duplication

| Key | Meaning |
|---|---|
| `deduplication.rule` | Conservative cross-representation rule. The finalized runner requires `and`. |
| `deduplication.primary_tau` | Primary P95-normalized distance threshold. |
| `deduplication.sensitivity_thresholds` | Thresholds used for the sensitivity audit; the primary value must be included. |
| `deduplication.normalization` | Finalized P95 distance normalization; required to be `p95`. |
| `deduplication.mis_backend` | Maximum-independent-set backend; the finalized runner requires PuLP/CBC. |

The AND graph is formed over every recipe pair, including within-family and
cross-family pairs. The maximum independent set is solved in two stages:
maximum cardinality first, followed by the deterministic isolation-based
tie-break among maximum-cardinality solutions.

### Fixed-cardinality selection

| Key | Meaning |
|---|---|
| `selection.size` | Requested suite cardinality. Output filenames track this value. |
| `selection.families` | Ordered family labels used by all percentage arrays. |
| `selection.family_lower_percent` | Per-family lower percentage bounds. |
| `selection.family_upper_percent` | Per-family upper percentage bounds. |
| `selection.ela_weight` | ELA contribution to joint distance. |
| `selection.performance_weight` | Performance contribution to joint distance. |
| `selection.trim_fraction` | Symmetric trim fraction for robust mean distance scaling. |
| `selection.spacing_lambda` | Weight on mean nearest-neighbor spacing in local refinement. |
| `selection.spacing_mu` | Weight on minimum pairwise spacing in local refinement. |
| `selection.max_swap_passes` | Maximum best-improvement local-swap passes; zero skips refinement. |

Percentage bounds are converted separately for the requested cardinality:
lower counts use `floor(K * p / 100)` and upper counts use
`ceil(K * p / 100)`, after which upper counts are capped by pool availability.
Setting every lower bound to 0% and every upper bound to 100% gives
unconstrained global selection. Equal lower and upper percentages impose an
exact family count when `K * p / 100` is integral; otherwise they produce the
one-count interval implied by floor/ceiling conversion.

Family constraints do not run selection independently by family. Both greedy
coverage selection and local refinement operate on the complete deduplicated
pool, so cross-family distances remain part of every decision.

### Optional train/test construction

| Key | Meaning |
|---|---|
| `train_test.enabled` | Append disjoint train/test construction. |
| `train_test.train_size` | Requested training-suite cardinality. |
| `train_test.test_size` | Requested test-suite cardinality. |
| `train_test.split_seed` | Deterministic within-family candidate-split seed. |
| `train_test.train_family_lower_percent` | Train-side lower family percentages. |
| `train_test.train_family_upper_percent` | Train-side upper family percentages. |
| `train_test.test_family_lower_percent` | Test-side lower family percentages. |
| `train_test.test_family_upper_percent` | Test-side upper family percentages. |

Separate side-specific bounds support different mixtures and family-holdout
protocols. A zero upper bound makes a family ineligible on that side. Shared
families are divided as evenly as possible by default. If that balanced
candidate split cannot support the requested train/test sizes and percentage
bounds, PuLP/CBC changes the per-family allocation by the minimum number of
recipes needed for feasibility; it still does not use ELA/Performance geometry.

## Long-run progress and logs

Every child process runs with unbuffered UTF-8 output. Console messages are
mirrored immediately into append-only stage logs:

```text
OUTPUT_ROOT/logs/ela.log
OUTPUT_ROOT/logs/performance.log
OUTPUT_ROOT/logs/prepare.log
OUTPUT_ROOT/logs/deduplicate.log
OUTPUT_ROOT/logs/select.log
...
```

`OUTPUT_ROOT/logs/pipeline.log` records timestamped outer-run and stage events.
Each command attempt has a timestamped header, exit code, and elapsed time, so
rerunning a stage preserves rather than overwrites its earlier log. On
PowerShell, a running stage can be monitored with:

```powershell
Get-Content OUTPUT_ROOT\logs\performance.log -Wait
```

ELA reports completed feature units, recipes, elapsed time, and ETA. Performance
reports completed runs, elapsed time, ETA, periodic worker-alive messages, and
checkpoint writes. Selection reports greedy additions and local-swap progress.
De-duplication reports every threshold and graph solve.
If the runner is interrupted from the console, it terminates the active child,
records the stage as interrupted, and leaves a compatible Performance partial
checkpoint available for the next same-configuration run.

## Resume integrity

Stage resume is conservative. A stage is skipped only when all of the following
still match:

- command and environment;
- hashes of declared input files and recipe JSON contents;
- hashes of stage implementation sources;
- Python executable/version and relevant package versions;
- completion status; and
- hashes of every declared output file, including every materialized JSON.

If a generated output is edited, truncated, removed, or replaced, the stage is
rerun. Performance additionally resumes at run level from its isolated partial CSV,
which is useful after interruption during the longest stage. The partial CSV
is retained only when its outer Performance stage signature still matches the recipe
catalog, algorithm and PORTAL sources, numerical configuration, Python
environment, and relevant dependency versions. A partial checkpoint from an
incompatible or explicitly forced run is removed before execution.

## Output layout

Each configured output root contains:

```text
metadata/             resolved config, recipe manifest, run manifest, state
logs/                 append-only master and per-stage logs
raw_features/         raw ELA and Performance tables plus Performance partial checkpoint
cleaned_features/     cleaned/scaled ELA and Performance representations and reports
deduplication/        threshold sweep, pair audits, MIS pools, figures, report
selection/            selected-list CSV and selection report
selected_recipes/     materialized unified JSONs for the requested suite
train_test/           optional pools, selected lists, report, and JSON folders
```

For example, `selection.size = 20` produces
`benchmark20_selected.csv` and `benchmark20_selection_report.md`.
