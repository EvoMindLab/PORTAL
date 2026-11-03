"""ELA (Exploratory Landscape Analysis) compute layer for the Generator tab.

Third-party feature/sampling implementation: PORTAL's adapted pflacco subset,
https://github.com/Reiyan/pflacco (MIT; see pflacco/LICENSE).
Reference: R. P. Prager and H. Trautmann (2024), Pflacco: Feature-Based
Landscape Analysis of Continuous and Constrained Optimization Problems in
Python, Evolutionary Computation 32(3), 211-216.
https://doi.org/10.1162/evco_a_00341
This module is PORTAL integration code; the feature algorithms are credited
to pflacco and the original methods referenced in its function docstrings.

Mirrors the feature-group selection used in the research pipeline
(``run_ela.py``) so the Studio's results are directly comparable with the
suite-level measurements produced by the reusable curation runner.

Design choices:

- **Feature groups.** Ten canonical pflacco groups, same as ``run_ela.py``.
  Each group is a callable that returns ``dict[feature_name, float]``;
  ``compute_ela`` runs every selected group, merges the dicts, and tags
  any group that raised so the caller can show the failure in the UI.
- **Sampling.** A single Latin Hypercube sample of ``sample_factor * d``
  points in ``[bounds_min, bounds_max]^d``, drawn with ``pflacco.sampling.
  create_initial_sample`` using the provided seed. The same design feeds
  every selected group (no re-sampling per group).
- **Excel output.** ``save_ela_xlsx`` writes one row per composite with
  failure-resilient NaN cells for any feature whose group raised.

This Qt-independent compute layer supports headless use and background
execution by the GUI panel.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from portal_studio.algorithms.random_state import isolate_random_state
from portal_studio.validation import validate_instance_for_gui

if TYPE_CHECKING:
    import pandas


# Canonical feature-group order — matches run_ela.py line-for-line so the
# Studio output is directly comparable to the suite-level ELA tables.
FEATURE_GROUPS: List[str] = [
    "ela_meta",
    "ela_distribution",
    "ela_level",
    "pca",
    "nbc",
    "dispersion",
    "information_content",
    "fitness_distance",
    "ela_conv",
    "gradient",
]
# Groups that are expensive (extra function evaluations) — default to off in the UI.
SLOW_GROUPS: frozenset = frozenset({"ela_conv", "gradient"})


@dataclass
class ELAResult:
    """Outcome of one composite's ELA run."""
    label: str
    features: Dict[str, float] = field(default_factory=dict)
    failed_groups: List[str] = field(default_factory=list)
    d: int = 0
    J: int = 0


class ELAComputationCanceled(RuntimeError):
    """Raised when a caller requests cooperative ELA cancellation."""


def _raise_if_stopped(should_stop: Optional[Callable[[], bool]]) -> None:
    if should_stop is not None and should_stop():
        raise ELAComputationCanceled("ELA run canceled")


def _evaluate_one(x: np.ndarray, instance: Dict[str, Any]) -> float:
    """Evaluate one point through the canonical unified fitness function."""
    from portal_core.fitness import fitness
    return float(np.squeeze(fitness(x, instance)))


def _evaluate_batch(X: np.ndarray, instance: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray([_evaluate_one(row, instance) for row in arr], dtype=float)


@isolate_random_state
def compute_ela(instance: Dict[str, Any],
                groups: Optional[Sequence[str]] = None,
                *,
                sample_factor: int = 1000,
                seed: int = 42,
                should_stop: Optional[Callable[[], bool]] = None,
                progress: Optional[Callable[[int, int, str], None]] = None,
                ) -> ELAResult:
    """Compute the requested ELA feature groups for one PORTAL instance.

    Args:
        instance: a unified runtime instance, or a transient Studio composition
            that is converted to the unified schema before evaluation. The same
            canonical evaluator is used here as in ``run_ela.py``.
        groups: subset of :data:`FEATURE_GROUPS` to compute. ``None`` means
            all groups.
        sample_factor: number of LHS samples is ``sample_factor * d`` (the
            research pipeline uses 1000).
        seed: LHS and stochastic feature RNG seed.

    Returns:
        :class:`ELAResult` with the merged feature dict and a list of any
        groups that raised during computation.
    """
    from pflacco.sampling import create_initial_sample
    import pflacco.classical_ela_features as clf
    import pflacco.misc_features as mf

    validate_instance_for_gui(instance)
    # The editor stages a new Layer-3 design as a transient composition. Before
    # ELA, convert it to the same unified runtime used by saved JSON files and
    # by the script-level ELA pipeline.
    if bool(instance.get("composite", False)):
        from portal_core.save_instance import composite_to_unified_payload
        from portal_core.load_instance import load_instance_payload
        instance = load_instance_payload(composite_to_unified_payload(instance))

    d = int(instance["d"])
    lb = float(instance["bounds_min"])
    ub = float(instance["bounds_max"])
    n_samples = int(sample_factor) * d

    X = create_initial_sample(d, n_samples, lower_bound=lb, upper_bound=ub, seed=seed)
    Xnp = X.to_numpy(dtype=float)
    y = _evaluate_batch(Xnp, instance)
    f = lambda x: _evaluate_one(np.asarray(x, dtype=float), instance)

    # All ten groups; the caller selects which to run.
    group_fns: Dict[str, Callable[[], Dict[str, float]]] = {
        "ela_meta":             lambda: clf.calculate_ela_meta(X, y),
        "ela_distribution":     lambda: clf.calculate_ela_distribution(X, y),
        "ela_level":            lambda: clf.calculate_ela_level(X, y),
        "pca":                  lambda: clf.calculate_pca(X, y),
        "nbc":                  lambda: clf.calculate_nbc(X, y),
        "dispersion":           lambda: clf.calculate_dispersion(X, y),
        "information_content":  lambda: clf.calculate_information_content(X, y, seed=seed),
        "fitness_distance":     lambda: mf.calculate_fitness_distance_correlation(X, y),
        "ela_conv":             lambda: clf.calculate_ela_conv(X, y, f, seed=seed),
        "gradient":             lambda: mf.calculate_gradient_features(f, d, lb, ub, seed=seed),
    }

    chosen = list(groups) if groups is not None else list(FEATURE_GROUPS)
    unknown = [g for g in chosen if g not in group_fns]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}; "
                         f"valid are {list(group_fns)}")

    features: Dict[str, float] = {}
    failed: List[str] = []
    for idx, g in enumerate(chosen, start=1):
        _raise_if_stopped(should_stop)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features.update(group_fns[g]())
        except Exception:
            # Per the research-pipeline convention: don't abort the whole
            # recipe — record the group as failed and move on.
            failed.append(g)
        if progress is not None:
            progress(idx, len(chosen), g)
        _raise_if_stopped(should_stop)

    # Match run_ela.py: prepend "ela_" to feature names that don't already
    # carry it, so output columns are uniformly namespaced.
    feats = {(k if k.startswith("ela_") else f"ela_{k}"): v
             for k, v in features.items()}

    J = int(len(instance.get("_schema_blocks", [])) or 1)
    return ELAResult(label="", features=feats, failed_groups=failed, d=d, J=J)


def compute_ela_batch(items: Sequence[Tuple[str, Dict[str, Any]]],
                      groups: Optional[Sequence[str]] = None,
                      *,
                      sample_factor: int = 1000,
                      seed: int = 42,
                      progress: Optional[Callable[[int, int, str], None]] = None,
                      should_stop: Optional[Callable[[], bool]] = None,
                      ) -> "pandas.DataFrame":
    """Compute ELA features across a list of ``(label, composite)`` pairs.

    Returns a wide DataFrame with one row per composite. Identity columns
    (``label``, ``d``, ``J``, ``failed_groups``) sit before the feature
    columns. Features absent from a row (because their group failed or was
    not selected) appear as NaN.
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    total = len(items)
    chosen_groups = list(groups) if groups is not None else list(FEATURE_GROUPS)
    total_steps = max(1, total * max(1, len(chosen_groups)))
    completed_steps = 0
    for i, (label, comp) in enumerate(items, start=1):
        _raise_if_stopped(should_stop)
        def group_progress(_idx: int, _n: int, group: str,
                           item_label: str = label) -> None:
            nonlocal completed_steps
            completed_steps += 1
            if progress is not None:
                progress(completed_steps, total_steps, f"{item_label} / {group}")
        res = compute_ela(comp, groups=groups,
                          sample_factor=sample_factor, seed=seed,
                          should_stop=should_stop,
                          progress=group_progress)
        row: Dict[str, Any] = {
            "label": label,
            "d": res.d,
            "J": res.J,
            "failed_groups": ",".join(res.failed_groups) if res.failed_groups else "",
        }
        row.update(res.features)
        rows.append(row)
    return pd.DataFrame(rows)


def save_ela_xlsx(df: "pandas.DataFrame", path: str | Path) -> str:
    """Write the ELA results DataFrame to an ``.xlsx`` file."""
    path = str(path)
    if not path.lower().endswith(".xlsx"):
        path = path + ".xlsx"
    df.to_excel(path, index=False)
    return path
