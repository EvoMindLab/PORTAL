"""Statistical-analysis methods for the SA sub-tab.

The GUI exposes the following comparisons:

- Friedman omnibus over matched final-error samples, with Holm-corrected
  pairwise Wilcoxon signed-rank post-hoc rows when the omnibus is significant.
- Wilcoxon signed-rank for paired final-error samples in the selected measure.

Pairwise rows report both raw and Holm-adjusted p-values. Significance is
decided from the adjusted p-value at alpha=0.05.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats
from portal_studio.experiment_runner.normalisation import read_normalised, factor


SIGNIFICANCE_ALPHA = 0.05


@dataclass
class SelectionItem:
    """One entry in the user's SA selection."""
    bundle_path: str
    runset_uid: str
    label: str
    algorithm: str
    task_label: str


@dataclass
class SaTable:
    """One result table rendered by the GUI."""
    caption: str
    columns: list[str]
    rows: list[list[str]]
    significant: list[list[bool]] = field(default_factory=list)


@dataclass
class SaResult:
    """Top-level SA payload, one set per Run Statistical Test invocation."""
    method: str
    metric: str
    tables: list[SaTable]
    message: str = ""


def _load_final_errors(item: SelectionItem) -> np.ndarray:
    """Per-run final_error array for one selected runset."""
    cx = sqlite3.connect(Path(item.bundle_path) / "summary.sqlite",
                         timeout=5.0)
    try:
        cur = cx.execute(
            """SELECT final_error FROM runs
               WHERE runset_uid = ? ORDER BY run_index""",
            (item.runset_uid,),
        )
        return np.asarray([row[0] for row in cur], dtype=float)
    finally:
        cx.close()


def validated_run_data(item: SelectionItem, metric: str = 'final_error', *, minimum_runs: int = 2):
    """Read complete valid trials and their identities, for grouping or testing."""
    factor(metric)
    path = (Path(item.bundle_path) / 'summary.sqlite').resolve()
    if not path.is_file():
        raise ValueError(f"Result database is missing for {item.label}.")
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5.0)) as cx:
        columns = {row[1] for row in cx.execute('PRAGMA table_info(runs)')}
        if 'pairing_key' not in columns:
            raise ValueError('These older results have no verified problem identities. Rerun the experiment.')
        runset = cx.execute(
            'SELECT runs_expected, runs_done, status FROM runsets WHERE runset_uid = ?',
            (item.runset_uid,),
        ).fetchone()
        rows = cx.execute(
            'SELECT run_index, final_error, pairing_key FROM runs '
            'WHERE runset_uid = ? ORDER BY run_index', (item.runset_uid,),
        ).fetchall()
    if (runset is None or runset[2] != 'finished' or len(rows) != runset[0]
            or len(rows) != runset[1] or len(rows) < minimum_runs):
        raise ValueError(
            f"{item.label} needs a complete successful runset with at least {minimum_runs} run(s).")
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{item.label} has missing or duplicate run identifiers.")
    values = np.asarray([row[1] for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{item.label} contains invalid final errors.")
    if any(not isinstance(row[2], str) or not row[2] for row in rows):
        raise ValueError(f"{item.label} lacks verified problem identities. Rerun this experiment.")
    keys = tuple((row[0], row[2]) for row in rows)
    if metric != 'final_error':
        values = read_normalised(item.bundle_path, item.runset_uid, metric)
    return values, keys


def _gather_matrix(selection: Sequence[SelectionItem], metric: str = 'final_error'
                   ) -> tuple[np.ndarray, list[SelectionItem]]:
    """Validate again at test time, then align by verified problem identity."""
    factor(metric)
    if not selection:
        return np.zeros((0, 0)), []
    arrays = []
    reference_keys = None
    for item in selection:
        values, keys = validated_run_data(item, metric)
        if reference_keys is not None and keys != reference_keys:
            raise ValueError(
                'Selected tasks evaluated different problem realisations or trial settings. '
                'Use matching source problems, realisation seeds, sample identifiers and budgets.')
        reference_keys = keys
        arrays.append(values)
    return np.vstack(arrays), list(selection)


def run_friedman_nemenyi(selection: Sequence[SelectionItem],
                         metric: str = "final_error") -> SaResult:
    """Omnibus Friedman + Holm-corrected pairwise Wilcoxon post-hoc."""
    matrix, items = _gather_matrix(selection, metric)
    if len(items) < 3:
        return SaResult(
            method="friedman", metric=metric, tables=[],
            message="Friedman needs at least 3 algorithms in the selection.",
        )

    stat, p_omnibus = stats.friedmanchisquare(*matrix)
    tables: list[SaTable] = [SaTable(
        caption=f"Friedman omnibus ({metric})",
        columns=["Statistic", "p-value", "Significant @ alpha=0.05"],
        rows=[[
            _fmt(stat), _fmt_p(p_omnibus),
            "yes" if p_omnibus < SIGNIFICANCE_ALPHA else "",
        ]],
        significant=[[False, p_omnibus < SIGNIFICANCE_ALPHA, False]],
    )]

    if p_omnibus < SIGNIFICANCE_ALPHA:
        tables.append(_pairwise_signed_rank_table(matrix, items, metric))
    else:
        tables.append(SaTable(
            caption=("Pairwise Wilcoxon post-hoc skipped because the "
                     "omnibus test was not significant."),
            columns=[],
            rows=[],
            significant=[],
        ))
    return SaResult(method="friedman", metric=metric, tables=tables)


def run_wilcoxon_signed_rank(selection: Sequence[SelectionItem],
                             metric: str = "final_error") -> SaResult:
    matrix, items = _gather_matrix(selection, metric)
    if len(items) < 2:
        return SaResult(
            method="signed-rank", metric=metric, tables=[],
            message="Need at least 2 algorithms for a pairwise test.",
        )
    table = _pairwise_signed_rank_table(matrix, items, metric)
    return SaResult(method="signed-rank", metric=metric, tables=[table])


def _pairwise_signed_rank_table(matrix: np.ndarray,
                                items: Sequence[SelectionItem],
                                metric: str) -> SaTable:
    pair_records: list[tuple[str, str, float, float]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if np.array_equal(matrix[i], matrix[j]):
                # With no nonzero differences, the default signed-rank test
                # is undefined. Do not turn an unavailable p-value into zero.
                stat, p = np.nan, np.nan
            else:
                stat, p = stats.wilcoxon(matrix[i], matrix[j])
            pair_records.append((items[i].label, items[j].label,
                                 float(stat), float(p)))
    return _pairwise_table(
        pair_records,
        statistic_column="W",
        caption=f"Pairwise Wilcoxon signed-rank on {metric}",
    )


def _pairwise_table(pair_records: Sequence[tuple[str, str, float, float]],
                    *, statistic_column: str, caption: str) -> SaTable:
    adjusted = _holm_adjust([record[3] for record in pair_records])
    rows: list[list[str]] = []
    sig: list[list[bool]] = []
    for (label_a, label_b, stat, p_raw), p_holm in zip(
            pair_records, adjusted, strict=True):
        is_sig = bool(p_holm < SIGNIFICANCE_ALPHA)
        rows.append([
            label_a, label_b,
            _fmt(stat), _fmt_p(p_raw), _fmt_p(p_holm),
            "yes" if is_sig else "",
        ])
        sig.append([False, False, False, False, is_sig, False])
    return SaTable(
        caption=(f"{caption} - {len(pair_records)} comparison(s), "
                 "Holm-corrected; undefined comparisons are n/a"),
        columns=["A", "B", statistic_column, "p-value", "Adjusted p", "Sig"],
        rows=rows,
        significant=sig,
    )


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if np.any(np.isinf(values) | (values < 0) | (values > 1)):
        raise ValueError('Each p-value must be in [0, 1] or unavailable (NaN).')
    order = np.flatnonzero(np.isfinite(values))
    order = order[np.argsort(values[order])]
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        # Retain the full comparison family, including unavailable tests.
        candidate = (m - rank) * float(values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.4g}"


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4g}"


def run_sa(selection: Sequence[SelectionItem], *, method: str,
           metric: str = "final_error") -> SaResult:
    """Dispatcher used by the GUI."""
    if method == "friedman":
        return run_friedman_nemenyi(selection, metric)
    if method == "signed-rank":
        return run_wilcoxon_signed_rank(selection, metric)
    raise ValueError(f"Unknown SA method: {method!r}")
