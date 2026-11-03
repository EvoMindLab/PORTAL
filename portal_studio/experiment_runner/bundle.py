"""Read-side adapter over result-store bundles.

A *bundle* is a directory that contains ``summary.sqlite`` and (usually)
``trend.sqlite`` — the on-disk shape :mod:`portal_studio.experiment_runner.store`
writes. The Results sub-tab opens these by path and inspects their
contents without going through the live :class:`BatchScheduler`.

Two value types:

- :class:`BundleSummary` — one row per bundle (the user's bundle list)
- :class:`RunsetRow` — one row per runset inside a bundle (the workspace
  table rows)

Both are plain dataclasses; the Qt layer wraps them in
``QAbstractTableModel`` adapters but doesn't subclass them.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from portal_studio.experiment_runner.store import (
    SUMMARY_DB_NAME, TREND_DB_NAME, ResultStore,
)


@dataclass
class BundleSummary:
    """Top-level metadata for one results directory."""
    bundle_id: str          # short, generated when the bundle is loaded
    source_path: str        # absolute path to the bundle dir
    runset_count: int
    runs_total_expected: int
    runs_total_done: int
    algorithms: list[str]   # distinct algorithm names across all runsets
    fe_budget: int
    composite_path: str | None
    # The Studio populates these via :func:`import_runsets`. They start at 0
    # and grow as the user pulls runsets into the shared imported_pool.
    imported_runset_count: int = 0


@dataclass
class RunsetRow:
    """One runset inside a bundle — what the workspace table renders."""
    bundle_id: str
    bundle_path: str
    runset_uid: str
    task_label: str
    algorithm: str
    runs_expected: int
    runs_done: int
    status: str
    algorithm_params: dict = field(default_factory=dict)
    # Filled by the Studio when this row is pulled into the imported pool.
    imported: bool = False
    imported_at: str | None = None
    # PORTAL only has one problem class (the composite itself), so we tag
    # every row with the same label — the filter strip still works.
    problem: str = "PORTAL"


def load_bundle(path: str | Path) -> BundleSummary:
    """Open a results directory and return its :class:`BundleSummary`.

    Validates that ``summary.sqlite`` exists and that the schema is
    readable. ``trend.sqlite`` may be missing (older / minimal exports);
    plot-side code paths handle that downstream.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")
    summary_path = p / SUMMARY_DB_NAME
    if not summary_path.is_file():
        raise ValueError(
            f"Not a results bundle (missing {SUMMARY_DB_NAME}): {p}")

    store = ResultStore(p)
    try:
        runsets = store.list_runsets()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Could not read {summary_path}: {exc}") from exc

    algorithms = sorted({r["algorithm"] for r in runsets})
    runs_expected = sum(int(r["runs_expected"]) for r in runsets)
    runs_done = sum(int(r["runs_done"]) for r in runsets)

    meta = _read_meta(summary_path)
    return BundleSummary(
        bundle_id=uuid.uuid4().hex[:10],
        source_path=str(p),
        runset_count=len(runsets),
        runs_total_expected=runs_expected,
        runs_total_done=runs_done,
        algorithms=algorithms,
        fe_budget=int(meta.get("fe_budget", 0) or 0),
        composite_path=meta.get("composite_path"),
    )


def list_bundle_runsets(bundle: BundleSummary) -> list[RunsetRow]:
    """Every runset row inside one bundle (un-imported by default)."""
    store = ResultStore(bundle.source_path)
    out: list[RunsetRow] = []
    for raw in store.list_runsets():
        out.append(RunsetRow(
            bundle_id=bundle.bundle_id,
            bundle_path=bundle.source_path,
            runset_uid=str(raw["runset_uid"]),
            task_label=str(raw["task_label"]),
            algorithm=str(raw["algorithm"]),
            runs_expected=int(raw["runs_expected"]),
            runs_done=int(raw["runs_done"]),
            status=str(raw["status"]),
            algorithm_params=dict(raw.get("algorithm_params") or {}),
        ))
    return out


def _read_meta(summary_path: Path) -> dict:
    """Pull the ``meta`` table as a plain dict.

    Reads the metadata table directly using the :class:`ResultStore` schema.
    """
    out: dict = {}
    cx = sqlite3.connect(summary_path, timeout=5.0)
    try:
        cur = cx.execute("SELECT key, value FROM meta")
        for k, v in cur:
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = v
    except sqlite3.DatabaseError:
        # Older bundles may have no meta table at all; treat as empty.
        pass
    finally:
        cx.close()
    return out
