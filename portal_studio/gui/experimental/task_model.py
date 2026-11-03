"""Per-task state shared across the Experiment tab panels.

Each authored task carries its own algorithm, parameters, composite source,
FE budget, runs, and seed strategy. Tasks have a life-cycle:

    pending → running → (completed | failed | canceled)

The :class:`TaskRegistry` is a Qt-aware container kept on the
:class:`ExperimentationTab`. The four column panels subscribe to its
signals — they never touch a task list directly. Each column edits the
slice of state it owns (Task Settings creates pendings; Pending Tasks
flips status to running; Finished Tasks owns metrics; SA owns the
selected-for-SA flag).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal


def _short_id() -> str:
    """Short hexadecimal task identifier."""
    return secrets.token_hex(2)


@dataclass
class Task:
    """One authored task. Lives across all four columns of the tab."""
    label: str
    algorithm: str
    algorithm_params: dict[str, Any] = field(default_factory=dict)
    runs: int = 31
    fe_budget: int = 100_000
    seed_strategy: str = "random"      # 'fixed' | 'random'
    master_seed: int = 42
    realization_type: str = "none"     # 'none' | 'type1' | 'type2'
    composite_label: str = "Generator instance"
    composite_source_kind: str = "generator"
    composite_path: str | None = None
    metric_outputs: list[str] = field(
        default_factory=lambda: ["best_so_far", "final_error"])
    # Runtime state.
    task_id: str = field(default_factory=_short_id)
    status: str = "pending"
    # Where the scheduler stored this task's runset (set on submit).
    output_dir: str | None = None
    runset_uid: str | None = None
    runs_done: int = 0
    runs_failed: int = 0
    message: str = ""
    # Scalar metrics summarised from summary.sqlite::runs once the task
    # finishes - used by the Finished Tasks table.
    metrics: dict[str, float] = field(default_factory=dict)
    # User flag — drives the SA column.
    in_sa: bool = False

    def info(self) -> str:
        """Short single-line summary, shown in the Task Info column."""
        comp = self.composite_label.replace("Generator composite", "Gen")
        comp = comp.replace("Generator instance", "Gen")
        return f"{self.algorithm}_{comp}"

    def progress_label(self) -> str:
        if self.status == "pending":
            return "Not Started"
        if self.status == "completed":
            return f"{self.runs_done}/{self.runs}"
        if self.status in ("failed", "canceled"):
            return self.message or self.status
        return f"{self.runs_done}/{self.runs}"


class TaskRegistry(QObject):
    """Source of truth for every authored task in the current session.

    Signals are coarse-grained: any column that cares re-renders from
    scratch. This avoids per-task diffing and keeps the four panels
    decoupled.
    """

    pending_changed = Signal()
    completed_changed = Signal()
    sa_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Insertion-ordered, keyed by task_id.
        self._by_id: dict[str, Task] = {}

    # ───────────────────────── reads ───────────────────────────────

    def all_tasks(self) -> list[Task]:
        return list(self._by_id.values())

    def by_id(self, task_id: str) -> Task | None:
        return self._by_id.get(task_id)

    def pending(self) -> list[Task]:
        return [t for t in self._by_id.values() if t.status == "pending"]

    def running(self) -> list[Task]:
        return [t for t in self._by_id.values() if t.status == "running"]

    def completed(self) -> list[Task]:
        # 'completed' or any terminal state (failed/canceled) so the user
        # can still see what happened and decide whether to retry.
        return [t for t in self._by_id.values()
                if t.status in ("completed", "failed", "canceled")]

    def in_sa(self) -> list[Task]:
        return [t for t in self._by_id.values() if t.in_sa]

    # ───────────────────────── writes ──────────────────────────────

    def add(self, task: Task) -> None:
        self._by_id[task.task_id] = task
        if task.status in ("completed", "failed", "canceled"):
            self.completed_changed.emit()
        else:
            self.pending_changed.emit()

    def remove(self, task_id: str) -> None:
        task = self._by_id.pop(task_id, None)
        if task is None:
            return
        # Touch all three buckets — caller doesn't have to know which one
        # this id was in.
        self.pending_changed.emit()
        self.completed_changed.emit()
        self.sa_changed.emit()

    def set_status(self, task_id: str, status: str, *,
                   message: str = "") -> None:
        task = self._by_id.get(task_id)
        if task is None:
            return
        prior = task.status
        task.status = status
        if message:
            task.message = message
        # Emit on the buckets the row entered/left. Pending panel
        # shows pending AND running rows, so transitions between those
        # two (and out of either) all need to refresh it.
        if prior in ("pending", "running") or status in ("pending", "running"):
            self.pending_changed.emit()
        if (prior in ("completed", "failed", "canceled")
                or status in ("completed", "failed", "canceled")):
            self.completed_changed.emit()

    def update_progress(self, task_id: str, runs_done: int, *,
                        runs_failed: int | None = None,
                        message: str | None = None) -> None:
        task = self._by_id.get(task_id)
        if task is None:
            return
        task.runs_done = runs_done
        if runs_failed is not None:
            task.runs_failed = runs_failed
        if message is not None:
            task.message = message
        self.pending_changed.emit()

    def set_metrics(self, task_id: str, metrics: dict[str, float]) -> None:
        task = self._by_id.get(task_id)
        if task is None:
            return
        task.metrics = dict(metrics)
        self.completed_changed.emit()

    def set_in_sa(self, task_id: str, in_sa: bool) -> None:
        task = self._by_id.get(task_id)
        if task is None or task.in_sa == in_sa:
            return
        task.in_sa = in_sa
        self.sa_changed.emit()

    def clear_sa(self) -> None:
        changed = False
        for task in self._by_id.values():
            if task.in_sa:
                task.in_sa = False
                changed = True
        if changed:
            self.sa_changed.emit()
