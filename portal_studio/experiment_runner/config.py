"""Batch + task configuration dataclasses, with JSONL load/save.

- one unified PORTAL instance per batch (no ``default_problem`` field; the
  instance carries its own dimensionality and block structure),
- a single per-batch seed strategy (``fixed`` int or ``random`` per run),
- multi-algorithm fan-out preserved (``algorithm: [X, Y, Z]`` shorthand).

JSONL shape (system-config line optional, ``#``-prefixed comments and blank
lines ignored)::

    # optional comment
    {"max_parallel_runs": 6, "output_dir": "results/raw_runs/exp1",
     "default_runs": 31, "fe_budget": 100000, "seed_strategy": "random",
     "master_seed": 42, "metric_outputs": ["best_so_far", "final_error"],
     "composite_path": "PORTAL_instance.json"}
    {"label": "pso_default", "algorithm": "PSO"}
    {"label": "compare_three", "algorithm": ["PSO", "DE", "CMA-ES"],
     "algorithm_params": {"pop_size": 30}}
    {"label": "tuned_de",  "algorithm": "DE",
     "per_algo_params": {"DE": {"F": 0.7, "CR": 0.9}}}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SYSTEM_KEYS = {
    "max_parallel_runs",
    "output_dir",
    "default_runs",
    "fe_budget",
    "seed_strategy",
    "master_seed",
    "metric_outputs",
    "composite_path",
    "default_algorithm_params",
    "default_per_algo_params",
}

_TASK_KEYS = {
    "label",
    "algorithm",
    "runs",
    "seed_strategy",
    "master_seed",
    "realization_type",
    "algorithm_params",
    "per_algo_params",
}


@dataclass
class TaskConfig:
    """One row in the batch's task list.

    ``algorithm`` is either a single name or a list — multi-algorithm
    tasks fan out at scheduler time into one run-unit per (algorithm,
    run_index) pair, all tagged with this task's label.
    """
    label: str
    algorithm: str | list[str]
    algorithm_params: dict[str, Any] = field(default_factory=dict)
    per_algo_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: int = 1
    seed_strategy: str = "random"      # 'fixed' | 'random'
    master_seed: int = 42
    realization_type: str = "none"     # 'none' | 'type1' | 'type2'

    def algorithm_list(self) -> list[str]:
        """Normalise ``algorithm`` to a list of names."""
        if isinstance(self.algorithm, str):
            return [self.algorithm]
        return list(self.algorithm)


@dataclass
class BatchConfig:
    """Top-level batch description.

    ``composite_path`` points to a unified PORTAL instance JSON. A value of
    ``None`` means "use the live Generator instance"; the GUI resolves this at
    submit time. JSONL on-disk batches must give an explicit path.
    """
    tasks: list[TaskConfig]
    fe_budget: int = 100_000
    metric_outputs: list[str] = field(
        default_factory=lambda: ["best_so_far", "final_error"])
    max_parallel_runs: int = field(
        default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    output_dir: str | None = None
    composite_path: str | None = None


# ───────────────────────── JSONL loader ─────────────────────────────

def load_batch_jsonl(path: str | Path) -> BatchConfig:
    """Parse a JSONL batch file into a :class:`BatchConfig`.

    The first non-comment line is treated as system config if it carries
    only keys from :data:`_SYSTEM_KEYS`; otherwise it's parsed as a task.
    Comments (``#``-prefixed) and blank lines are skipped. Duplicate task
    labels raise ``ValueError``.
    """
    records = _read_json_lines(Path(path))
    if not records:
        raise ValueError("Batch file must contain at least one task record.")

    system: dict[str, Any] = {}
    task_records = records
    if _looks_like_system_record(records[0][1]):
        _line, system = records[0]
        _validate_keys(system, _SYSTEM_KEYS, "system config")
        task_records = records[1:]
    if not task_records:
        raise ValueError("Batch file must contain at least one task record.")

    default_runs = int(system.get("default_runs", 1))
    default_seed_strategy = str(system.get("seed_strategy", "random"))
    default_master_seed = int(system.get("master_seed", 42))
    default_algo_params = dict(system.get("default_algorithm_params", {}))
    default_per_algo = {
        k: dict(v) for k, v in
        system.get("default_per_algo_params", {}).items()
    }

    tasks: list[TaskConfig] = []
    seen: set[str] = set()
    for line_no, record in task_records:
        _validate_keys(record, _TASK_KEYS, f"task at line {line_no}")
        if "algorithm" not in record:
            raise ValueError(
                f"Task at line {line_no} is missing required 'algorithm' key.")
        label = str(record.get("label") or f"task_{len(tasks) + 1}")
        if label in seen:
            raise ValueError(
                f"Duplicate task label {label!r} (line {line_no}).")
        seen.add(label)

        algo = record["algorithm"]
        if isinstance(algo, list):
            algo = [str(a) for a in algo]
            if not algo:
                raise ValueError(
                    f"Task {label!r}: 'algorithm' list must be non-empty.")
        else:
            algo = str(algo)

        algo_params = dict(default_algo_params)
        algo_params.update(record.get("algorithm_params", {}))
        per_algo = {k: dict(v) for k, v in default_per_algo.items()}
        for k, v in record.get("per_algo_params", {}).items():
            per_algo.setdefault(k, {}).update(v)

        tasks.append(TaskConfig(
            label=label,
            algorithm=algo,
            algorithm_params=algo_params,
            per_algo_params=per_algo,
            runs=int(record.get("runs", default_runs)),
            seed_strategy=str(record.get("seed_strategy", default_seed_strategy)),
            master_seed=int(record.get("master_seed", default_master_seed)),
            realization_type=str(record.get("realization_type", "none")),
        ))

    return BatchConfig(
        tasks=tasks,
        fe_budget=int(system.get("fe_budget", 100_000)),
        metric_outputs=list(system.get(
            "metric_outputs", ["best_so_far", "final_error"])),
        max_parallel_runs=int(system.get(
            "max_parallel_runs", max(1, (os.cpu_count() or 2) - 1))),
        output_dir=system.get("output_dir"),
        composite_path=system.get("composite_path"),
    )


def save_batch_jsonl(cfg: BatchConfig, path: str | Path) -> Path:
    """Write a :class:`BatchConfig` back out as JSONL.

    Round-trips with :func:`load_batch_jsonl` modulo formatting. Empty
    fields (default values) are elided so power users can hand-edit the
    file without noise.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system: dict[str, Any] = {
        "max_parallel_runs": cfg.max_parallel_runs,
        "fe_budget": cfg.fe_budget,
        "metric_outputs": cfg.metric_outputs,
    }
    if cfg.output_dir:
        system["output_dir"] = cfg.output_dir
    if cfg.composite_path:
        system["composite_path"] = cfg.composite_path

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(system) + "\n")
        for t in cfg.tasks:
            record: dict[str, Any] = {
                "label": t.label,
                "algorithm": t.algorithm,
            }
            if t.runs != 1:
                record["runs"] = t.runs
            if t.seed_strategy != "random":
                record["seed_strategy"] = t.seed_strategy
            if t.master_seed != 42:
                record["master_seed"] = t.master_seed
            if t.realization_type != "none":
                record["realization_type"] = t.realization_type
            if t.algorithm_params:
                record["algorithm_params"] = t.algorithm_params
            if t.per_algo_params:
                record["per_algo_params"] = t.per_algo_params
            fh.write(json.dumps(record) + "\n")
    return out_path


# ──────────────────────────── internals ─────────────────────────────

def _read_json_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read JSONL with comment + blank-line skipping. Returns (line_no, dict)."""
    out: list[tuple[int, dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} — invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_no} — each JSONL record must be an object.")
            out.append((line_no, value))
    return out


def _looks_like_system_record(record: dict[str, Any]) -> bool:
    """Heuristic: a system-config line has no ``label`` and no ``algorithm``."""
    return "label" not in record and "algorithm" not in record


def _validate_keys(record: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {sorted(unknown)!r}. "
            f"Allowed: {sorted(allowed)!r}.")
