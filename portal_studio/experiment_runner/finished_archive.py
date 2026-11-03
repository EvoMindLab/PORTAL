"""Portable finished-task results, including the databases consumed by SA.

JSONL remains a batch-configuration format. These archives restore recorded
results without importing an optimiser, reopening a problem, or running it.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import tempfile
import uuid
import zipfile
from contextlib import closing
from pathlib import Path


FORMAT = "portal-finished-results"
VERSION = 1
TERMINAL = {"completed", "failed", "canceled"}
TASK_FIELDS = {
    "label", "algorithm", "algorithm_params", "runs", "fe_budget",
    "seed_strategy", "master_seed", "realization_type", "composite_label",
    "composite_source_kind", "composite_path", "metric_outputs", "status",
    "runset_uid", "runs_done", "runs_failed", "message",
}


def _read_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _tables(cx):
    return {r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _repair_trends(bundle, tasks):
    """Rebuild copied aggregates; original files and individual runs stay intact."""
    from portal_studio.experiment_runner.store import ResultStore
    if not (bundle / 'trend.sqlite').is_file():
        return
    with closing(sqlite3.connect(bundle / 'summary.sqlite')) as cx:
        cx.execute('ATTACH DATABASE ? AS trend_db', (str(bundle / 'trend.sqlite'),))
        with cx:
            for task in tasks:
                ResultStore._rebuild_trace(cx, task['runset_uid'], budget=task['fe_budget'])


def _snapshot(source, destination, uids, *, trend=False):
    """Take a consistent backup, retaining only selected runsets."""
    with closing(_read_db(source)) as src, closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)
        marks = ",".join("?" for _ in uids)
        tables = _tables(dst)
        if trend:
            dst.execute(f"DELETE FROM trends WHERE runset_uid NOT IN ({marks})", uids)
        else:
            from portal_studio.experiment_runner.store import migrate_performance_checkpoints
            migrate_performance_checkpoints(dst)
            tables = _tables(dst)
            for table in ("performance_checkpoints", "runs", "runsets"):
                if table in tables:
                    dst.execute(f"DELETE FROM {table} WHERE runset_uid NOT IN ({marks})", uids)
            for table in ("raw_traces", "normalised_traces", "raw_trace_runs"):
                if table in tables:
                    dst.execute(f"DELETE FROM {table} WHERE run_uid NOT IN (SELECT run_uid FROM runs)")
            if "meta" in tables:
                row = dst.execute("SELECT value FROM meta WHERE key='run_failures'").fetchone()
                if row:
                    failures = [f for f in json.loads(row[0]) if f.get("runset_uid") in uids]
                    dst.execute("UPDATE meta SET value=? WHERE key='run_failures'", (json.dumps(failures),))
        dst.commit()


def _validate_task(task, bundle):
    if not isinstance(task, dict) or not set(task).issubset(TASK_FIELDS):
        raise ValueError("Invalid finished-task settings.")
    required = {"label", "algorithm", "runs", "fe_budget", "status", "runs_done", "runs_failed"}
    if not required.issubset(task) or task["status"] not in TERMINAL:
        raise ValueError("Only finished tasks can be archived.")
    for key in ("runs", "fe_budget", "runs_done", "runs_failed"):
        if type(task[key]) is not int or task[key] < (1 if key in ("runs", "fe_budget") else 0):
            raise ValueError(f"Invalid task {key}.")
    for key in ("label", "algorithm"):
        if not isinstance(task[key], str) or not task[key]:
            raise ValueError(f"Invalid task {key}.")
    if bundle is None:
        if task["status"] == "completed" or task["runs_done"]:
            raise ValueError("A task with successful runs is missing its result database.")
        return
    with closing(_read_db(bundle / "summary.sqlite")) as cx:
        if cx.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Invalid result database.")
        row = cx.execute(
            "SELECT algorithm, runs_expected, runs_done, status FROM runsets WHERE runset_uid=?",
            (task.get("runset_uid"),),
        ).fetchone()
        values = cx.execute(
            "SELECT run_index, final_error FROM runs WHERE runset_uid=? ORDER BY run_index",
            (task.get("runset_uid"),),
        ).fetchall()
        if row is None or row[0] != task["algorithm"] or row[1] != task["runs"]:
            raise ValueError("Task settings do not match their result database.")
        if len(values) != task["runs_done"] or row[2] != len(values):
            raise ValueError("Task run counts do not match their recorded results.")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for _, v in values):
            raise ValueError("Recorded run errors must be finite and nonnegative.")
        if task["status"] == "completed":
            if (row[3] != "finished" or task["runs_failed"] or len(values) != task["runs"]
                    or [i for i, _ in values] != list(range(task["runs"]))):
                raise ValueError("A completed task requires a complete successful runset.")


def _write_csvs(root, records):
    """Human-readable tables; SQLite remains the authoritative result data."""
    with (root / "run_errors.csv").open("w", newline="", encoding="utf-8-sig") as errors, \
            (root / "convergence.csv").open("w", newline="", encoding="utf-8-sig") as traces:
        ew, tw = csv.writer(errors), csv.writer(traces)
        ew.writerow(["bundle", "task", "algorithm", "problem_source", "fe_budget", "realization_type",
                     "run_index", "seed", "fe_used", "final_error",
                     "reference_scale", "normalised_error", "pairing_key"])
        tw.writerow(["bundle", "task", "algorithm", "run_index", "fe", "raw_error", "normalised_error"])
        for record in records:
            if record["bundle"] is None:
                continue
            task = record["task"]
            with closing(_read_db(root / record["bundle"] / "summary.sqlite")) as cx:
                cx.row_factory = sqlite3.Row
                rows = cx.execute("SELECT * FROM runs WHERE runset_uid=? ORDER BY run_index",
                                  (task["runset_uid"],)).fetchall()
                tables = _tables(cx)
                for row in rows:
                    data = dict(row)
                    scale = data.get("reference_scale")
                    normalised = row["final_error"] / scale if scale and scale > 0 else None
                    ew.writerow([record["bundle"], task["label"], task["algorithm"], task.get("composite_label"),
                                 task["fe_budget"], task.get("realization_type", "none"), row["run_index"], row["seed"],
                                 row["fe_used"], row["final_error"], scale, normalised,
                                 data.get("pairing_key")])
                    norm = dict(cx.execute("SELECT fe, value FROM normalised_traces WHERE run_uid=?",
                                           (row["run_uid"],))) if "normalised_traces" in tables else {}
                    if "raw_traces" in tables:
                        for fe, value in cx.execute("SELECT fe, value FROM raw_traces WHERE run_uid=? ORDER BY fe",
                                                    (row["run_uid"],)):
                            tw.writerow([record["bundle"], task["label"], task["algorithm"], row["run_index"], fe,
                                         value, norm.get(fe)])


def export_finished_tasks(tasks: list[dict], path: str | Path) -> Path:
    """Write selected terminal tasks and independent copies of their results."""
    if not tasks:
        raise ValueError("Select at least one finished task.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portal-results-") as tmp:
        root = Path(tmp)
        records, groups = [], {}
        for task in tasks:
            settings = {key: value for key, value in task.items() if key in TASK_FIELDS}
            source = Path(task["output_dir"]).resolve() if task.get("output_dir") else None
            group = None
            if source is not None and (source / "summary.sqlite").is_file() and task.get("runset_uid"):
                if source not in groups:
                    groups[source] = (f"bundles/b{len(groups):04d}", [])
                group, uids = groups[source]
                if task["runset_uid"] not in uids:
                    uids.append(task["runset_uid"])
            records.append({"task": settings, "bundle": group})
        for source, (name, uids) in groups.items():
            dest = root / name
            dest.mkdir(parents=True)
            _snapshot(source / "summary.sqlite", dest / "summary.sqlite", uids)
            if (source / "trend.sqlite").is_file():
                _snapshot(source / "trend.sqlite", dest / "trend.sqlite", uids, trend=True)
            _repair_trends(dest, [r['task'] for r in records if r['bundle'] == name])
        for record in records:
            _validate_task(record["task"], root / record["bundle"] if record["bundle"] else None)
        (root / "manifest.json").write_text(json.dumps(
            {"format": FORMAT, "version": VERSION, "tasks": records}, indent=2, allow_nan=False), encoding="utf-8")
        _write_csvs(root, records)
        # Replace the destination only after a complete archive has been written.
        fd, staging = tempfile.mkstemp(prefix=".portal-results-", suffix=".zip", dir=path.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(root.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(root).as_posix())
            os.replace(staging, path)
        finally:
            if os.path.exists(staging):
                os.unlink(staging)
    return path


def import_finished_tasks(path: str | Path, destination: str | Path) -> list[dict]:
    """Restore an archive into a new local bundle; never execute its tasks."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / f"import_{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix=".import-", dir=destination) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("The archive contains duplicate entries.")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
                raise ValueError("Unsupported finished-results archive. Task-settings JSONL is not a result archive.")
            records = manifest.get("tasks")
            if not isinstance(records, list) or not records:
                raise ValueError("The archive contains no finished tasks.")
            groups = set()
            for record in records:
                name = record["bundle"]
                if name is not None:
                    # Archive paths are never used as arbitrary filesystem paths.
                    suffix = name.removeprefix("bundles/b") if isinstance(name, str) else ""
                    if (not isinstance(name, str) or not name.startswith("bundles/b")
                            or not suffix.isascii() or not suffix.isdigit() or len(suffix) != 4):
                        raise ValueError("Invalid result-bundle path in archive.")
                    groups.add(name)
            for name in groups:
                dest = root / name
                dest.mkdir(parents=True)
                for db in ("summary.sqlite", "trend.sqlite"):
                    member = f"{name}/{db}"
                    if member in names:
                        (dest / db).write_bytes(archive.read(member))
                if not (dest / "summary.sqlite").is_file():
                    raise ValueError("The archive is missing a result database.")
                if (dest / "trend.sqlite").is_file():
                    with closing(_read_db(dest / "trend.sqlite")) as cx:
                        if cx.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                            raise ValueError("Invalid convergence database.")
                        cx.execute("SELECT runset_uid, metric, fe, mean, stderr, count FROM trends LIMIT 1")
            out = []
            for record in records:
                task = dict(record["task"])
                _validate_task(task, root / record["bundle"] if record["bundle"] else None)
                task["output_dir"] = str((final / record["bundle"]).resolve()) if record["bundle"] else None
                out.append(task)
            for name in groups:
                _repair_trends(root / name, [r['task'] for r in records if r['bundle'] == name])
        # Keep extracted data after the temporary validation directory is closed.
        os.replace(root, final)
    return out
