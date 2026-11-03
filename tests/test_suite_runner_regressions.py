"""Public, bounded regressions for the curation and augmentation runners."""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Recipe_Selection import _execution, run_curation_solvers as performance, run_ela as ela
from Recipe_Selection.augmentation_pipeline.config import DEFAULT_CONFIG as AUGMENT_DEFAULTS
from Recipe_Selection.augmentation_pipeline.runner import AugmentationDatasetRunner
from Recipe_Selection.suite_pipeline.config import DEFAULT_CONFIG, validate_config
from Recipe_Selection.suite_pipeline.orchestrator import SuitePipelineRunner
from pflacco.classical_ela_features import calculate_dispersion


def _numeric_worker_state():
    # A spawned worker must limit NumPy even when the caller supplied 8 threads.
    np.ones((4, 4)) @ np.ones((4, 4))
    return {
        "environment": {name: os.environ.get(name) for name in _execution.NUMERIC_THREAD_VARIABLES},
        "threads": [pool["num_threads"] for pool in threadpool_info()],
    }


def _bounds_file(tmp_path, name, lower, upper):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"bounds_min": lower, "bounds_max": upper}), encoding="utf-8")
    return path


def test_mixed_bounds_dispersion_matches_uncached_reference(tmp_path):
    unit = np.random.default_rng(42).uniform(size=(200, 2))
    paths = [_bounds_file(tmp_path, "narrow", -100, 100),
             _bounds_file(tmp_path, "wide", -200, 200)]
    cfg = {"OUTPUT_CSV": str(tmp_path / "ela.csv")}
    by_bounds = ela._build_dispersion_stats_by_bounds(cfg, paths, unit, 2, 200, 42)
    cfg["ELA_DISPERSION_STATS_BY_BOUNDS"] = by_bounds
    assert set(by_bounds) == {(-100.0, 100.0), (-200.0, 200.0)}
    assert by_bounds[(-200, 200)]["mean"] == pytest.approx(2 * by_bounds[(-100, 100)]["mean"])
    for lower, upper in by_bounds:
        x = unit * (upper - lower) + lower
        y = np.sum(x**2, axis=1)
        cached = calculate_dispersion(
            x, y, dist_full_stats=ela._dispersion_stats_for_recipe(cfg, lower, upper),
        )
        exact = calculate_dispersion(x, y)
        for key in exact:
            if not key.endswith("costs_runtime"):
                assert cached[key] == pytest.approx(exact[key], rel=1e-12, abs=1e-12)


def test_one_dispersion_reference_per_distinct_domain(tmp_path, monkeypatch):
    paths = [_bounds_file(tmp_path, "a", -100, 100),
             _bounds_file(tmp_path, "b", -100, 100),
             _bounds_file(tmp_path, "c", -200, 200)]
    calls = []
    def build(cfg, sample, d, n, seed, lower, upper):
        calls.append((lower, upper))
        return {"mean": upper - lower, "median": upper - lower}
    monkeypatch.setattr(ela, "_build_or_load_dispersion_full_stats", build)
    actual = ela._build_dispersion_stats_by_bounds({}, paths, np.zeros((3, 2)), 2, 3, 42)
    assert len(calls) == len(actual) == 2


def test_dispersion_caches_do_not_collide_for_nearby_bounds(tmp_path):
    cfg = {"OUTPUT_CSV": str(tmp_path / "ela.csv")}
    a = ela._dispersion_stats_cache_path(cfg, 2, 200, 42, -100, 100)
    b = ela._dispersion_stats_cache_path(cfg, 2, 200, 42, -100.000001, 100)
    assert a != b
    assert a.name.endswith("_bounds-100_100.npz")


def test_dispersion_cache_reuses_saved_domain_statistics(tmp_path, monkeypatch):
    cfg = {"OUTPUT_CSV": str(tmp_path / "ela.csv")}
    unit = np.random.default_rng(1).uniform(size=(30, 2))
    expected = ela._build_or_load_dispersion_full_stats(cfg, unit, 2, 30, 42, -100, 100)
    def unexpected(*args, **kwargs):
        pytest.fail("An existing domain cache must not recompute pairwise distances.")
    monkeypatch.setattr("scipy.spatial.distance.pdist", unexpected)
    assert ela._build_or_load_dispersion_full_stats(cfg, unit, 2, 30, 42, -100, 100) == expected


def test_legacy_unqualified_dispersion_caches_are_not_reused(tmp_path):
    cfg = {"OUTPUT_CSV": str(tmp_path / "ela.csv")}
    unit = np.random.default_rng(1).uniform(size=(30, 2))
    current = ela._dispersion_stats_cache_path(cfg, 2, 30, 42, -100, 100)
    legacy = current.with_name(current.name.replace("full_stats_v2_", "full_stats_"))
    legacy.parent.mkdir(parents=True)
    # Old rounded filenames could alias statistics from another domain.
    np.savez(legacy, mean=-1.0, median=-1.0)
    actual = ela._build_or_load_dispersion_full_stats(cfg, unit, 2, 30, 42, -100, 100)
    assert current.exists() and current != legacy
    assert actual["mean"] > 0 and actual["median"] > 0


def test_missing_dispersion_domain_is_not_silently_reused():
    cfg = {"ELA_DISPERSION_STATS_BY_BOUNDS": {(-100, 100): {"mean": 1, "median": 1}}}
    with pytest.raises(ValueError, match="No ELA dispersion reference"):
        ela._dispersion_stats_for_recipe(cfg, -200, 200)
    assert ela._dispersion_stats_for_recipe({}, -200, 200) is None


def test_ela_worker_selects_its_own_domain_statistics(tmp_path, monkeypatch):
    payload = json.loads((ROOT / "examples/recipes/demo_01.json").read_text(encoding="utf-8-sig"))
    payload.update(bounds_min=-200, bounds_max=200)
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(ela, "_load_instance", lambda *args: (
        {"d": 2, "bounds_min": -200, "bounds_max": 200}, "single",
    ))
    expected = {"mean": 2.0, "median": 2.0}
    observed = []
    def compute(*args, **kwargs):
        observed.append(kwargs["dispersion_full_stats"])
        return {"ela_probe": 1.0}, []
    monkeypatch.setattr(ela, "_compute_ela", compute)
    cfg = {"ELA_SAMPLE_FACTOR": 100, "ELA_SEED": 42,
           "ELA_DISPERSION_STATS_BY_BOUNDS": {
               (-100, 100): {"mean": 1.0, "median": 1.0}, (-200, 200): expected,
           }}
    row = ela.process_sample((str(path), cfg))
    assert row["error"] is None
    assert observed == [expected]


@pytest.mark.parametrize("checkpoints", [
    [100.5, 200.5], [100.0], [True], ["100"], [None], [0], [-1], [],
    [100, 100], [200, 100],
])
def test_invalid_checkpoints_rejected_without_measurement(checkpoints):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["performance"]["checkpoints"] = checkpoints
    with pytest.raises(ValueError, match="performance.checkpoints"):
        validate_config(cfg)


def test_integer_checkpoints_remain_valid():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["performance"]["checkpoints"] = [1, 100, 1000]
    validate_config(cfg)


def test_cli_rejects_fractional_checkpoints_before_creating_outputs(tmp_path):
    output = tmp_path / "must_not_exist"
    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        f"[run]\noutput_root = '{output.as_posix()}'\n[performance]\ncheckpoints = [100.5]\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "run_suite_pipeline.py"), "--config", str(cfg)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "performance.checkpoints[0] must be an integer" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("requested,cpus,tasks,platform,expected", [
    (121, 128, 1000, "win32", 61), (128, 128, 1000, "linux", 128),
    (128, 128, 1000, "darwin", 128), (8, 32, 3, "win32", 3),
    (100, 4, 1000, "win32", 4), (1, 128, 1000, "win32", 1),
])
def test_process_worker_cap(requested, cpus, tasks, platform, expected):
    assert _execution.cap_process_workers(
        requested, cpus, task_count=tasks, platform=platform,
    ) == expected


def test_performance_resolver_caps_automatic_and_explicit_windows_workers(monkeypatch):
    monkeypatch.setattr(_execution.sys, "platform", "win32")
    assert performance._resolve_max_parallel_runs(None, cpu_count=128) == 61
    assert performance._resolve_max_parallel_runs(128, cpu_count=128) == 61
    assert performance._resolve_max_parallel_runs(2, cpu_count=128) == 2


def test_augmentation_resolver_caps_workers_and_pending_work(tmp_path, monkeypatch):
    cfg = copy.deepcopy(AUGMENT_DEFAULTS)
    cfg["run"]["output_root"] = str(tmp_path / "output")
    runner = AugmentationDatasetRunner(cfg, project_root=ROOT)
    monkeypatch.setattr(_execution.sys, "platform", "win32")
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert runner._resolve_workers(1000) == 61
    assert runner._resolve_workers(4) == 4
    cfg["execution"]["max_workers"] = 2
    assert runner._resolve_workers(1000) == 2


def test_thread_environment_is_scoped_and_restored_on_failure(monkeypatch):
    for index, name in enumerate(_execution.NUMERIC_THREAD_VARIABLES):
        if index % 2:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "8")
    before = {name: os.environ.get(name) for name in _execution.NUMERIC_THREAD_VARIABLES}
    with pytest.raises(RuntimeError, match="probe"):
        with _execution.numeric_worker_environment():
            assert all(os.environ[name] == "1" for name in _execution.NUMERIC_THREAD_VARIABLES)
            raise RuntimeError("probe")
    assert {name: os.environ.get(name) for name in before} == before


def test_existing_native_thread_pools_are_limited_and_restored(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    with threadpool_limits(limits=4):
        before = [(pool["prefix"], pool["num_threads"]) for pool in threadpool_info()]
        with _execution.single_threaded_numerics():
            assert threadpool_info()
            assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        assert [(pool["prefix"], pool["num_threads"]) for pool in threadpool_info()] == before
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_spawned_workers_override_inherited_thread_settings(monkeypatch):
    for name in _execution.NUMERIC_THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    with _execution.numeric_worker_environment(), ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn"),
        initializer=_execution.initialize_numeric_worker,
    ) as executor:
        pending = [executor.submit(_numeric_worker_state) for _ in range(2)]
        states = [future.result(timeout=30) for future in pending]
    for state in states:
        assert all(value == "1" for value in state["environment"].values())
        assert state["threads"] and all(value == 1 for value in state["threads"])
    assert all(os.environ[name] == "8" for name in _execution.NUMERIC_THREAD_VARIABLES)


def test_serial_performance_measurement_limits_threads_and_restores_caller(monkeypatch):
    monkeypatch.setattr(performance, "process_solver_variant", lambda task: _numeric_worker_state())
    with threadpool_limits(limits=4):
        rows = list(performance._iter_completed_run_tasks([{}], 1))
        assert all(value == 1 for value in rows[0][0]["threads"])
        assert all(pool["num_threads"] == 4 for pool in threadpool_info())


def test_execution_helper_participates_in_resume_fingerprints(tmp_path, monkeypatch):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["run"].update(recipes_root=str(ROOT / "Suite"), output_root=str(tmp_path / "suite"))
    runner = SuitePipelineRunner(cfg, project_root=ROOT)
    helper = ROOT / "Recipe_Selection/_execution.py"
    for stage in ("ela", "performance"):
        assert helper in runner._build_stage_plans()[stage].sources
    cfg = copy.deepcopy(AUGMENT_DEFAULTS)
    cfg["run"]["output_root"] = str(tmp_path / "augmentation")
    runner = AugmentationDatasetRunner(cfg, project_root=ROOT)
    runner._compute_signatures()
    before = runner.implementation_fingerprint
    import Recipe_Selection.augmentation_pipeline.runner as augmentation
    original = augmentation._hash_file
    monkeypatch.setattr(augmentation, "_hash_file", lambda path: (
        "changed-helper" if path == helper else original(path)
    ))
    runner._compute_signatures()
    assert runner.implementation_fingerprint != before
