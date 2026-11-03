"""Exercise optional NiaPy integration, including real parallel workers."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from portal_studio.algorithms.registry import REGISTERED_PACKAGE_SPECS, SOLVERS
from portal_studio.algorithms.runner import RunCanceled

pytest.importorskip("niapy")

NIAPY_NAMES = sorted(name for name, spec in REGISTERED_PACKAGE_SPECS.items()
                     if spec.package == "niapy")


class _SphereProblem:
    def __init__(self):
        self.d = 3
        self.lb = np.full(self.d, -5.0)
        self.ub = np.full(self.d, 5.0)
        self.max_fe = 201
        self.algorithm_seed = 42
        self.fe_count = 0
        self.best_f = np.inf
        self.best_x = None
        self.convergence = []

    def is_done(self):
        return self.fe_count >= self.max_fe

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        assert not self.is_done(), "Evaluation budget exceeded"
        assert np.all(np.isfinite(x))
        assert np.all(x >= self.lb) and np.all(x <= self.ub)
        self.fe_count += 1
        value = float(np.sum(x ** 2))
        if value < self.best_f:
            self.best_f = value
            self.best_x = x.copy()
        self.convergence.append((self.fe_count, self.best_f))
        return value


@pytest.mark.parametrize("name", NIAPY_NAMES)
def test_niapy_catalog_runs_with_finite_bounded_exact_budget(name):
    problem = _SphereProblem()
    result = SOLVERS[name](pop_size=10).run(problem)
    assert result["fe_used"] == problem.max_fe
    assert np.isfinite(result["best_f"])
    assert np.all(np.isfinite(result["best_x"]))
    assert len(result["convergence"]) == problem.max_fe


@pytest.mark.parametrize("source_name", ["f01_r007", "f04_r005"])
@pytest.mark.parametrize("name", NIAPY_NAMES)
def test_niapy_gui_defaults_run_on_unified_single_and_multiblock_recipes(name, source_name):
    from portal_core.load_instance import load_instance
    from portal_studio.algorithms.runner import run_solver_on_instance

    source = Path(__file__).resolve().parents[1] / f"Benchmark_S50/{source_name}.json"
    result = run_solver_on_instance(
        instance=load_instance(str(source)), solver_name=name, budget=201, seed=42)
    assert result.problem.fe_count == 201
    assert np.isfinite(result.problem.best_f)
    assert len(result.problem.tape) == 201


def _execute_niapy_mixed(unit, composite, output_dir):
    """Inject an internal NiaPy error in only the middle run in each worker."""
    from niapy.algorithms.basic.de import DifferentialEvolution
    from portal_studio.algorithms import niapy_catalog
    from portal_studio.experiment_runner.runner import execute_run

    class FailingDE(DifferentialEvolution):
        def run_iteration(self, *args, **kwargs):
            raise ValueError("NiaPy iteration failed")

    original = niapy_catalog.load_niapy_class
    if unit.run_index == 1:
        niapy_catalog.load_niapy_class = lambda _path: FailingDE
    try:
        return execute_run(unit, composite, output_dir)
    finally:
        niapy_catalog.load_niapy_class = original


@pytest.mark.parametrize("workers", [1, 2])
def test_real_niapy_batch_continues_after_internal_iteration_error(tmp_path, monkeypatch, workers):
    from portal_core.load_instance import load_instance
    from portal_studio.experiment_runner.config import BatchConfig, TaskConfig
    from portal_studio.experiment_runner import scheduler

    monkeypatch.setattr(scheduler, "execute_run", _execute_niapy_mixed)
    monkeypatch.setattr(scheduler.os, "cpu_count", lambda: 4)
    cfg = BatchConfig(tasks=[TaskConfig(label="mixed", algorithm="NIAPY DE", runs=3,
                                       algorithm_params={"pop_size": 10})],
                      fe_budget=101, max_parallel_runs=workers)
    source = Path(__file__).resolve().parents[1] / "Benchmark_S50/f04_r005.json"
    final = scheduler.BatchScheduler(cfg, load_instance(str(source)), str(tmp_path)).run()
    assert final.status == "finished"
    assert (final.completed_runs, final.failed_runs) == (2, 1)
    assert "NiaPy iteration failed" in final.failures[0].message
    with sqlite3.connect(tmp_path / "summary.sqlite") as cx:
        assert cx.execute("SELECT run_index FROM runs ORDER BY run_index").fetchall() == [(0,), (2,)]


def _run_failing_objective(error_type):
    """Module-level entry point so Windows process workers can import it."""
    class FailingProblem(_SphereProblem):
        def evaluate(self, x):
            raise error_type("objective stopped in worker")

    SOLVERS["NIAPY DE"](pop_size=10).run(FailingProblem())


@pytest.mark.parametrize("executor_class", [ThreadPoolExecutor, ProcessPoolExecutor])
@pytest.mark.parametrize("error_type", [ValueError, RunCanceled])
def test_niapy_workers_propagate_objective_errors_and_cancellation(
    executor_class, error_type,
):
    with executor_class(max_workers=1) as executor:
        future = executor.submit(_run_failing_objective, error_type)
        with pytest.raises(error_type, match="objective stopped in worker"):
            future.result(timeout=30)


def _run_failing_experiment(source_path, output_dir):
    from niapy.algorithms.basic.de import DifferentialEvolution
    from portal_core.load_instance import load_instance
    from portal_studio.algorithms import niapy_catalog
    from portal_studio.experiment_runner.runner import RunUnit, execute_run

    class FailingDE(DifferentialEvolution):
        def run_iteration(self, *args, **kwargs):
            raise ValueError("optimizer failed in worker")

    original_loader = niapy_catalog.load_niapy_class
    niapy_catalog.load_niapy_class = lambda _path: FailingDE
    try:
        unit = RunUnit(
            runset_uid="case::DE", task_label="case", algorithm="NIAPY DE",
            algorithm_params={"pop_size": 10}, run_index=0, seed=42,
            fe_budget=101,
        )
        execute_run(unit, load_instance(source_path), output_dir)
    finally:
        niapy_catalog.load_niapy_class = original_loader


def test_failed_niapy_experiment_worker_does_not_persist_success(tmp_path):
    from portal_studio.experiment_runner.store import ResultStore

    ResultStore(str(tmp_path)).ensure_schema()
    source = Path(__file__).resolve().parents[1] / "Benchmark_S50/f01_r007.json"
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_failing_experiment, str(source), str(tmp_path))
        with pytest.raises(ValueError, match="optimizer failed in worker"):
            future.result(timeout=30)
    with sqlite3.connect(tmp_path / "summary.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
