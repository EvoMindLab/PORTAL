import types

import numpy as np
import pytest

from portal_studio.algorithms.package_adapters import run_niapy_optimizer
from portal_studio.algorithms.runner import RunCanceled


class _Problem:
    def __init__(self):
        self.d = 2
        self.lb = np.array([-5.0, -5.0])
        self.ub = np.array([5.0, 5.0])
        self.max_fe = 7
        self.fe_count = 0
        self.best_f = np.inf
        self.best_x = None
        self.convergence = []

    def evaluate(self, x):
        self.fe_count += 1
        value = float(np.sum(np.asarray(x, dtype=float) ** 2))
        if value < self.best_f:
            self.best_f = value
            self.best_x = np.asarray(x, dtype=float).copy()
        self.convergence.append((self.fe_count, self.best_f))
        return value

    def is_done(self):
        return self.fe_count >= self.max_fe


class _FakeOptimizer:
    captured = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, task):
        self.__class__.captured = {
            "task": task,
            "kwargs": self.kwargs,
        }
        for _ in range(task.max_evals):
            task.problem.evaluate(np.zeros(2))


class _OverCallingOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, task):
        for _ in range(task.max_evals + 5):
            task.problem.evaluate(np.zeros(2))


@pytest.fixture
def fake_niapy(monkeypatch):
    class Problem:
        def __init__(self, *, dimension, lower, upper):
            self.dimension = dimension
            self.lower = lower
            self.upper = upper

        def evaluate(self, x):
            return self._evaluate(x)

    def import_fake_module(name):
        if name == "niapy.problems":
            return types.SimpleNamespace(Problem=Problem)
        if name == "niapy.task":
            return types.SimpleNamespace(
                Task=lambda **kwargs: types.SimpleNamespace(**kwargs)
            )
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "portal_studio.algorithms.package_adapters.import_module",
        import_fake_module,
    )


def test_run_niapy_optimizer_builds_problem_and_uses_fe_budget(fake_niapy):
    problem = _Problem()

    run_niapy_optimizer(
        problem,
        _FakeOptimizer,
        {"population_size": 12},
        seed=42,
    )

    assert problem.fe_count == 7
    task = _FakeOptimizer.captured["task"]
    assert task.max_evals == 7
    assert task.enable_logging is False
    assert task.problem.dimension == 2
    np.testing.assert_array_equal(task.problem.lower, problem.lb)
    np.testing.assert_array_equal(task.problem.upper, problem.ub)
    assert _FakeOptimizer.captured["kwargs"] == {"population_size": 12, "seed": 42}


def test_run_niapy_optimizer_does_not_evaluate_after_portal_budget(fake_niapy):
    problem = _Problem()

    run_niapy_optimizer(problem, _OverCallingOptimizer, {}, seed=42)

    assert problem.fe_count == problem.max_fe


@pytest.mark.parametrize("error_type", [ValueError, RunCanceled])
def test_run_niapy_optimizer_propagates_suppressed_worker_errors(
    fake_niapy, error_type,
):
    error = error_type("worker stopped")

    class SuppressingOptimizer:
        def __init__(self, **kwargs):
            self.exception = None

        def run(self, task):
            self.exception = error
            return None, None

    with pytest.raises(error_type, match="worker stopped") as caught:
        run_niapy_optimizer(_Problem(), SuppressingOptimizer, {})
    assert caught.value is error


def test_run_niapy_optimizer_missing_dependency_has_clear_error(monkeypatch):
    def fail_import(_name):
        raise ModuleNotFoundError("No module named 'niapy'")

    monkeypatch.setattr(
        "portal_studio.algorithms.package_adapters.import_module",
        fail_import,
    )
    problem = _Problem()

    with pytest.raises(RuntimeError, match="niapy is required"):
        run_niapy_optimizer(problem, _FakeOptimizer, {})


@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_run_niapy_optimizer_rejects_empty_or_nonfinite_results(fake_niapy, value):
    problem = _Problem()

    class InvalidOptimizer:
        def __init__(self, **kwargs):
            pass

        def run(self, task):
            if value is not None:
                problem.fe_count = 1
                problem.best_f = value
                problem.best_x = np.zeros(2)

    with pytest.raises(RuntimeError, match="without a finite evaluated solution"):
        run_niapy_optimizer(problem, InvalidOptimizer, {})
