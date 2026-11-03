"""PORTAL adapters for separately installed NiaPy and pymoo optimizers.

NiaPy: https://github.com/NiaOrg/NiaPy
G. Vrbancic et al. (2018), JOSS 3(23), 613.
https://doi.org/10.21105/joss.00613

pymoo: https://github.com/anyoptimization/pymoo
J. Blank and K. Deb (2020), IEEE Access 8, 89497-89509.
https://doi.org/10.1109/ACCESS.2020.2990567

Only PORTAL budget, seeding, evaluation, and error-handling integration lives
here; the optimizer implementations are supplied by those packages.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


def run_pymoo_optimizer(
    problem,
    algorithm_cls: type,
    algorithm_kwargs: dict[str, Any],
    *,
    seed: int | None = None,
) -> None:
    """Run a PyMOO single-objective algorithm through the PORTAL problem
    interface.

    PyMOO calls its objective function once per generation with the whole
    population at once, which maps directly onto PORTAL's own
    ``evaluate_batch``. Its ``("n_evals", N)`` termination is already
    expressed in the same units as PORTAL's FE budget, so no unit conversion
    is needed here.
    """
    try:
        problem_module = import_module("pymoo.core.problem")
        optimize_module = import_module("pymoo.optimize")
    except Exception as exc:  # noqa: BLE001 - normalize optional dependency errors
        raise RuntimeError(
            "pymoo is required for this package-backed algorithm. "
            "Install pymoo or choose another algorithm."
        ) from exc

    Problem = problem_module.Problem
    minimize = optimize_module.minimize

    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    d = int(problem.d)

    def obj_func_batch(x: np.ndarray) -> np.ndarray:
        values = problem.evaluate_batch(np.asarray(x, dtype=float))
        if len(values) < x.shape[0]:
            # Budget ran out mid-generation: evaluate_batch truncates rather
            # than raising, so pad the remaining rows with +inf (never
            # selected) to satisfy PyMOO's one-F-per-row contract.
            pad = np.full(x.shape[0] - len(values), np.inf)
            values = np.concatenate([values, pad])
        return values

    class _CallbackProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=d, n_obj=1, n_ieq_constr=0, xl=lb, xu=ub)

        def _evaluate(self, x, out, *args, **kwargs) -> None:
            out["F"] = obj_func_batch(x).reshape(-1, 1)

    algorithm = algorithm_cls(**algorithm_kwargs)
    minimize(
        _CallbackProblem(),
        algorithm,
        ("n_evals", int(problem.max_fe)),
        seed=seed,
        verbose=False,
    )


def run_niapy_optimizer(
    problem,
    algorithm_cls: type,
    algorithm_kwargs: dict[str, Any],
    *,
    seed: int | None = None,
) -> None:
    """Run a NiaPy algorithm through the PORTAL problem interface.

    NiaPy's own ``Task(max_evals=...)`` already expresses an FE budget
    natively, so no unit conversion is needed here. NiaPy evaluates one
    candidate per call rather than a whole population per call (contrast
    PyMOO's ``evaluate_batch``-shaped calling convention above).
    """
    try:
        problem_module = import_module("niapy.problems")
        task_module = import_module("niapy.task")
    except Exception as exc:  # noqa: BLE001 - normalize optional dependency errors
        raise RuntimeError(
            "niapy is required for this package-backed algorithm. "
            "Install niapy or choose another algorithm."
        ) from exc

    NiapyProblem = problem_module.Problem
    Task = task_module.Task

    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    d = int(problem.d)

    def obj_func(x) -> float:
        is_done = getattr(problem, "is_done", None)
        if callable(is_done):
            exhausted = bool(is_done())
        else:
            exhausted = int(getattr(problem, "fe_count", 0)) >= int(problem.max_fe)
        if exhausted:
            return float("inf")
        return float(problem.evaluate(np.asarray(x, dtype=float)))

    class _CallbackProblem(NiapyProblem):
        def __init__(self) -> None:
            super().__init__(dimension=d, lower=lb, upper=ub)

        def _evaluate(self, x) -> float:
            return obj_func(x)

    task = Task(
        problem=_CallbackProblem(),
        max_evals=int(problem.max_fe),
        enable_logging=False,
    )
    algorithm = algorithm_cls(seed=seed, **algorithm_kwargs)
    algorithm.run(task)
    # NiaPy stores failures instead of raising them outside the main process
    # or thread. Propagate the original exception so GUI workers report a
    # failed/canceled run rather than persisting an incomplete result.
    exception = getattr(algorithm, "exception", None)
    if exception is not None:
        raise exception
    if (int(problem.fe_count) == 0 or not np.isfinite(problem.best_f)
            or problem.best_x is None or not np.all(np.isfinite(problem.best_x))):
        raise RuntimeError("NiaPy run ended without a finite evaluated solution.")
