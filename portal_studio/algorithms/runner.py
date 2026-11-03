"""GUI-facing solver runner.

Pipes an in-memory instance dict through a tempfile + RecordingProblem +
solver from the registry, and returns the populated tape plus the known
global optimum for the error chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from portal_studio.algorithms._temp_instance import instance_as_tempfile
from portal_core.optimum import global_optimum_value
from portal_studio.algorithms.recording_problem import RecordingProblem
from portal_studio.algorithms.registry import SOLVERS
from portal_studio.algorithms.domain_validation import validate_centres_in_domain
from portal_studio.algorithms.random_state import seeded_random_state
from portal_studio.validation import validate_instance_for_gui


class RunCanceled(Exception):
    """Raised from a run's progress callback to abort the solver early.

    Propagates up through ``RecordingProblem.evaluate`` and the solver loop so
    the GUI can stop a long run when the user clicks Cancel.
    """


@dataclass
class RunResult:
    """Result bundle for one solver run."""
    problem: RecordingProblem
    solver_name: str
    seed: int
    budget: int
    global_optimum: float


def run_solver_on_instance(*, instance: dict, solver_name: str,
                           budget: int, seed: int,
                           params: dict | None = None,
                           realization_type: str = "none",
                           realization_base_seed="train",
                           realization_sample_id: int = 1,
                           progress: Callable[[int, int], None] | None = None,
                           ) -> RunResult:
    if solver_name not in SOLVERS:
        raise KeyError(f"Unknown solver: {solver_name!r}; have {list(SOLVERS)}")
    cls = SOLVERS[solver_name]
    validate_instance_for_gui(instance)
    with seeded_random_state(seed), instance_as_tempfile(instance) as path:
        problem = RecordingProblem(path, max_fe=int(budget),
                                   record_convergence=False,
                                   realization_type=realization_type,
                                   realization_base_seed=realization_base_seed,
                                   realization_sample_id=int(realization_sample_id),
                                   progress_callback=progress)
        # Check the actual realised instance, including Type 2 coordinate
        # relabelling, before any solver evaluates it or an optimum is used.
        validate_centres_in_domain(problem.instance)
        problem.algorithm_seed = int(seed)
        cls(**(params or {})).run(problem)
    return RunResult(
        problem=problem,
        solver_name=solver_name,
        seed=int(seed),
        budget=int(budget),
        global_optimum=global_optimum_value(problem.instance),
    )
