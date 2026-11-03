"""Random Search -- Uniform random sampling within the box bounds, one FE per sample."""
from __future__ import annotations

import os
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if os.path.join(_REPO, "algorithms") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "algorithms"))

from base_algorithm import BaseAlgorithm   # noqa: E402


class RandomSearch(BaseAlgorithm):
    """Uniform random sampling within the box bounds, one FE per sample."""

    def __init__(self) -> None:
        super().__init__(name="Random Search")

    def run(self, problem) -> dict:
        while not problem.is_done():
            x = problem.lb + np.random.rand(problem.d) * (problem.ub - problem.lb)
            problem.evaluate(x)
        return self._make_result(problem)
