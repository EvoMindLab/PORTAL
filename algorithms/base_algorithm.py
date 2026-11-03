# SPDX-License-Identifier: MIT
# PORTAL interface code; see the root LICENSE and LICENSING.md.
"""
Base Algorithm
==============
Abstract base class that every algorithm wrapper must inherit.

Rules:
    1. Your wrapper MUST inherit BaseAlgorithm.
    2. Your wrapper MUST implement run(problem).
    3. Inside run(), call problem.evaluate(x) or problem.evaluate_batch(X)
       for every fitness call.
    4. At the end of run(), call self._make_result(problem) to build
       the standard result dictionary.
    5. Never import fitness.py or load_instance.py directly.
"""

from abc import ABC, abstractmethod
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from portal_problem import PortalProblem


class BaseAlgorithm(ABC):
    """
    Abstract base class for all algorithm wrappers.

    Every algorithm wrapper:
        - Inherits this class
        - Implements run(problem) -> dict
        - Calls problem.evaluate(x) or problem.evaluate_batch(X) for fitness
          evaluations
        - Returns self._make_result(problem) at the end
    """

    def __init__(self, name: str):
        """
        Args:
            name : human-readable algorithm name (used in results)
        """
        self.name = name

    @abstractmethod
    def run(self, problem: PortalProblem) -> dict:
        """
        Run the algorithm on one PORTAL problem instance.

        The problem is already loaded and reset before this is called.
        The algorithm must call problem.evaluate(x) for every fitness
        evaluation and stop when problem.is_done() returns True. Algorithms may
        use problem.evaluate_batch(X) when candidates are independent and the
        batched call does not change the search trajectory.

        Args:
            problem : PortalProblem instance, ready to use

        Returns:
            result dict with keys:
                'algorithm'   : str    — algorithm name
                'best_f'      : float  — best fitness found
                'best_x'      : ndarray — best solution vector, shape (d,)
                'fe_used'     : int    — total function evaluations consumed
                'convergence' : list   — [(fe, best_f), ...] convergence curve
        """
        pass

    def _make_result(self, problem: PortalProblem) -> dict:
        """
        Build the standard result dictionary from a finished problem.

        Call this at the very end of run().

        Args:
            problem : the PortalProblem after the run is complete

        Returns:
            standard result dict
        """
        return {
            'algorithm'  : self.name,
            'best_f'     : float(problem.best_f),
            'best_x'     : problem.best_x.copy() if problem.best_x is not None
                           else np.full(problem.d, np.nan),
            'fe_used'    : int(problem.fe_count),
            'convergence': list(problem.convergence),
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
