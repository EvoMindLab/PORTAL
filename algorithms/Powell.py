# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
Bounded Powell Direction-Set Local Search
=========================================
Reference:
    Powell, M. J. D. (1964).
    An efficient method for finding the minimum of a function of several
    variables without calculating derivatives.
    The Computer Journal, 7(2), 155-162.
    DOI: 10.1093/comjnl/7.2.155

This is a PORTAL adaptation of Powell's derivative-free direction-set
method. It uses bounded batched one-dimensional searches along a direction
set, then adds the net displacement as a Powell-style conjugate direction.
Bounds and random restarts are added so the wrapper remains valid for bounded
PORTAL instances and consumes the full FE budget.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class PowellLocalSearch(BaseAlgorithm):
    """Budget-controlled bounded Powell-type local search."""

    def __init__(
        self,
        initial_step_fraction: float = 0.25,
        min_step_fraction: float = 1e-6,
        shrink: float = 0.5,
        line_evals: int = 8,
        tol: float = 1e-12,
    ):
        super().__init__(name='Powell Local Search')
        self.initial_step_fraction = initial_step_fraction
        self.min_step_fraction = min_step_fraction
        self.shrink = shrink
        self.line_evals = line_evals
        self.tol = tol

    def run(self, problem: PortalProblem) -> dict:
        d = problem.d
        lb = problem.lb
        ub = problem.ub
        span = ub - lb
        scale = float(np.mean(span))

        if problem.is_done():
            return self._make_result(problem)

        min_step = max(self.min_step_fraction * scale, 1e-15)

        x = problem.random_solution()
        f = problem.evaluate(x)
        directions = np.eye(d)
        step = max(self.initial_step_fraction * scale, min_step)

        while not problem.is_done():
            cycle_start_x = x.copy()
            cycle_start_f = f

            for j in range(d):
                if problem.is_done():
                    break
                x, f = self._line_search(problem, x, f, directions[j], step, lb, ub)

            if problem.is_done():
                break

            displacement = x - cycle_start_x
            disp_norm = np.linalg.norm(displacement)
            if disp_norm > 0.0:
                new_dir = displacement / disp_norm
                x_new, f_new = self._line_search(problem, x, f, new_dir, step, lb, ub)
                if f_new < f:
                    directions[:-1] = directions[1:]
                    directions[-1] = new_dir
                    x, f = x_new, f_new

            improvement = cycle_start_f - f
            if improvement <= self.tol * max(1.0, abs(cycle_start_f)):
                step *= self.shrink

            if step < min_step and not problem.is_done():
                x = problem.random_solution()
                f = problem.evaluate(x)
                directions = np.eye(d)
                step = max(self.initial_step_fraction * scale, min_step)

        return self._make_result(problem)

    def _line_search(self, problem, x, f, direction, step, lb, ub):
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            return x, f
        direction = direction / norm

        lo, hi = self._alpha_bounds(x, direction, step, lb, ub)
        if hi <= lo:
            return x, f

        best_x = x
        best_f = f

        n_eval = min(max(1, int(self.line_evals)), problem.remaining_fe())
        if n_eval <= 0:
            return best_x, best_f

        if n_eval == 1:
            alphas = np.array([(lo + hi) / 2.0])
        else:
            alphas = np.linspace(lo, hi, n_eval + 2)[1:-1]
        candidates = np.clip(x + alphas[:, None] * direction, lb, ub)
        values = problem.evaluate_batch(candidates)
        if len(values) == 0:
            return best_x, best_f

        local_best = int(np.argmin(values))
        if values[local_best] < best_f:
            best_x = candidates[local_best].copy()
            best_f = float(values[local_best])

        return best_x, best_f

    def _alpha_bounds(self, x, direction, step, lb, ub):
        lo = -step
        hi = step
        pos = direction > 0
        neg = direction < 0

        if np.any(pos):
            hi = min(hi, np.min((ub[pos] - x[pos]) / direction[pos]))
            lo = max(lo, np.max((lb[pos] - x[pos]) / direction[pos]))
        if np.any(neg):
            hi = min(hi, np.min((lb[neg] - x[neg]) / direction[neg]))
            lo = max(lo, np.max((ub[neg] - x[neg]) / direction[neg]))

        return float(lo), float(hi)
