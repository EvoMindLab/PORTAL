# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
SPSO-2011 — Standard Particle Swarm Optimisation 2011
=====================================================
Reference:
    Zambrano-Bigiarini, M., Clerc, M., & Rojas, R. (2013).
    Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline
    for future PSO improvements.

Key mechanisms:
    - Adaptive random informant topology.
    - Rotationally invariant hypersphere sampling around the gravity
      centre defined by the current, personal-best, and neighborhood-best
      positions.
    - Budget-controlled run: internal topology resets never stop the run;
      PORTAL's max_fe is the stopping criterion.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class SPSO2011(BaseAlgorithm):
    """Standard PSO 2011 with adaptive random topology."""

    def __init__(
        self,
        pop_size: int = 40,
        informants: int = 3,
        w: float = 1.0 / (2.0 * np.log(2.0)),
        c: float = 0.5 + np.log(2.0),
    ):
        super().__init__(name='SPSO')
        self.pop_size = pop_size
        self.informants = informants
        self.w = w
        self.c = c

    def run(self, problem: PortalProblem) -> dict:
        d = problem.d
        lb = problem.lb
        ub = problem.ub

        if self.pop_size < 1:
            raise ValueError("SPSO requires pop_size >= 1.")
        if problem.is_done():
            return self._make_result(problem)

        n = min(self.pop_size, problem.remaining_fe())

        pos = lb + np.random.rand(n, d) * (ub - lb)
        vel = (lb - pos) + np.random.rand(n, d) * (ub - lb)
        fit = np.array([problem.evaluate(pos[i]) for i in range(n)])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_fit = float(pbest_fit[gbest_idx])

        topology = self._make_topology(n)

        while not problem.is_done():
            generation_improved = False

            for i in range(n):
                if problem.is_done():
                    break

                informant_idx = np.flatnonzero(topology[i])
                best_neighbor = informant_idx[np.argmin(pbest_fit[informant_idx])]
                lbest = pbest_pos[best_neighbor]

                attractor = self._sample_hypersphere_attractor(
                    pos[i], pbest_pos[i], lbest
                )
                vel[i] = self.w * vel[i] + (attractor - pos[i])
                pos[i] = pos[i] + vel[i]

                out = (pos[i] < lb) | (pos[i] > ub)
                if np.any(out):
                    pos[i] = np.clip(pos[i], lb, ub)
                    vel[i, out] = 0.0

                f_new = problem.evaluate(pos[i])
                if f_new < pbest_fit[i]:
                    pbest_fit[i] = f_new
                    pbest_pos[i] = pos[i].copy()

                    if f_new < gbest_fit:
                        gbest_fit = f_new
                        generation_improved = True

            if not generation_improved and not problem.is_done():
                topology = self._make_topology(n)

        return self._make_result(problem)

    def _make_topology(self, n: int) -> np.ndarray:
        if n == 1:
            return np.ones((1, 1), dtype=bool)

        k = max(1, int(self.informants))
        prob = 1.0 - (1.0 - 1.0 / n) ** k
        topology = np.random.rand(n, n) < prob
        np.fill_diagonal(topology, True)
        return topology

    def _sample_hypersphere_attractor(self, x, pbest, lbest):
        c = self.c
        if np.allclose(pbest, lbest):
            centre = x + 0.5 * c * (pbest - x)
        else:
            centre = x + (c / 3.0) * (pbest + lbest - 2.0 * x)

        radius = np.linalg.norm(centre - x)
        if radius <= 0.0:
            return centre.copy()

        direction = np.random.normal(size=x.size)
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            return centre.copy()

        direction /= norm
        r = radius * (np.random.rand() ** (1.0 / x.size))
        return centre + r * direction
