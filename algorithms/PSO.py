# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
Standard PSO — Canonical Particle Swarm Optimization with Inertia Weight
=========================================================================
Reference:
    Kennedy, J., & Eberhart, R. C. (1995).
    Particle swarm optimization.
    Proceedings of the IEEE International Conference on Neural Networks,
    Vol. 4, pp. 1942–1948.
    DOI: 10.1109/ICNN.1995.488968

    Shi, Y., & Eberhart, R. C. (1998).
    A modified particle swarm optimizer.
    Proceedings of the IEEE Congress on Evolutionary Computation (CEC 1998),
    pp. 69–73.
    DOI: 10.1109/ICEC.1998.699146

Key mechanism:
    Each particle has a position x and a velocity v.
    Velocity is updated by three components:
        1. Inertia:    w  * v          (momentum from previous step)
        2. Cognitive:  c1 * r1 * (pbest - x)  (pull toward personal best)
        3. Social:     c2 * r2 * (gbest - x)  (pull toward global best)

    Inertia weight w decreases linearly from w_max to w_min over the run,
    encouraging broad exploration early and fine exploitation late.

Parameters (defaults follow Shi & Eberhart 1998 recommendations):
    pop_size : swarm size      (default: 30 or 10*d for small d, capped at 50)
    w_max    : initial inertia (default: 0.9)
    w_min    : final inertia   (default: 0.4)
    c1       : cognitive coeff (default: 2.0)
    c2       : social coeff    (default: 2.0)
    v_max_r  : v_max as ratio of search range (default: 0.2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class StandardPSO(BaseAlgorithm):
    """
    Canonical PSO with linearly decreasing inertia weight.
    """

    def __init__(
        self,
        pop_size: int  = None,   # None = auto: min(10*d, 50)
        w_max:    float = 0.9,
        w_min:    float = 0.4,
        c1:       float = 2.0,
        c2:       float = 2.0,
        v_max_r:  float = 0.2,   # v_max = v_max_r * (ub - lb)
    ):
        super().__init__(name='Standard PSO')
        self.pop_size = pop_size
        self.w_max    = w_max
        self.w_min    = w_min
        self.c1       = c1
        self.c2       = c2
        self.v_max_r  = v_max_r

    def run(self, problem: PortalProblem) -> dict:
        """
        Run Standard PSO on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb          # shape (d,)
        ub     = problem.ub          # shape (d,)
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        N     = self.pop_size if self.pop_size is not None else min(10 * d, 50)
        if N < 1:
            raise ValueError("Standard PSO requires pop_size >= 1.")
        N = min(N, problem.remaining_fe())
        if N < 1:
            return self._make_result(problem)

        w_max = self.w_max
        w_min = self.w_min
        c1    = self.c1
        c2    = self.c2

        # Velocity bounds: ±v_max per dimension
        v_max = self.v_max_r * (ub - lb)   # shape (d,)

        # ── Initialisation ────────────────────────────────────────
        # Positions: uniform random in [lb, ub]
        pos = lb + np.random.rand(N, d) * (ub - lb)

        # Velocities: uniform random in [-v_max, +v_max]
        vel = -v_max + np.random.rand(N, d) * (2 * v_max)

        # Evaluate initial population
        fit = problem.evaluate_batch(pos)

        # Personal bests
        pbest_pos = pos.copy()           # shape (N, d)
        pbest_fit = fit.copy()           # shape (N,)

        # Global best
        g_idx    = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[g_idx].copy()   # shape (d,)
        gbest_fit = float(pbest_fit[g_idx])

        # ── Main loop ─────────────────────────────────────────────
        # Standard synchronous PSO generation: update all particles using the
        # same current global best, batch-evaluate, then refresh pbest/gbest.
        while not problem.is_done():

            # Linearly decreasing inertia weight
            # w goes from w_max (start) to w_min (end of budget)
            progress = problem.fe_count / max_fe        # in [0, 1]
            w = w_max - (w_max - w_min) * progress     # linear decay

            batch_size = min(N, problem.remaining_fe())
            if batch_size <= 0:
                break

            idx = np.arange(batch_size)

            # Random coefficients (drawn fresh each particle each step)
            r1 = np.random.rand(batch_size, d)
            r2 = np.random.rand(batch_size, d)

            # ── Velocity update ───────────────────────────────────
            vel[idx] = (
                w * vel[idx]
                + c1 * r1 * (pbest_pos[idx] - pos[idx])
                + c2 * r2 * (gbest_pos - pos[idx])
            )

            # Clamp velocity to [-v_max, +v_max]
            vel[idx] = np.clip(vel[idx], -v_max, v_max)

            # ── Position update ───────────────────────────────────
            pos[idx] = np.clip(pos[idx] + vel[idx], lb, ub)

            # ── Evaluate ──────────────────────────────────────────
            f_new = problem.evaluate_batch(pos[idx])
            evaluated_idx = idx[:len(f_new)]

            # ── Update personal bests ─────────────────────────────
            improved = f_new < pbest_fit[evaluated_idx]
            if np.any(improved):
                improved_idx = evaluated_idx[improved]
                pbest_fit[improved_idx] = f_new[improved]
                pbest_pos[improved_idx] = pos[improved_idx].copy()

                # ── Update global best ────────────────────────────
                local_best_pos = int(np.argmin(pbest_fit))
                if pbest_fit[local_best_pos] < gbest_fit:
                    gbest_fit = float(pbest_fit[local_best_pos])
                    gbest_pos = pbest_pos[local_best_pos].copy()

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)
