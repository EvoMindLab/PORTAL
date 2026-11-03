# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
QPSO — Quantum-Behaved Particle Swarm Optimization
====================================================
Reference:
    Sun, J., Feng, B., & Xu, W. (2004).
    Particle swarm optimization with particles having quantum behavior.
    Proceedings of the 2004 IEEE Congress on Evolutionary Computation,
    Vol. 1, pp. 325–331.
    DOI: 10.1109/CEC.2004.1330875

    Sun, J., Xu, W., & Feng, B. (2004).
    A global search strategy of quantum-behaved particle swarm optimization.
    IEEE Conference on Cybernetics and Intelligent Systems, pp. 111–116.
    DOI: 10.1109/ICCIS.2004.1460396

Source:
    Implemented directly from paper equations.
    No official author code is publicly available.

Key mechanism:
    QPSO completely removes the velocity vector used in standard PSO.
    Each particle is modelled as a quantum particle in a delta potential
    well centred on a local attractor p, which is a random convex
    combination of the particle's pbest and the global best (mbest):

        phi  = rand()
        p    = phi * pbest + (1 - phi) * gbest        (local attractor)
        mbest = mean of all pbest positions            (mean best)

    The new position is sampled from the quantum wave function
    (double-exponential / Laplace distribution):

        u    = rand()
        x    = p ± beta * |mbest - x| * ln(1/u)

    where the sign is chosen randomly (+/-) with equal probability.
    The contraction-expansion coefficient beta decreases linearly
    from beta_max to beta_min over the run, controlling convergence.

Parameters (defaults follow Sun et al. 2004):
    pop_size : swarm size                   (default: min(10*d, 50))
    beta_max : initial contraction-expansion (default: 1.0)
    beta_min : final contraction-expansion   (default: 0.5)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class QPSO(BaseAlgorithm):
    """
    Quantum-Behaved Particle Swarm Optimization.
    No velocity vector — positions sampled from quantum wave function.
    """

    def __init__(
        self,
        pop_size : int   = None,   # None = auto: min(10*d, 50)
        beta_max : float = 1.0,    # initial contraction-expansion coefficient
        beta_min : float = 0.5,    # final contraction-expansion coefficient
    ):
        super().__init__(name='QPSO')
        self.pop_size = pop_size
        self.beta_max = beta_max
        self.beta_min = beta_min

    def run(self, problem: PortalProblem) -> dict:
        """
        Run QPSO on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb        # shape (d,)
        ub     = problem.ub        # shape (d,)
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        N        = self.pop_size if self.pop_size is not None else min(10 * d, 50)
        beta_max = self.beta_max
        beta_min = self.beta_min

        # ── Initialisation ────────────────────────────────────────
        # Positions: uniform random in [lb, ub]
        # No velocity in QPSO
        pos = lb + np.random.rand(N, d) * (ub - lb)

        # Evaluate initial population
        fit = np.array([problem.evaluate(pos[i]) for i in range(N)])

        # Personal bests
        pbest_pos = pos.copy()    # shape (N, d)
        pbest_fit = fit.copy()    # shape (N,)

        # Global best
        g_idx     = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[g_idx].copy()   # shape (d,)

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            # Linearly decreasing beta
            progress = problem.fe_count / max_fe
            beta = beta_max - (beta_max - beta_min) * progress

            # Mean best position: mean of all personal bests
            # mbest = (1/N) * sum of all pbest positions
            mbest = np.mean(pbest_pos, axis=0)   # shape (d,)

            for i in range(N):
                if problem.is_done():
                    break

                # ── Local attractor ───────────────────────────────
                # p = phi * pbest_i + (1 - phi) * gbest
                # phi drawn independently per dimension
                phi = np.random.rand(d)
                p   = phi * pbest_pos[i] + (1 - phi) * gbest_pos

                # ── Quantum position update ───────────────────────
                # u drawn independently per dimension
                u = np.random.rand(d)

                # Random sign: +1 or -1 per dimension
                sign = np.where(np.random.rand(d) < 0.5, 1.0, -1.0)

                # New position from quantum wave function:
                # x = p + sign * beta * |mbest - x| * ln(1/u)
                new_pos = p + sign * beta * np.abs(mbest - pos[i]) * np.log(1.0 / (u + 1e-300))

                # Clamp to search bounds
                new_pos = np.clip(new_pos, lb, ub)
                pos[i]  = new_pos

                # ── Evaluate ──────────────────────────────────────
                f_new = problem.evaluate(pos[i])

                # ── Update personal best ──────────────────────────
                if f_new < pbest_fit[i]:
                    pbest_fit[i] = f_new
                    pbest_pos[i] = pos[i].copy()

                    # ── Update global best ────────────────────────
                    if f_new < problem.best_f:
                        gbest_pos = pos[i].copy()

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)
