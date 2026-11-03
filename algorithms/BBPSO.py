# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
BBPSO — Bare Bones Particle Swarm Optimization
================================================
Reference:
    Kennedy, J. (2003).
    Bare bones particle swarms.
    Proceedings of the 2003 IEEE Swarm Intelligence Symposium (SIS'03),
    pp. 80–87.
    DOI: 10.1109/SIS.2003.1202251

Source:
    Implemented directly from paper equations.
    No official author code is publicly available.

Key mechanism:
    BBPSO strips PSO down to its absolute minimum — no velocity vector,
    no inertia weight, no acceleration coefficients, no parameters.

    Each particle samples its new position dimension-by-dimension from
    a Gaussian distribution:

        mu    = (pbest + gbest) / 2       (midpoint of personal and global best)
        sigma = |pbest - gbest|           (spread between personal and global best)
        x_new = Gaussian(mu, sigma)       (sample new position)

Implementation notes:
    Variant: This implements the g=gbest (global best) variant.
        The paper presents and compares multiple variants including
        lbest (von Neumann topology) and random-neighbour variants.
        The gbest variant is used here as it is the most commonly
        cited baseline in the literature.

    Collapsed state (sigma=0): When pbest == gbest, sigma becomes 0
        and the Gaussian collapses to a point. The paper treats this
        as natural convergence behaviour but in practice it causes
        complete stagnation. A small perturbation (1% of search range)
        is applied to escape. This is not in the paper but is a
        standard practical addition required for any bounded problem.

    Boundary handling: The paper does not specify boundary handling.
        np.clip is applied since PORTAL requires solutions within
        [bounds_min, bounds_max]. This is a necessary engineering
        addition not described in the paper.

    Population size: The paper uses N=20 arranged in a 4x5 von Neumann
        grid (relevant for topology variants). For the gbest variant
        used here, auto-sizing (min(10*d, 50)) is appropriate.

Parameters (defaults follow Kennedy 2003):
    pop_size : swarm size  (default: min(10*d, 50))
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class BBPSO(BaseAlgorithm):
    """
    Bare Bones Particle Swarm Optimization (gbest variant).
    Parameter-free — samples new positions from Gaussian(mu, sigma).
    """

    def __init__(
        self,
        pop_size : int = None,   # None = auto: min(10*d, 50)
    ):
        super().__init__(name='BBPSO')
        self.pop_size = pop_size

    def run(self, problem: PortalProblem) -> dict:
        """
        Run BBPSO on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d  = problem.d
        lb = problem.lb        # shape (d,)
        ub = problem.ub        # shape (d,)

        # ── Algorithm parameters ──────────────────────────────────
        N = self.pop_size if self.pop_size is not None else min(10 * d, 50)

        # ── Initialisation ────────────────────────────────────────
        # Positions: uniform random in [lb, ub]
        # No velocity in BBPSO
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

            for i in range(N):
                if problem.is_done():
                    break

                # ── Gaussian sampling ─────────────────────────────
                # mu    = midpoint of pbest and gbest (paper Eq. 1)
                # sigma = absolute difference between pbest and gbest
                mu    = 0.5 * (pbest_pos[i] + gbest_pos)
                sigma = np.abs(pbest_pos[i] - gbest_pos)

                # Collapsed state: sigma=0 when pbest==gbest.
                # Paper treats this as natural convergence but it
                # causes complete stagnation on bounded problems.
                # Apply small perturbation (1% of range) to escape.
                # Not in paper — practical addition for bounded search.
                sigma = np.where(sigma < 1e-10, 0.01 * (ub - lb), sigma)

                # Sample new position from Gaussian
                new_pos = np.random.normal(mu, sigma)

                # Clamp to search bounds.
                # Not in paper — required since Gaussian has infinite
                # tails and PORTAL requires bounded solutions.
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
