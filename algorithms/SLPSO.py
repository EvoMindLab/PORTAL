# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
SL-PSO — Social Learning Particle Swarm Optimization
======================================================
Reference:
    Cheng, R., & Jin, Y. (2015).
    A social learning particle swarm optimization algorithm for
    scalable optimization.
    Information Sciences, 291, 43–60.
    DOI: 10.1016/j.ins.2014.08.039

Source:
    Implemented directly from paper Algorithm 1 and Equations 1–8.
    The third-party Python reimplementation (github.com/vsg-root/SLPSO)
    was reviewed but NOT used — its README explicitly states it does
    not match the paper results, and it contains multiple structural
    errors.

Key mechanism:
    SL-PSO removes both pbest and gbest entirely. Each particle (the
    "imitator") learns from a randomly chosen "demonstrator" that has
    better fitness — inspired by social/observational learning.

    The swarm is sorted worst-first (index 0 = worst, index m-1 = best).
    For particle i (0-indexed):
      - A scalar p_i ~ Uniform(0,1) is drawn
      - If p_i <= PL(i), the particle learns this generation
      - For each dimension d independently:
            k = random index from {i+1, ..., m-1}  (a better particle)
            delta[i][d] = r1*delta[i][d]
                        + r2*(pos[k][d] - pos[i][d])
                        + r3*epsilon*(mean[d] - pos[i][d])
            pos[i][d] += delta[i][d]
        where r1,r2,r3 ~ Uniform(0,1) drawn per dimension

    Learning probability (paper Eq.8):
        PL(i) = (i / m)^alpha
        In 0-indexed ascending sort (0=worst, m-1=best):
            PL[0]   = 0     (worst particle — rarely learns... wait)

    Actually paper uses 1-indexed with i=1=worst, i=m=best:
        PL(i) = (1 - i/m)^alpha   [paper Eq.8]
        i=1 (worst): PL ~ (1-1/m)^alpha ~ 1  → learns most
        i=m (best):  PL = 0                   → never learns
    In 0-indexed ascending (0=worst, m-1=best):
        PL[k] = (1 - (k+1)/m)^alpha

    A mean-field term (r3*epsilon*(mean-pos)) prevents collapse:
        epsilon = beta * D / M   (paper Eq.4)

    Swarm size augmented: m = M + floor(D/10)  (paper Section III-A)

Parameters (defaults follow Cheng & Jin 2015):
    pop_size : base swarm size M    (default: 100, paper uses 100)
    beta     : mean-field strength  (default: 0.01)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class SL_PSO(BaseAlgorithm):
    """
    Social Learning Particle Swarm Optimization.
    No pbest, no gbest — particles learn from better-ranked demonstrators.
    Each dimension independently selects its own demonstrator.
    """

    def __init__(
        self,
        pop_size : int   = 100,   # base swarm size M (paper: 100)
        beta     : float = 0.01,  # mean-field coefficient
    ):
        super().__init__(name='SL-PSO')
        self.pop_size = pop_size
        self.beta     = beta

    def run(self, problem: PortalProblem) -> dict:
        """
        Run SL-PSO on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d   = problem.d
        lb  = problem.lb        # shape (d,)
        ub  = problem.ub        # shape (d,)

        # ── Algorithm parameters ──────────────────────────────────
        M = self.pop_size

        # Augmented swarm size (paper Section III-A)
        m = M + int(np.floor(d / 10))

        # Mean-field coefficient (paper Eq.4)
        epsilon = self.beta * d / M

        # Learning probability exponent (paper Eq.8 footnote)
        # alpha = log(ceil(D/M)) / log(0.5)
        ceil_val = int(np.ceil(d / M))
        if ceil_val <= 1:
            alpha = 0.0
        else:
            alpha = np.log(ceil_val) / np.log(0.5)

        # Learning probabilities PL for 0-indexed worst-first sort
        # Paper Eq.8 (1-indexed, i=1=worst, i=m=best):
        #   PL(i) = (1 - i/m)^alpha
        # 0-indexed (k=0=worst, k=m-1=best):
        #   PL[k] = (1 - (k+1)/m)^alpha
        # → PL[0] ~ 1 (worst, learns most)
        # → PL[m-1] = 0 (best, never learns)
        PL = (1.0 - (np.arange(m) + 1) / m) ** alpha   # shape (m,)

        # ── Initialisation ────────────────────────────────────────
        pos   = lb + np.random.rand(m, d) * (ub - lb)
        delta = np.zeros((m, d))   # previous update memory

        # Evaluate initial population
        fit = np.array([problem.evaluate(pos[i]) for i in range(m)])

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            # ── Sort worst-first (paper Algorithm 1, line 4) ──────
            # index 0 = worst, index m-1 = best
            idx   = np.argsort(fit)[::-1]
            pos   = pos[idx]
            fit   = fit[idx]
            delta = delta[idx]

            # Mean position across all particles (paper Eq.3)
            mean_pos = np.mean(pos, axis=0)   # shape (d,)

            # ── Update each particle i from 0 (worst) to m-2 ─────
            # Best particle (i = m-1) never learns — PL[m-1] = 0
            for i in range(m - 1):
                if problem.is_done():
                    break

                # ── Scalar learning gate (paper Algorithm 1, Eq.8) ─
                # Draw ONE scalar p_i: if p_i > PL[i], skip particle
                p_i = np.random.rand()
                if p_i > PL[i]:
                    continue

                # ── Per-dimension update (paper Algorithm 1 inner loop)
                # k drawn INDEPENDENTLY per dimension (paper line 24:
                # "k is generated independently for each element j")
                r1 = np.random.rand(d)
                r2 = np.random.rand(d)
                r3 = np.random.rand(d)

                # For each dimension, pick a random demonstrator
                # from {i+1, ..., m-1} (all better-ranked particles)
                k_vec = np.random.randint(i + 1, m, size=d)   # shape (d,)

                # Demonstrator positions: pos[k_vec[j], j] for each dim j
                demo_pos = pos[k_vec, np.arange(d)]   # shape (d,)

                # Position update (paper Eq.1)
                new_delta = (r1 * delta[i]
                           + r2 * (demo_pos - pos[i])
                           + r3 * epsilon * (mean_pos - pos[i]))

                pos[i]   = pos[i] + new_delta
                delta[i] = new_delta

                # Clamp to bounds
                # Paper specifies no boundary handling — this is a
                # necessary addition for bounded PORTAL problems.
                pos[i] = np.clip(pos[i], lb, ub)

                # ── Evaluate ──────────────────────────────────────
                f_new = problem.evaluate(pos[i])
                fit[i] = f_new

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)
