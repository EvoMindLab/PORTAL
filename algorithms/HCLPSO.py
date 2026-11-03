# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
HCLPSO — Heterogeneous Comprehensive Learning Particle Swarm Optimization
==========================================================================
Reference:
    Lynn, N., & Suganthan, P. N. (2015).
    Heterogeneous comprehensive learning particle swarm optimization
    with enhanced exploration and exploitation.
    Swarm and Evolutionary Computation, 24, 11–24.
    DOI: 10.1016/j.swevo.2015.05.002

Reference source:
    https://github.com/P-N-Suganthan/CODES
    File: HCLPSO_PS_15_35.m
    The reference code was consulted for clarification.
    Suganthan is co-author; this is authoritative.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    HCLPSO splits the swarm into two heterogeneous sub-groups with
    distinct velocity update rules:

    Group 1 (Exploration, fixed size n_g1=15):
        Uses CLPSO-style comprehensive learning — no gbest term.
        Acceleration coefficient K decreases from 3.0 to 1.5.
            vel = w*vel + K*rand*(fri_best - pos)
        Candidates for exemplar selection drawn from Group 1 only.
        This group maintains diversity and explores widely.

    Group 2 (Exploitation, size N - n_g1):
        Uses standard PSO with CLPSO-style exemplars plus gbest term.
        c1 decreases 2.5→0.5, c2 increases 0.5→2.5 over the run.
            vel = w*vel + c1*rand*(fri_best - pos) + c2*rand*(gbest - pos)
        Candidates for exemplar selection drawn from whole swarm.
        This group converges toward the global best region.

    Both groups share the same comprehensive learning exemplar
    construction and stagnation-based refresh (threshold = 5),
    identical to CLPSO. The key difference from CLPSO is the
    heterogeneous velocity update: Group 1 has no gbest attraction
    while Group 2 explicitly pulls toward gbest.

    Learning probability Pc (from MATLAB code, 0-indexed):
        j = linspace(0, 10, N)
        Pc[i] = 0.25 * (exp(j[i]) - exp(j[0])) / (exp(j[N-1]) - exp(j[0]))
    Range is [0, 0.25] — different from CLPSO's [0, 0.5].

    Boundary handling: only evaluate particles within bounds
    (no clipping). Same approach as CLPSO official MATLAB code.

Parameters (exactly matching HCLPSO_PS_15_35.m):
    pop_size  : total swarm size N   (default: 50, paper: 50)
    n_g1      : exploration group    (default: 15, fixed in paper)
    stay_thr  : stagnation threshold (default: 5)
    v_max_r   : v_max ratio          (default: 0.2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class HCLPSO(BaseAlgorithm):
    """
    Heterogeneous Comprehensive Learning PSO.
    PORTAL Python implementation informed by HCLPSO_PS_15_35.m.
    Two sub-groups: exploration (no gbest) and exploitation (with gbest).
    """

    def __init__(
        self,
        pop_size : int   = 50,    # total swarm size (paper: 50)
        n_g1     : int   = 15,    # exploration group size (paper: 15, fixed)
        stay_thr : int   = 5,     # stagnation threshold (MATLAB: 5)
        v_max_r  : float = 0.2,   # v_max = v_max_r * (ub - lb)
    ):
        super().__init__(name='HCLPSO')
        self.pop_size = pop_size
        self.n_g1     = n_g1
        self.stay_thr = stay_thr
        self.v_max_r  = v_max_r

    def run(self, problem: PortalProblem) -> dict:
        """
        Run HCLPSO on a PORTAL problem.

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
        N        = self.pop_size
        n_g1     = self.n_g1
        stay_thr = self.stay_thr
        v_max    = self.v_max_r * (ub - lb)   # shape (d,)

        # Approximate max generations (for parameter schedules)
        # MATLAB uses max_iteration; we approximate from budget
        max_iter = max_fe // N

        # Inertia weight schedule: 0.99 → 0.20
        # MATLAB: Weight(k) = 0.99 - k*0.79/max_iter
        # We compute on-the-fly using iteration counter

        # ── Learning probability Pc (per particle) ────────────────
        # MATLAB: j = 0:1/(N-1):1; j=j*10;
        #         Pc = 0.25*(exp(j)-exp(j(1)))/(exp(j(N))-exp(j(1)))
        j  = np.linspace(0, 1, N) * 10.0
        Pc = 0.0 + 0.25 * (np.exp(j) - np.exp(j[0])) / \
             (np.exp(j[-1]) - np.exp(j[0]))   # shape (N,)

        # ── Initialisation ────────────────────────────────────────
        pos = lb + np.random.rand(N, d) * (ub - lb)
        vel = -v_max + np.random.rand(N, d) * (2 * v_max)

        # Evaluate initial population
        fit = np.array([problem.evaluate(pos[i]) for i in range(N)])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()

        g_idx     = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_fit = pbest_fit[g_idx]

        # Stagnation counters
        stagnation = np.zeros(N, dtype=int)

        # ── Exemplar initialisation ───────────────────────────────
        # fri_best[i, d] = index of exemplar particle for dim d of particle i
        fri_best = np.tile(np.arange(N).reshape(-1, 1), (1, d))

        # Build initial exemplars for both groups
        for i in range(n_g1):
            fri_best[i] = self._new_exemplar(
                i, n_g1, d, Pc[i], pbest_fit, pool='g1'
            )
        for i in range(n_g1, N):
            fri_best[i] = self._new_exemplar(
                i, N, d, Pc[i], pbest_fit, pool='all'
            )

        # Build fri_best_pos from fri_best indices
        fri_best_pos = np.array([
            [pbest_pos[fri_best[i, j], j] for j in range(d)]
            for i in range(N)
        ])

        # ── Main loop ─────────────────────────────────────────────
        k = 0   # iteration counter
        while not problem.is_done():
            k += 1

            # Iteration-based parameter schedules
            # MATLAB: uses 1-indexed k from 1..max_iter
            # We use k starting at 1
            progress = min(k / max(max_iter, 1), 1.0)

            w  = 0.99 - progress * 0.79         # 0.99 → 0.20
            K  = 3.0  - progress * 1.5          # 3.0  → 1.5
            c1 = 2.5  - progress * 2.0          # 2.5  → 0.5
            c2 = 0.5  + progress * 2.0          # 0.5  → 2.5

            # ── Group 1: Exploration (no gbest) ───────────────────
            for i in range(n_g1):
                if problem.is_done():
                    break

                r = np.random.rand(d)
                vel[i] = w * vel[i] + K * r * (fri_best_pos[i] - pos[i])

                # Clamp velocity
                vel[i] = np.clip(vel[i], -v_max, v_max)
                pos[i] = pos[i] + vel[i]

                # Only evaluate if within bounds (MATLAB line 85)
                if np.all(pos[i] >= lb) and np.all(pos[i] <= ub):
                    f_new = problem.evaluate(pos[i])
                    fit[i] = f_new

                    if f_new < pbest_fit[i]:
                        pbest_fit[i] = f_new
                        pbest_pos[i] = pos[i].copy()
                        stagnation[i] = 0
                    else:
                        stagnation[i] += 1

                    if pbest_fit[i] < gbest_fit:
                        gbest_fit = pbest_fit[i]
                        gbest_pos = pbest_pos[i].copy()
                else:
                    stagnation[i] += 1

            # ── Group 2: Exploitation (with gbest) ────────────────
            for i in range(n_g1, N):
                if problem.is_done():
                    break

                r1 = np.random.rand(d)
                r2 = np.random.rand(d)
                vel[i] = (w * vel[i]
                          + c1 * r1 * (fri_best_pos[i] - pos[i])
                          + c2 * r2 * (gbest_pos        - pos[i]))

                vel[i] = np.clip(vel[i], -v_max, v_max)
                pos[i] = pos[i] + vel[i]

                if np.all(pos[i] >= lb) and np.all(pos[i] <= ub):
                    f_new = problem.evaluate(pos[i])
                    fit[i] = f_new

                    if f_new < pbest_fit[i]:
                        pbest_fit[i] = f_new
                        pbest_pos[i] = pos[i].copy()
                        stagnation[i] = 0
                    else:
                        stagnation[i] += 1

                    if pbest_fit[i] < gbest_fit:
                        gbest_fit = pbest_fit[i]
                        gbest_pos = pbest_pos[i].copy()
                else:
                    stagnation[i] += 1

            # ── Refresh exemplars for stagnated particles ─────────
            for i in range(n_g1):
                if stagnation[i] > stay_thr:
                    fri_best[i] = self._new_exemplar(
                        i, n_g1, d, Pc[i], pbest_fit, pool='g1'
                    )
                    fri_best_pos[i] = [
                        pbest_pos[fri_best[i, j], j] for j in range(d)
                    ]
                    stagnation[i] = 0

            for i in range(n_g1, N):
                if stagnation[i] > stay_thr:
                    fri_best[i] = self._new_exemplar(
                        i, N, d, Pc[i], pbest_fit, pool='all'
                    )
                    fri_best_pos[i] = [
                        pbest_pos[fri_best[i, j], j] for j in range(d)
                    ]
                    stagnation[i] = 0

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)

    # ── Helper ────────────────────────────────────────────────────

    def _new_exemplar(
        self,
        i        : int,
        pool_size: int,
        d        : int,
        pc_i     : float,
        pbest_fit: np.ndarray,
        pool     : str,
    ) -> np.ndarray:
        """
        Build exemplar row for particle i.
        Construct exemplars using the heterogeneous comprehensive-learning policy.

        pool='g1'  → candidates from indices 0..n_g1-1 (group 1 only)
        pool='all' → candidates from indices 0..N-1 (whole swarm)

        MATLAB uses 1-indexed ceil(pool_size*rand) for candidate selection.
        Python equivalent: randint(0, pool_size).
        """
        fri = np.full(d, i, dtype=int)   # default: self

        # Tournament selection per dimension
        f1 = np.random.randint(0, pool_size, size=d)
        f2 = np.random.randint(0, pool_size, size=d)
        friend = np.where(pbest_fit[f1] < pbest_fit[f2], f1, f2)

        # toss: 0 with probability Pc[i] (use friend), 1 otherwise (keep self)
        # MATLAB: toss = ceil(rand - Pc[i])
        #         ceil(rand - Pc) = 0 if rand < Pc, else 1
        toss = (np.random.rand(d) >= pc_i).astype(int)

        # Force at least one dimension to use a friend
        # MATLAB: if toss==ones(1,D): toss(randperm(D)(1)) = 0
        if np.all(toss == 1):
            toss[np.random.randint(d)] = 0

        fri = (1 - toss) * friend + toss * fri
        return fri
