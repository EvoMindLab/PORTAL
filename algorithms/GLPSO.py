# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
GL-PSO — Genetic Learning Particle Swarm Optimization
=======================================================
Reference:
    Gong, Y.-J., Li, J.-J., Zhou, Y., Li, Y., Chung, H. S.-H.,
    Shi, Y.-H., & Zhang, J. (2016).
    Genetic learning particle swarm optimization.
    IEEE Transactions on Cybernetics, 46(10), 2277–2290.
    DOI: 10.1109/TCYB.2015.2475174

Reference source:
    https://github.com/YuejiaoGong/genetic_learning_PSO
    File: glpso.cpp  (GLPSO_1.0, March 6th, 2013)
    Released by Yue-Jiao Gong (first author).
    The reference code was consulted for clarification.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    GL-PSO replaces the standard pbest/gbest velocity update with a
    two-layer genetic-learning paradigm:

    Layer 1 — Exemplar Evolution (GA layer):
        Each particle maintains an explicit exemplar vector E_i.
        Every generation, a new candidate exemplar is bred via:
        (a) Crossover (per dimension d):
            Pick random particle n from swarm.
            If pbest[n] is BETTER than pbest[i], OR i is the global best:
                newE[d] = pbest[n][d]           (random copy from better)
            Else:
                r ~ Uniform(0,1)
                newE[d] = r*pbest[i][d] + (1-r)*pbest[bestpar][d]
                                                (uniform crossover with gbest)
        (b) Mutation (per dimension d):
            If rand < pm: newE[d] = rand(lb, ub)  (random reinit)
        (c) Greedy selection:
            Evaluate newE. If f(newE) < f(E_i): E_i ← newE; stop=0
            Else: stop++
        (d) Tournament rescue (when stop > sg):
            Pick NSEL random exemplars; best one replaces E_i.
            This uses the exemplar pool — no new evaluation needed.

    Layer 2 — PSO Update (using exemplar, not pbest/gbest):
        vel = w * vel + c1 * rand * (E_i - pos)
        Only ONE acceleration coefficient (no c2 or gbest term).
        Boundary: absorbing reflection with halved velocity.

    The exemplar evolves to be a high-quality guide that blends
    information from multiple pbests via genetic operators.

Parameters (exactly matching glpso.cpp):
    pop_size : swarm size N        (default: 50)
    w        : inertia weight      (default: 0.7298)
    c1       : acceleration coeff  (default: 1.49618)
    sg       : stagnation gap      (default: 7)
    pm       : mutation prob/dim   (default: 0.01)
    nsel     : tournament size     (default: 10)
    v_max_r  : v_max ratio         (default: 0.2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class GL_PSO(BaseAlgorithm):
    """
    Genetic Learning Particle Swarm Optimization.
    PORTAL Python implementation informed by the glpso.cpp reference.
    Exemplars evolved by GA crossover+mutation+selection.
    No gbest term in velocity update.
    """

    def __init__(
        self,
        pop_size : int   = 50,       # swarm size (C++: PARSIZE=50)
        w        : float = 0.7298,   # inertia weight
        c1       : float = 1.49618,  # acceleration coefficient
        sg       : int   = 7,        # exemplar stagnation threshold
        pm       : float = 0.01,     # mutation probability per dimension
        nsel     : int   = 10,       # tournament size (C++: NSEL=10)
        v_max_r  : float = 0.2,      # v_max = v_max_r * (ub - lb)
    ):
        super().__init__(name='GL-PSO')
        self.pop_size = pop_size
        self.w        = w
        self.c1       = c1
        self.sg       = sg
        self.pm       = pm
        self.nsel     = nsel
        self.v_max_r  = v_max_r

    def run(self, problem: PortalProblem) -> dict:
        """
        Run GL-PSO on a PORTAL problem.

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
        N    = self.pop_size
        w    = self.w
        c1   = self.c1
        sg   = self.sg
        pm   = self.pm
        nsel = self.nsel
        vmax = self.v_max_r * (ub - lb)   # shape (d,)

        # ── Initialisation ────────────────────────────────────────
        pos = lb + np.random.rand(N, d) * (ub - lb)

        # Velocity init: rand(lb - pos, ub - pos) per dimension
        # C++: randval(lbound - pos[j], ubound - pos[j])
        vel = np.array([
            lb - pos[i] + np.random.rand(d) * (ub - lb)
            for i in range(N)
        ])

        # Evaluate initial population
        fit = np.array([problem.evaluate(pos[i]) for i in range(N)])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()

        best_par  = int(np.argmin(pbest_fit))

        # Exemplar initialisation
        # C++: exemplar[i].f = worstvalue (1e300), then exemplar_update(i)
        ex_pos  = np.full((N, d), np.inf)    # exemplar positions
        ex_fit  = np.full(N, 1e300)          # exemplar fitness
        ex_stop = np.zeros(N, dtype=int)     # exemplar stagnation counters

        # Run initial exemplar update for all particles
        for i in range(N):
            if problem.is_done():
                break
            ex_pos[i], ex_fit[i], ex_stop[i] = self._exemplar_update(
                i, N, d, lb, ub, pm, sg, nsel,
                pbest_pos, pbest_fit, best_par,
                ex_pos, ex_fit, ex_stop,
                problem
            )

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            for i in range(N):
                if problem.is_done():
                    break

                # ── Update exemplar (GA layer) ────────────────────
                ex_pos[i], ex_fit[i], ex_stop[i] = self._exemplar_update(
                    i, N, d, lb, ub, pm, sg, nsel,
                    pbest_pos, pbest_fit, best_par,
                    ex_pos, ex_fit, ex_stop,
                    problem
                )

                if problem.is_done():
                    break

                # ── Velocity update (PSO layer) ───────────────────
                # C++: vel[j] = w*vel[j] + c1*rand*(exemplar[i].x[j] - pos[j])
                rnd    = np.random.rand(d)
                vel[i] = w * vel[i] + c1 * rnd * (ex_pos[i] - pos[i])

                # Clamp velocity
                vel[i] = np.clip(vel[i], -vmax, vmax)

                # ── Position update ───────────────────────────────
                pos[i] = pos[i] + vel[i]

                # Boundary: absorbing reflection with halved velocity
                # C++: if pos < lb: pos = lb; vel *= -0.5
                #       if pos > ub: pos = ub; vel *= -0.5
                below = pos[i] < lb
                above = pos[i] > ub
                pos[i] = np.where(below, lb, np.where(above, ub, pos[i]))
                vel[i] = np.where(below | above, vel[i] * -0.5, vel[i])

                # ── Evaluate ──────────────────────────────────────
                f_new = problem.evaluate(pos[i])

                # ── Update pbest ──────────────────────────────────
                if f_new < pbest_fit[i]:
                    pbest_fit[i] = f_new
                    pbest_pos[i] = pos[i].copy()

                    # Update best_par (global best index)
                    if f_new < pbest_fit[best_par]:
                        best_par = i

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)

    # ── Exemplar update (GA operators) ───────────────────────────

    def _exemplar_update(
        self,
        i, N, d, lb, ub, pm, sg, nsel,
        pbest_pos, pbest_fit, best_par,
        ex_pos, ex_fit, ex_stop,
        problem
    ):
        """
        Full exemplar update for particle i.
        Translates exemplar_crossover + exemplar_mutation +
        exemplar_selection + exemplar_tour_selection from C++.
        """
        # ── Crossover ─────────────────────────────────────────────
        new_ex = np.empty(d)
        for j in range(d):
            n = np.random.randint(N)
            if pbest_fit[n] < pbest_fit[i] or i == best_par:
                # Random copy from a better particle's pbest
                new_ex[j] = pbest_pos[n, j]
            else:
                # Uniform crossover between pbest[i] and pbest[bestpar]
                r = np.random.rand()
                new_ex[j] = (r * pbest_pos[i, j]
                             + (1 - r) * pbest_pos[best_par, j])

        # ── Mutation ──────────────────────────────────────────────
        for j in range(d):
            if np.random.rand() < pm:
                new_ex[j] = lb[j] + np.random.rand() * (ub[j] - lb[j])

        # ── Greedy selection ──────────────────────────────────────
        f_new_ex = problem.evaluate(new_ex)

        if f_new_ex < ex_fit[i]:
            ex_pos[i]  = new_ex
            ex_fit[i]  = f_new_ex
            ex_stop[i] = 0
        else:
            ex_stop[i] += 1

        # ── Tournament rescue (when stagnated > sg) ───────────────
        # C++: pick NSEL random exemplars; best replaces current
        # No evaluation — reuses existing exemplar fitnesses
        if ex_stop[i] > sg:
            ex_stop[i] = 0
            candidates = np.random.randint(0, N, size=nsel)
            best_cand  = candidates[np.argmin(ex_fit[candidates])]
            ex_pos[i]  = ex_pos[best_cand].copy()
            ex_fit[i]  = ex_fit[best_cand]

        return ex_pos[i], ex_fit[i], ex_stop[i]
