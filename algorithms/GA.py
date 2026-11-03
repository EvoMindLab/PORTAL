# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
SGA — Real-Coded Genetic Algorithm with SBX Crossover and Polynomial Mutation
==============================================================================
Reference:
    Deb, K., & Agrawal, R. B. (1995).
    Simulated binary crossover for continuous search space.
    Complex Systems, 9(2), 115–148.

    Deb, K., & Goyal, M. (1996).
    A combined genetic adaptive search (GeneAS) for engineering design.
    Computer Science and Informatics, 26(4), 30–45.

    This is the standard real-coded GA used as a baseline in CEC competitions.
    Parameters follow the canonical settings from the CEC literature.

Key mechanism:
    Real-coded GA operating directly in continuous search space.

    SELECTION: Binary tournament (k=2), with elitism (best always survives).

    CROSSOVER: Simulated Binary Crossover (SBX, Deb & Agrawal 1995)
        Mimics single-point crossover behaviour in real-coded space.
        For each dimension j with probability p_c:
            u = rand(0,1)
            if u <= 0.5:
                beta = (2u)^(1/(eta_c+1))
            else:
                beta = (1/(2(1-u)))^(1/(eta_c+1))
            child1[j] = 0.5*((1+beta)*parent1[j] + (1-beta)*parent2[j])
            child2[j] = 0.5*((1-beta)*parent1[j] + (1+beta)*parent2[j])
        eta_c=20: higher = closer to parents (more exploitative)

    MUTATION: Polynomial Mutation (Deb & Goyal 1996)
        Per-dimension with probability p_m = 1/D:
            u = rand(0,1)
            if u < 0.5:
                delta = (2u)^(1/(eta_m+1)) - 1
            else:
                delta = 1 - (2(1-u))^(1/(eta_m+1))
            x[j] = x[j] + delta*(ub[j]-lb[j])
        eta_m=20: higher = smaller perturbations

    BOUNDARY: clip to [lb, ub] after crossover and mutation.

Parameters (canonical CEC baseline settings):
    pop_size  : population size   (default: 100)
    eta_c     : SBX distribution index    (default: 20)
    eta_m     : polynomial mutation index (default: 20)
    p_c       : crossover probability     (default: 0.9)
    p_m       : mutation probability per gene (default: 1/D)
    elitism   : keep best individual      (default: True)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class SGA(BaseAlgorithm):
    """
    Real-coded Genetic Algorithm with SBX crossover and polynomial mutation.
    Standard CEC baseline configuration.
    """

    def __init__(
        self,
        pop_size : int   = 100,
        eta_c    : float = 20.0,   # SBX distribution index
        eta_m    : float = 20.0,   # polynomial mutation index
        p_c      : float = 0.9,    # crossover probability
        p_m      : float = None,   # mutation prob per gene (None → 1/D)
        elitism  : bool  = True,
    ):
        super().__init__(name='SGA')
        self.pop_size = pop_size
        self.eta_c    = eta_c
        self.eta_m    = eta_m
        self.p_c      = p_c
        self.p_m      = p_m
        self.elitism  = elitism

    def run(self, problem: PortalProblem) -> dict:
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        range_ = ub - lb

        p_m = self.p_m if self.p_m is not None else 1.0 / d

        # ── Initialisation ────────────────────────────────────────
        pop     = lb + np.random.rand(self.pop_size, d) * range_
        fitness = np.array([problem.evaluate(pop[i])
                            for i in range(self.pop_size)])

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            N = len(pop)

            # ── Binary tournament selection ───────────────────────
            def tournament(n):
                idx = np.empty(n, dtype=int)
                for k in range(n):
                    a, b = np.random.randint(0, N, 2)
                    idx[k] = a if fitness[a] <= fitness[b] else b
                return idx

            # Build offspring population
            offspring     = np.empty_like(pop)
            offspring_fit = np.empty(N)

            # Process pairs
            i = 0
            while i < N:
                # Select two parents via tournament
                p1_idx = tournament(1)[0]
                p2_idx = tournament(1)[0]
                p1 = pop[p1_idx].copy()
                p2 = pop[p2_idx].copy()

                # ── SBX Crossover ─────────────────────────────────
                c1, c2 = p1.copy(), p2.copy()
                if np.random.rand() < self.p_c:
                    for j in range(d):
                        if np.random.rand() <= 0.5:   # per-dim 50%
                            if abs(p1[j] - p2[j]) > 1e-14:
                                u = np.random.rand()
                                if u <= 0.5:
                                    beta = (2.0 * u) ** (1.0 / (self.eta_c + 1))
                                else:
                                    beta = (1.0 / (2.0 * (1.0 - u))) ** \
                                           (1.0 / (self.eta_c + 1))
                                c1[j] = 0.5 * ((1 + beta) * p1[j] +
                                               (1 - beta) * p2[j])
                                c2[j] = 0.5 * ((1 - beta) * p1[j] +
                                               (1 + beta) * p2[j])

                # ── Polynomial Mutation ───────────────────────────
                for child in (c1, c2):
                    for j in range(d):
                        if np.random.rand() < p_m:
                            u = np.random.rand()
                            if u < 0.5:
                                delta = (2.0 * u) ** \
                                        (1.0 / (self.eta_m + 1)) - 1.0
                            else:
                                delta = 1.0 - (2.0 * (1.0 - u)) ** \
                                        (1.0 / (self.eta_m + 1))
                            child[j] += delta * range_[j]

                # ── Clip to bounds ────────────────────────────────
                c1 = np.clip(c1, lb, ub)
                c2 = np.clip(c2, lb, ub)

                # Store offspring
                if i < N:
                    offspring[i] = c1
                if i + 1 < N:
                    offspring[i + 1] = c2
                i += 2

            # ── Evaluate offspring ────────────────────────────────
            for k in range(N):
                if problem.is_done():
                    break
                offspring_fit[k] = problem.evaluate(offspring[k])

            # ── Elitism + replacement ─────────────────────────────
            if self.elitism:
                # Find best in current population
                best_idx = int(np.argmin(fitness))
                best_x   = pop[best_idx].copy()
                best_f   = fitness[best_idx]

                # Replace population with offspring
                pop     = offspring.copy()
                fitness = offspring_fit.copy()

                # If best not preserved, replace worst offspring with it
                worst_idx = int(np.argmax(fitness))
                if best_f < fitness[worst_idx]:
                    pop[worst_idx]     = best_x
                    fitness[worst_idx] = best_f
            else:
                pop     = offspring.copy()
                fitness = offspring_fit.copy()

        return self._make_result(problem)
