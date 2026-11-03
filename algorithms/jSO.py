# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
jSO — Single Objective Real-Parameter Optimization Algorithm
=============================================================
Reference:
    Brest, J., Sepesy Maučec, M., & Bošković, B. (2017).
    Single objective real-parameter optimization: Algorithm jSO.
    Proceedings of the 2017 IEEE Congress on Evolutionary Computation
    (CEC 2017), pp. 1311–1318.
    DOI: 10.1109/CEC.2017.7969456

Reference source:
    jSO-SOURCE-RESULTS-29May2017.tar.gz — lshade.cc, de.h, main.cc
    The reference code was consulted for clarification.
    Written by Janez Brest, Mirjam Sepesy Maučec, Borko Bošković.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    jSO = improved i-L-SHADE = improved L-SHADE. All L-SHADE mechanisms
    are retained with these additions and modifications:

    1. WEIGHTED MUTATION (the core jSO innovation):
       Different scaling factors for the two difference vectors:
           jF = F * 0.7   if nfes < 0.2 * maxFES
           jF = F * 0.8   if nfes < 0.4 * maxFES
           jF = F * 1.2   otherwise
           child = target + jF*(pbest-target) + F*(r1-r2)
       Early in the run, pbest attraction is weakened (0.7x) to
       encourage exploration; late in the run it is strengthened (1.2x).

    2. TIME-VARYING CR LOWER BOUND:
       if nfes < 0.25*maxFES: CR = max(CR, 0.7)
       if nfes < 0.50*maxFES: CR = max(CR, 0.6)
       Enforces high CR (diverse crossover) early in the search.

    3. TIME-VARYING F UPPER BOUND:
       if nfes < 0.60*maxFES: F = min(F, 0.7)
       Limits mutation strength in early/middle phases.

    4. p_best AVOIDANCE:
       When nfes < 0.5*maxFES: resample p_best until p_best != target.
       Prevents a particle from using itself as pbest in the first half.

    5. TIME-VARYING p_best RATE:
       p_rate = p_init * (1 - 0.5 * nfes / maxFES)
       Narrows the pbest pool as search matures (0.25 → 0.125).

    6. MEMORY UPDATE — Janez averaging:
       memory_sf[k] = (weighted_lehmer(S_F) + old_sf) / 2
       memory_cr[k] = (weighted_lehmer(S_CR) + old_cr) / 2
       The new estimate is averaged with the previous slot value.

    7. LAST MEMORY SLOT forced to mu_sf=0.9, mu_cr=0.9 when selected.

    8. MEMORY INITIALISATION: memory_sf=0.3, memory_cr=0.8
       (L-SHADE uses 0.5 for both)

    9. SELECTION: non-strict (<=), ties go to child.
       (L-SHADE uses strict <)

    10. ARCHIVE: simple counter-based (no deduplication).
        arc_size = round(arc_rate * NP), dynamic with population reduction.

Parameters (exactly matching main.cc and lshade.cc):
    p_init       : initial pbest fraction     (default: 0.25)
    arc_rate     : archive size multiplier    (default: 1.0)
    H            : memory size               (default: 5)
    min_pop      : minimum population        (default: 4)
    init_pop_fn  : initial pop = fn(D)       (default: sqrt(D)*ln(D)*25)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class JSO(BaseAlgorithm):
    """
    jSO: Improved L-SHADE with weighted mutation and time-varying parameters.
    PORTAL Python implementation informed by lshade.cc (jSO, CEC 2017).
    CEC 2017 winner.
    """

    def __init__(
        self,
        p_init    : float = 0.25,   # initial pbest fraction (main.cc: 0.25)
        arc_rate  : float = 1.0,    # archive multiplier (main.cc: 1.0)
        H         : int   = 5,      # memory size (main.cc: 5)
        min_pop   : int   = 4,      # minimum population
    ):
        super().__init__(name='jSO')
        self.p_init   = p_init
        self.arc_rate = arc_rate
        self.H        = H
        self.min_pop  = min_pop

    def run(self, problem: PortalProblem) -> dict:
        """
        Run jSO on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        # Initial population: round(sqrt(D)*ln(D)*25)  (main.cc)
        pop_size     = max(self.min_pop,
                           int(round(np.sqrt(d) * np.log(d) * 25)))
        max_pop_size = pop_size
        min_pop      = self.min_pop
        H            = self.H
        arc_rate     = self.arc_rate
        p_init       = self.p_init

        if max_fe < pop_size:
            raise ValueError(
                f"jSO needs at least {pop_size} evaluations for its initial "
                f"population, but the supplied budget is {max_fe}."
            )

        # Archive
        arc_size      = int(round(arc_rate * pop_size))
        arc_ind_count = 0   # counter (C++ style: circular overwrite)

        # ── Historical memory ─────────────────────────────────────
        # jSO initialises to 0.3 / 0.8 (NOT 0.5/0.5 like L-SHADE)
        memory_sf  = np.full(H, 0.3)
        memory_cr  = np.full(H, 0.8)
        memory_pos = 0

        # ── Initialisation ────────────────────────────────────────
        pop     = lb + np.random.rand(pop_size, d) * (ub - lb)
        fitness = np.array([problem.evaluate(pop[i]) for i in range(pop_size)])
        nfes    = pop_size

        # Archive (pre-allocated as in C++, filled with counter)
        archive = np.zeros((arc_size, d))

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            sorted_index = np.argsort(fitness)   # best first

            pop_sf = np.zeros(pop_size)
            pop_cr = np.zeros(pop_size)

            # Time-varying p_best rate (jSO)
            p_rate = p_init * (1.0 - 0.5 * nfes / max_fe)
            p_num  = max(2, int(round(pop_size * p_rate)))

            success_sf_list = []   # (F, CR, dif_fitness) triples
            for target in range(pop_size):
                if problem.is_done():
                    break

                # ── Select memory slot ────────────────────────────
                r = np.random.randint(H)

                # Last slot forced to 0.9/0.9 (jSO)
                if r == H - 1:
                    mu_sf = 0.9
                    mu_cr = 0.9
                else:
                    mu_sf = memory_sf[r]
                    mu_cr = memory_cr[r]

                # ── Generate CR ───────────────────────────────────
                if mu_cr < 0:
                    cr = 0.0
                else:
                    cr = float(np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0))

                # Time-varying CR lower bounds (jSO)
                if nfes < 0.25 * max_fe and cr < 0.7:
                    cr = 0.7
                elif nfes < 0.50 * max_fe and cr < 0.6:
                    cr = 0.6

                # ── Generate F ────────────────────────────────────
                F = 0.0
                while F <= 0.0:
                    F = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                F = min(F, 1.0)

                # Time-varying F upper bound (jSO)
                if nfes < 0.6 * max_fe and F > 0.7:
                    F = 0.7

                pop_sf[target] = F
                pop_cr[target] = cr

                # ── Weighted jF for pbest term (jSO innovation) ───
                if nfes < 0.2 * max_fe:
                    jF = F * 0.7
                elif nfes < 0.4 * max_fe:
                    jF = F * 0.8
                else:
                    jF = F * 1.2

                # ── Select p_best ─────────────────────────────────
                # Avoid self as pbest in first half of run (jSO)
                if nfes < 0.5 * max_fe:
                    p_best_ind = sorted_index[np.random.randint(p_num)]
                    attempts = 0
                    while p_best_ind == target and attempts < 100:
                        p_best_ind = sorted_index[np.random.randint(p_num)]
                        attempts += 1
                else:
                    p_best_ind = sorted_index[np.random.randint(p_num)]

                # ── Select r1 (≠ target) ──────────────────────────
                r1 = np.random.randint(pop_size)
                while r1 == target:
                    r1 = np.random.randint(pop_size)

                # ── Select r2 (≠ target, ≠ r1) from pop+archive ──
                pool_size = pop_size + arc_ind_count
                r2 = np.random.randint(pool_size)
                while r2 == target or r2 == r1:
                    r2 = np.random.randint(pool_size)

                # r2 source: population or archive
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]

                # ── Mutation + crossover ──────────────────────────
                child = np.empty(d)
                j_rand = np.random.randint(d)
                for j in range(d):
                    if np.random.rand() < cr or j == j_rand:
                        child[j] = (pop[target][j]
                                    + jF * (pop[p_best_ind][j] - pop[target][j])
                                    + F  * (pop[r1][j]         - x_r2[j]))
                    else:
                        child[j] = pop[target][j]

                # ── Boundary: midpoint (modifySolutionWithParentMedium)
                below = child < lb
                above = child > ub
                child = np.where(below, (pop[target] + lb) / 2, child)
                child = np.where(above, (pop[target] + ub) / 2, child)

                # ── Evaluate ──────────────────────────────────────
                f_child = problem.evaluate(child)
                nfes   += 1

                # ── Selection: two cases matching C++ lshade.cc ──────
                # Case 1: tie (==) — replace only, no archive, no recording
                # Case 2: strict improvement (<) — archive + record + replace
                if f_child == fitness[target]:
                    pop[target]     = child
                    fitness[target] = f_child

                elif f_child < fitness[target]:
                    # Add parent to archive
                    if arc_size > 1:
                        if arc_ind_count < arc_size:
                            archive[arc_ind_count] = pop[target].copy()
                            arc_ind_count += 1
                        else:
                            rnd = np.random.randint(arc_size)
                            archive[rnd] = pop[target].copy()

                    # Record fitness difference for weighted memory update
                    dif = fitness[target] - f_child   # always > 0 here
                    success_sf_list.append((pop_sf[target], pop_cr[target], dif))

                    pop[target]     = child
                    fitness[target] = f_child

            # ── Memory update (jSO: weighted Lehmer + Janez averaging) ──
            if len(success_sf_list) > 0:
                old_sf = memory_sf[memory_pos]
                old_cr = memory_cr[memory_pos]

                sf_arr  = np.array([x[0] for x in success_sf_list])
                cr_arr  = np.array([x[1] for x in success_sf_list])
                dif_arr = np.array([x[2] for x in success_sf_list])

                sum_dif = dif_arr.sum()
                w = dif_arr / (sum_dif + 1e-300)

                # Weighted Lehmer mean
                new_sf = (np.sum(w * sf_arr**2) /
                          (np.sum(w * sf_arr) + 1e-300))

                if np.max(cr_arr) == 0 or memory_cr[memory_pos] < 0:
                    new_cr = -1.0
                else:
                    new_cr = (np.sum(w * cr_arr**2) /
                              (np.sum(w * cr_arr) + 1e-300))

                # Janez averaging with old slot (jSO)
                memory_sf[memory_pos] = (new_sf + old_sf) / 2.0
                if new_cr < 0:
                    memory_cr[memory_pos] = -1.0
                else:
                    memory_cr[memory_pos] = (new_cr + old_cr) / 2.0

                memory_pos = (memory_pos + 1) % H

            # ── Linear Population Size Reduction ─────────────────
            plan_pop_size = int(round(
                ((min_pop - max_pop_size) / max_fe) * nfes + max_pop_size
            ))

            if pop_size > plan_pop_size:
                reduction = min(pop_size - plan_pop_size, pop_size - min_pop)
                if reduction > 0:
                    worst_sorted = np.argsort(fitness)[::-1]
                    remove_idx   = worst_sorted[:reduction]
                    keep         = np.ones(pop_size, dtype=bool)
                    keep[remove_idx] = False
                    pop      = pop[keep]
                    fitness  = fitness[keep]
                    pop_size = int(np.sum(keep))

                    # Resize archive
                    arc_size = int(round(arc_rate * pop_size))
                    if arc_ind_count > arc_size:
                        arc_ind_count = arc_size

                    # Resize archive array
                    archive = archive[:max(arc_size, 1)]

        return self._make_result(problem)
