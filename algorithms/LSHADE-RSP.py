# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
LSHADE-RSP — L-SHADE with Rank-based Selective Pressure
=========================================================
Reference:
    Stanovov, V., Akhmedova, S., & Semenkin, E. (2018).
    LSHADE algorithm with rank-based selective pressure strategy
    for solving CEC 2017 benchmark problems.
    Proceedings of the 2018 IEEE Congress on Evolutionary Computation
    (CEC 2018).
    DOI: 10.1109/CEC.2018.8477977

Reference source:
    https://github.com/VladimirStanovov/LSHADE-RSP
    main.cpp (author reference)
    The reference code was consulted for clarification.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    LSHADE-RSP = L-SHADE + Rank-based Selective Pressure for r1/r2,
    plus several jSO-style parameter modifications.

    RANK-BASED SELECTIVE PRESSURE (RSP):
        FitTemp[i] = 3.0 * (N - i)  (linear, sorted best-first)
        Both r1 and r2 drawn from this distribution (not uniform).
        Better individuals more likely to be selected as donors.

    jSO-STYLE MODIFICATIONS (all from main.cpp):
        - Weighted F: F2 = F*0.7/0.8/1.2 for pbest term
        - CR lower bounds: >=0.7 first 25%, >=0.6 first 50%
        - F upper bound: <=0.7 first 60%
        - pbest avoidance: no self-reference in first 50%
        - Memory last slot (H): forced Cauchy(0.9,0.1)/Normal(0.9,0.1)
        - Janez averaging in memory update

    DIFFERENCES FROM jSO:
        - Population: int(75*D^(2/3)) not sqrt(D)*ln(D)*25
        - Boundary: midpoint (not random resampling)
        - Memory init: 0.3/0.8 (same as jSO)
        - Archive selection: ratio-based (CurrentArchSize/(CurrentArchSize+N))
          not a fixed probability
        - Archive SIZE: decreasing linearly to near-zero
        - psize: grows from psizeParam/2 to psizeParam (0.085→0.17)
        - Population reduction: LINEAR (not non-linear like NL-SHADE-RSP)
        - SaveSuccessCrF: on <= (non-strict, unlike jSO's strict <)
        - Memory advances only when successes exist (no reset to 0.5)

Parameters (exactly matching main.cpp Initialize/main calls):
    psize_param : p_best parameter    (default: 0.17, C++: NewPsize)
    arc_rate    : archive multiplier  (default: 1.0,  C++: NewArchSize)
    H           : memory size         (default: 5,    C++: hardcoded)
    min_pop     : minimum population  (default: 4)
    init_fn     : initial pop formula (default: int(75*D^(2/3)))
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class LSHADE_RSP(BaseAlgorithm):
    """
    LSHADE-RSP: L-SHADE with Rank-based Selective Pressure.
    PORTAL Python implementation informed by the author's main.cpp.
    """

    def __init__(
        self,
        psize_param : float = 0.17,   # p_best parameter (C++: NewPsize=0.17)
        arc_rate    : float = 1.0,    # archive multiplier (C++: NewArchSize=1.0)
        H           : int   = 5,      # memory size (C++: hardcoded MemorySize=5)
        min_pop     : int   = 4,      # NIndsMin
    ):
        super().__init__(name='LSHADE-RSP')
        self.psize_param = psize_param
        self.arc_rate    = arc_rate
        self.H           = H
        self.min_pop     = min_pop

    def run(self, problem: PortalProblem) -> dict:
        """
        Run LSHADE-RSP on a PORTAL problem.
        """
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        # pop_size = int(75 * D^(2/3))  (C++: main.cpp line)
        max_pop  = int(75 * (d ** (2.0 / 3.0)))
        max_pop  = max(max_pop, self.min_pop)
        min_pop  = self.min_pop
        pop_size = max_pop
        H        = self.H
        arc_rate = self.arc_rate
        psize_param = self.psize_param

        if max_fe < pop_size:
            raise ValueError(
                f"LSHADE-RSP needs at least {pop_size} evaluations for its "
                f"initial population, but the supplied budget is {max_fe}."
            )

        # Archive: initial size = arc_rate*(NIndsMax-NIndsMin)
        # then decreases linearly (see LSHADE-RSP archive formula)
        arc_max_init  = max(min_pop, int(arc_rate * (max_pop - min_pop)))
        arc_size      = arc_max_init
        arch_cur      = 0   # CurrentArchiveSize (counter)

        # ── Memory — init 0.3/0.8 (C++) ───────────────────────────
        memory_f  = np.full(H, 0.3)
        memory_cr = np.full(H, 0.8)
        memory_pos = 0   # MemoryIter

        # ── Initialisation ────────────────────────────────────────
        pop     = lb + np.random.rand(pop_size, d) * (ub - lb)
        fitness = np.array([problem.evaluate(pop[i]) for i in range(pop_size)])
        nfes    = pop_size

        # Archive storage (pre-allocate at max possible size)
        archive     = np.zeros((max(1, arc_max_init + 1), d))
        archive_fit = np.zeros(max(1, arc_max_init + 1))

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            N = pop_size

            # Sort best-first
            sorted_index = np.argsort(fitness)

            # ── Linear RSP weights: FitTemp[i] = 3*(N-i) ─────────
            fit_temp = np.array([3.0 * (N - i) for i in range(N)],
                                dtype=float)
            fit_temp /= fit_temp.sum()

            # ── psize grows from psizeParam/2 to psizeParam ───────
            psize    = (psize_param / 2.0) / max_fe * nfes + psize_param / 2.0
            psizeval = max(2, int(N * psize))
            psizeval = min(psizeval, N)

            success_f  = []
            success_cr = []
            fit_delta  = []

            pop_temp = pop.copy()
            fit_temp_arr = fitness.copy()

            for i in range(N):
                if problem.is_done():
                    break

                # ── Memory slot selection ─────────────────────────
                # IntRandom(MemorySize+1) → range [0, H] inclusive
                mem_idx = np.random.randint(H + 1)

                # Generate F
                if mem_idx < H:
                    F = 0.0
                    while F <= 0.0:
                        F = memory_f[mem_idx] + 0.1 * np.tan(
                            np.pi * (np.random.rand() - 0.5))
                else:
                    # Last slot: Cauchy(0.9, 0.1)
                    F = 0.0
                    while F <= 0.0:
                        F = 0.9 + 0.1 * np.tan(
                            np.pi * (np.random.rand() - 0.5))
                F = min(F, 1.0)

                # F upper bound (jSO-style)
                if nfes / max_fe < 0.6 and F > 0.7:
                    F = 0.7

                # Weighted F for pbest term (jSO-style)
                t = nfes / max_fe
                if t < 0.2:
                    F2 = F * 0.7
                elif t < 0.4:
                    F2 = F * 0.8
                else:
                    F2 = F * 1.2

                # Generate CR
                if mem_idx < H:
                    mu_cr = memory_cr[mem_idx]
                    if mu_cr < 0:
                        CR = 0.0
                    else:
                        CR = float(np.clip(np.random.normal(mu_cr, 0.1),
                                           0.0, 1.0))
                else:
                    # Last slot: Normal(0.9, 0.1)
                    CR = float(np.clip(np.random.normal(0.9, 0.1), 0.0, 1.0))

                # CR lower bounds (jSO-style)
                if t < 0.25:
                    CR = max(CR, 0.7)
                elif t < 0.50:
                    CR = max(CR, 0.6)

                # ── pbest selection ───────────────────────────────
                prand = sorted_index[np.random.randint(psizeval)]
                # Avoidance: no self in first half
                if t < 0.5:
                    attempts = 0
                    while prand == i and attempts < 100:
                        prand = sorted_index[np.random.randint(psizeval)]
                        attempts += 1

                # ── r1: RSP from population (≠ prand) ────────────
                r1 = sorted_index[np.random.choice(N, p=fit_temp)]
                for _ in range(25):
                    if r1 != prand:
                        break
                    r1 = sorted_index[np.random.choice(N, p=fit_temp)]
                # Note: C++ only enforces r1 ≠ prand, NOT r1 ≠ i.
                # This matches the official code; r1 can equal the target.

                # ── r2: ratio-based archive or RSP ────────────────
                # C++: if rand < CurrentArchSize/(CurrentArchSize+N): archive
                if (arch_cur > 0 and
                        np.random.rand() < arch_cur / (arch_cur + N)):
                    r2_arch = np.random.randint(arch_cur)
                    x_r2    = archive[r2_arch]
                else:
                    r2 = sorted_index[np.random.choice(N, p=fit_temp)]
                    for _ in range(25):
                        if r2 != prand and r2 != r1:
                            break
                        r2 = sorted_index[np.random.choice(N, p=fit_temp)]
                    x_r2 = pop[r2]

                # ── Mutation ──────────────────────────────────────
                donor = (pop[i]
                         + F2 * (pop[prand] - pop[i])
                         + F  * (pop[r1]    - x_r2))

                # ── Boundary: midpoint (FindLimits) ───────────────
                below = donor < lb
                above = donor > ub
                donor = np.where(below, (lb + pop[i]) / 2, donor)
                donor = np.where(above, (ub + pop[i]) / 2, donor)

                # ── Crossover: binomial ───────────────────────────
                j_rand = np.random.randint(d)
                mask   = np.random.rand(d) < CR
                mask[j_rand] = True
                trial  = np.where(mask, donor, pop[i])

                # ── Evaluate ──────────────────────────────────────
                f_trial = problem.evaluate(trial)
                nfes   += 1
                fit_temp_arr[i] = f_trial
                pop_temp[i]     = trial

                # ── Record success (non-strict <=, C++) ───────────
                # C++: if(FitTemp[i] <= FitMass[i]) SaveSuccessCrF(...)
                # Paper Fig.1 uses strict <; C++ uses <=. On a tie
                # fit_delta=0, contributing zero weight to Lehmer mean.
                if f_trial <= fitness[i]:
                    success_f.append(F)
                    success_cr.append(CR)
                    fit_delta.append(abs(fitness[i] - f_trial))

            # ── Selection + archive (batch) ───────────────────────
            for i in range(N):
                if fit_temp_arr[i] <= fitness[i]:
                    # Archive parent
                    if arch_cur < arc_size:
                        archive[arch_cur]     = pop[i].copy()
                        archive_fit[arch_cur] = fitness[i]
                        arch_cur += 1
                    elif arc_size > 0:
                        rnd = np.random.randint(arc_size)
                        archive[rnd]     = pop[i].copy()
                        archive_fit[rnd] = fitness[i]

                    pop[i]     = pop_temp[i]
                    fitness[i] = fit_temp_arr[i]

            # ── Linear population size reduction ──────────────────
            new_pop = max(min_pop,
                          int((min_pop - max_pop) / max_fe * nfes + max_pop))

            # Archive size: decreasing formula (C++)
            # newArchSize = (MaxFEval-NFEval)/MaxFEval * (arc_rate*(NIndsMax-NIndsMin))
            new_arc = max(min_pop,
                          int((max_fe - nfes) / max_fe
                              * (arc_rate * (max_pop - min_pop))))
            arc_size = new_arc
            if arch_cur > arc_size:
                arch_cur = arc_size

            if pop_size > new_pop:
                reduction = min(pop_size - new_pop, pop_size - min_pop)
                if reduction > 0:
                    worst = np.argsort(fitness)[::-1][:reduction]
                    keep  = np.ones(pop_size, dtype=bool)
                    keep[worst] = False
                    pop      = pop[keep]
                    fitness  = fitness[keep]
                    pop_size = int(np.sum(keep))

            # ── Memory update (Janez averaging, only on success) ──
            if len(success_f) > 0:
                sf_arr = np.array(success_f)
                cr_arr = np.array(success_cr)
                df_arr = np.array(fit_delta)

                old_f  = memory_f[memory_pos]
                old_cr = memory_cr[memory_pos]

                w = df_arr / (df_arr.sum() + 1e-300)

                new_f  = (w * sf_arr**2).sum() / ((w * sf_arr).sum() + 1e-300)

                # terminal CR check
                cr_max = np.max(cr_arr)
                if memory_cr[memory_pos] == -1 or cr_max == 0:
                    new_cr = -1.0
                else:
                    new_cr = (w * cr_arr**2).sum() / \
                             ((w * cr_arr).sum() + 1e-300)

                # Janez averaging
                memory_f[memory_pos]  = (new_f + old_f) / 2.0
                if new_cr < 0:
                    memory_cr[memory_pos] = -1.0
                else:
                    memory_cr[memory_pos] = (new_cr + old_cr) / 2.0

                memory_pos = (memory_pos + 1) % H
            # (no update / no reset when no successes — C++ behaviour)

        return self._make_result(problem)
