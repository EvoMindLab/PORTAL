# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
NL-SHADE-RSP — Non-Linear SHADE with Rank-based Selective Pressure
===================================================================
Reference:
    Stanovov, V., Akhmedova, S., & Semenkin, E. (2021).
    NL-SHADE-RSP algorithm with adaptive archive and selective pressure
    for CEC 2021 numerical optimization.
    Proceedings of the 2021 IEEE Congress on Evolutionary Computation
    (CEC 2021), pp. 809-816.
    DOI: 10.1109/CEC45853.2021.9504959

Reference source:
    NL-SHADE-RSP.zip — nl_shade_rsp.cpp
    The reference code was consulted for clarification.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    NL-SHADE-RSP builds on SHADE with these additions/changes:

    1. NON-LINEAR POPULATION SIZE REDUCTION:
       plan = round((min-max) * t^(1-t) + max)  where t = nfes/maxFES
       Same formula applies to archive size.

    2. EXPONENTIAL + BINOMIAL CROSSOVER (50/50 per generation):
       Each generation randomly picks ONE crossover type for ALL individuals:
       - Binomial (CrossExponential=0): uses time-varying CrToUse
       - Exponential (CrossExponential=1): uses memory-sampled Cr
       Both use jrand (WillCrossover) to guarantee at least one dimension.

    3. CR SORTED BY FITNESS RANK:
       All NInds CR values are generated, then sorted ascending.
       Each individual gets the CR corresponding to its fitness rank
       via BackIndexes: better individuals get SMALLER CR (exploit),
       worse individuals get LARGER CR (explore).

    4. EXPONENTIAL RANK WEIGHTS FOR r2 (RSP):
       ComponentSelector3: FitTemp3[i] = exp(-i/NInds)
       Sorted best-first, so better individuals have higher probability.

    5. ADAPTIVE ARCHIVE PROBABILITY:
       ArchProbs = ArchSuccess/(ArchSuccess+NoArchSuccess) in [0.1, 0.9]
       where improvements are measured as relative fitness gain.

    6. LINEAR p_best GROWTH (from C++, not paper):
       psizeval = max(2, NInds * (0.2/MaxFEval*NFEval + 0.2))
       p grows from 0.2 to 0.4 over the run (C++ formula).

    7. BOUNDARY: random resampling (not midpoint).
    8. MEMORY init: 0.2 (not 0.5).
    9. MEMORY POINTER: advances every generation (even on failure).
    10. SELECTION: <= archives + replaces; < also records F/CR.

Parameters (exactly matching nl_shade_rsp.cpp Initialize call):
    init_rate : initial pop = rate * D    (default: 30, C++: GNVars*30)
    min_pop   : minimum population        (default: 4)
    H_rate    : memory size = H_rate * D  (default: 20, C++: 20*GNVars)
    arc_rate  : archive multiplier        (default: 2.1)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class NL_SHADE_RSP(BaseAlgorithm):
    """
    NL-SHADE-RSP: Non-Linear SHADE with Rank-based Selective Pressure.
    PORTAL Python implementation informed by nl_shade_rsp.cpp.
    CEC 2021 winner.
    """

    def __init__(
        self,
        init_rate : int   = 30,    # initial pop = rate * D (C++: GNVars*30)
        min_pop   : int   = 4,
        H_rate    : int   = 20,    # memory size = H_rate * D
        arc_rate  : float = 2.1,
    ):
        super().__init__(name='NL-SHADE-RSP')
        self.init_rate = init_rate
        self.min_pop   = min_pop
        self.H_rate    = H_rate
        self.arc_rate  = arc_rate

    def run(self, problem: PortalProblem) -> dict:
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        max_pop  = self.init_rate * d
        min_pop  = self.min_pop
        pop_size = max_pop
        H        = self.H_rate * d
        arc_rate = self.arc_rate

        if max_fe < pop_size:
            raise ValueError(
                f"NL-SHADE-RSP needs at least {pop_size} evaluations for "
                f"its initial population, but the supplied budget is {max_fe}."
            )

        arc_size = max(min_pop, int(round(arc_rate * pop_size)))
        arch_cur = 0   # CurrentArchiveSize

        # ── Memory — initialised to 0.2 (C++) ─────────────────────
        memory_sf  = np.full(H, 0.2)
        memory_cr  = np.full(H, 0.2)
        memory_pos = 0

        # ── Initialisation ────────────────────────────────────────
        pop     = lb + np.random.rand(pop_size, d) * (ub - lb)
        fitness = np.array([problem.evaluate(pop[i]) for i in range(pop_size)])
        nfes    = pop_size

        archive = np.zeros((max(1, int(np.ceil(arc_rate * max_pop))), d))

        arch_probs = 0.5

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            N = pop_size

            # Sort best-first; build BackIndexes
            sorted_index = np.argsort(fitness)   # sorted_index[0]=best
            back_index   = np.empty(N, dtype=int)
            for j in range(N):
                back_index[sorted_index[j]] = j

            # ── Exponential rank weights for ComponentSelector3 ───
            # C++: FitTemp3[i] = exp(-i/NInds), sorted best-first
            fit_temp3 = np.exp(-np.arange(N) / N)
            fit_temp3 /= fit_temp3.sum()

            # ── pbest pool size (linear growth C++) ───────────────
            # psizeval = max(2, NInds*(0.2/MaxFEval*NFEval + 0.2))
            psizeval = max(2, int(round(N * (0.2 / max_fe * nfes + 0.2))))
            psizeval = min(psizeval, N)

            # ── 50/50 crossover type for this generation ──────────
            cross_exponential = (np.random.rand() < 0.5)

            # ── Time-varying CR for binomial crossover ────────────
            t = nfes / max_fe
            cr_to_use = 0.0 if t <= 0.5 else (t - 0.5) * 2.0

            # ── Generate CR and F for all individuals ─────────────
            mem_idx = np.random.randint(0, H, N)
            mu_cr   = memory_cr[mem_idx]
            mu_sf   = memory_sf[mem_idx]

            cr_generated = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)
            f_generated  = np.zeros(N)
            for i in range(N):
                f = 0.0
                while f <= 0.0:
                    f = mu_sf[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                f_generated[i] = min(f, 1.0)

            # ── Sort CR ascending; assign by fitness rank ─────────
            # C++: qSort1(CrGenerated); Cr = CrGenerated[BackIndexes[i]]
            cr_generated_sorted = np.sort(cr_generated)
            # cr_to_assign[i] = cr_generated_sorted[back_index[i]]

            # Archive usage tracking
            arch_success    = 0.0
            no_arch_success = 0.0
            n_arch_usages   = 0

            success_sf = []
            success_cr = []
            fit_delta  = []
            arch_usages = np.zeros(N)

            pop_temp = pop.copy()
            fit_temp = fitness.copy()

            for i in range(N):
                if problem.is_done():
                    break

                F  = f_generated[i]
                CR = cr_generated_sorted[back_index[i]]   # sorted assignment

                # ── pbest: from top psizeval (sorted best-first) ──
                pbest_rank = np.random.randint(psizeval)
                pbest_idx  = sorted_index[pbest_rank]

                # ── r1: uniform ≠ i ───────────────────────────────
                r1 = np.random.randint(N)
                for _ in range(25):
                    if r1 != pbest_idx and r1 != i:
                        break
                    r1 = np.random.randint(N)

                # ── r2: exponential rank or archive ───────────────
                use_archive = (arch_cur > 0 and np.random.rand() < arch_probs)
                if use_archive:
                    r2_arch = np.random.randint(arch_cur)
                    x_r2    = archive[r2_arch]
                    arch_usages[i] = 1
                else:
                    # ComponentSelector3: exp rank from sorted pop
                    r2_rank = np.random.choice(N, p=fit_temp3)
                    r2      = sorted_index[r2_rank]
                    for _ in range(25):
                        if r2 != pbest_idx and r2 != r1:
                            break
                        r2_rank = np.random.choice(N, p=fit_temp3)
                        r2      = sorted_index[r2_rank]
                    x_r2 = pop[r2]
                    arch_usages[i] = 0

                # ── Mutation ──────────────────────────────────────
                donor = (pop[i]
                         + F * (pop[pbest_idx] - pop[i])
                         + F * (pop[r1]        - x_r2))

                # ── Boundary: random resampling (FindLimits) ──────
                out   = (donor < lb) | (donor > ub)
                donor = np.where(out,
                                 lb + np.random.rand(d) * (ub - lb),
                                 donor)

                # ── Crossover ─────────────────────────────────────
                j_rand = np.random.randint(d)

                if not cross_exponential:
                    # Binomial with time-varying cr_to_use
                    mask  = np.random.rand(d) < cr_to_use
                    mask[j_rand] = True
                    trial = np.where(mask, donor, pop[i])
                else:
                    # Exponential with memory-sampled CR
                    trial     = pop[i].copy()
                    start_loc = np.random.randint(d)
                    L         = start_loc + 1
                    while np.random.rand() < CR and L < d:
                        L += 1
                    trial[start_loc:L] = donor[start_loc:L]
                    trial[j_rand]      = donor[j_rand]   # jrand guarantee

                # ── Evaluate ──────────────────────────────────────
                f_trial = problem.evaluate(trial)
                nfes   += 1
                fit_temp[i] = f_trial
                pop_temp[i] = trial

            # ── Selection + archive (batch) ───────────────────────
            for i in range(N):
                if fit_temp[i] <= fitness[i]:
                    if arch_cur < arc_size:
                        archive[arch_cur] = pop[i].copy()
                        arch_cur += 1
                    elif arc_size > 0:
                        rnd = np.random.randint(arc_size)
                        archive[rnd] = pop[i].copy()

                    if fitness[i] != 0:
                        rel_imp = (fitness[i] - fit_temp[i]) / abs(fitness[i])
                    else:
                        rel_imp = fitness[i] - fit_temp[i]

                    if arch_usages[i] == 1:
                        arch_success += rel_imp
                        n_arch_usages += 1
                    else:
                        no_arch_success += rel_imp

                    if fit_temp[i] < fitness[i]:
                        success_sf.append(f_generated[i])
                        success_cr.append(cr_generated_sorted[back_index[i]])
                        fit_delta.append(fitness[i] - fit_temp[i])

                    pop[i]     = pop_temp[i]
                    fitness[i] = fit_temp[i]

            # ── Update archive probability ─────────────────────────
            if n_arch_usages > 0:
                arch_success    /= n_arch_usages
                no_arch_success /= max(1, N - n_arch_usages)
                if arch_success == 0:
                    arch_probs = 0.5
                else:
                    arch_probs = arch_success / (arch_success + no_arch_success)
                arch_probs = float(np.clip(arch_probs, 0.1, 0.9))
            else:
                arch_probs = 0.5

            # ── Non-linear population size reduction ──────────────
            t2 = nfes / max_fe
            t2 = min(t2, 1.0)
            new_pop = max(min_pop,
                          int(round((min_pop - max_pop) * (t2 ** (1 - t2)) + max_pop)))
            new_arc = max(min_pop,
                          int(round(((min_pop - max_pop) * (t2 ** (1 - t2)) + max_pop)
                                    * arc_rate)))
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

            # ── Memory update (advances every generation) ─────────
            if len(success_sf) > 0:
                sf_arr  = np.array(success_sf)
                cr_arr2 = np.array(success_cr)
                df_arr  = np.array(fit_delta)
                w = df_arr / (df_arr.sum() + 1e-300)

                memory_sf[memory_pos] = (
                    (w * sf_arr**2).sum() / ((w * sf_arr).sum() + 1e-300)
                )
                memory_cr[memory_pos] = (
                    (w * cr_arr2**2).sum() / ((w * cr_arr2).sum() + 1e-300)
                )
            else:
                # C++: reset to 0.5 when no successes
                memory_sf[memory_pos] = 0.5
                memory_cr[memory_pos] = 0.5

            # Advance every generation (C++ convention)
            memory_pos = (memory_pos + 1) % H

        return self._make_result(problem)
