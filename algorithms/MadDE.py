# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
MadDE — Multiple Adaptation Differential Evolution
====================================================
Reference:
    Biswas, S., Saha, D., De, S., Cobb, A. D., Das, S., & Jalaian, B. (2021).
    Improving differential evolution through Bayesian hyperparameter
    optimization.
    Proceedings of the 2021 IEEE Congress on Evolutionary Computation
    (CEC 2021), pp. 832–840.
    DOI: 10.1109/CEC45853.2021.9504792

Official code:
    MadDE-main.zip — MadDE/MadDE.m, MadDE/gnR1R2.m
    This is a direct Python translation of those files.
    CEC 2021 competition entry.

Key mechanism:
    MadDE = L-SHADE framework + three novel DE operators + q-best crossover.

    THREE MUTATION OPERATORS with adaptive probability (same scheme as IMODE):

        Op1: DE/current-to-pbest/1 with archive
            vi = x + F*(pbest - x + r1 - r2)
            pbest from top p% (p_best_rate=0.18); r2 from pop ∪ archive

        Op2: DE/current-to-rand/1 with archive
            vi = x + F*(r1 - r2)
            Simple random difference, no pbest bias; r2 from pop ∪ archive

        Op3: DE/weighted-rand-to-qbest/1 with time-varying attraction
            vi = F*(r1 + α*(qbest - r3))
            q_best_rate = 2p - p*(nfes/max) → starts 0.36, ends 0.18
            attraction α = 0.5 + 0.5*(nfes/max) → starts 0.5, ends 1.0
            r3 from population only

    Q-BEST CROSSOVER (novel):
        Standard binomial mask determines which dims come from vi vs parent.
        But the "parent" is not purely the target individual — with probability
        q_cr_rate=0.01 per row, that row's parent is replaced by a solution
        drawn from the top q% of the combined pop+archive pool.
        This injects q-best solutions as crossover parents with low probability.

    OPERATOR PROBABILITY ADAPTATION (same as IMODE):
        count_S[k] = max(0, mean relative improvement for op k)
        if nonzero: prob = max(0.1, min(0.9, count_S/sum))
        else: [1/3, 1/3, 1/3]

    POPULATION INIT:
        Official code uses Sobol quasi-random sequence (MATLAB sobolset).
        Replaced with uniform random — sobolset is MATLAB-specific.

    BOUNDARY: boundConstraint.m (midpoint).
    ARCHIVE: updateArchive (deduplicate + randperm trim).
    MEMORY: Weighted Lehmer mean; reset to 0.5 on zero successes.
    POPULATION REDUCTION: Linear (same as L-SHADE).

Parameters (exactly matching MadDE.m):
    init_rate  : pop = init_rate * D^2  (default: 2)
    min_pop    : minimum population     (default: 4)
    arc_rate   : archive multiplier     (default: 2.3)
    H_rate     : memory size = H*D      (default: 10)
    p          : pbest fraction         (default: 0.18)
    q_cr_rate  : q-best crossover prob  (default: 0.01)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class MadDE(BaseAlgorithm):
    """
    MadDE: Multiple Adaptation Differential Evolution.
    Direct Python translation of official MadDE.m.
    CEC 2021 competition entry.
    """

    def __init__(
        self,
        init_rate : int   = 2,     # pop = init_rate * D^2
        min_pop   : int   = 4,
        arc_rate  : float = 2.3,
        H_rate    : int   = 10,    # memory_size = H_rate * D
        p         : float = 0.18,  # p_best_rate
        q_cr_rate : float = 0.01,  # q-best crossover probability
    ):
        super().__init__(name='MadDE')
        self.init_rate = init_rate
        self.min_pop   = min_pop
        self.arc_rate  = arc_rate
        self.H_rate    = H_rate
        self.p         = p
        self.q_cr_rate = q_cr_rate

    def run(self, problem: PortalProblem) -> dict:
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        max_pop  = self.init_rate * d * d   # 2*D^2
        min_pop  = self.min_pop
        pop_size = max_pop
        H        = self.H_rate * d          # 10*D
        arc_rate = self.arc_rate
        p_best_rate = self.p
        q_cr_rate   = self.q_cr_rate

        # ── Memory — init 0.2 ─────────────────────────────────────
        if max_fe < max_pop:
            raise ValueError(
                f"MadDE needs at least {max_pop} evaluations for its initial "
                f"population, but the supplied budget is {max_fe}."
            )

        memory_sf  = np.full(H, 0.2)
        memory_cr  = np.full(H, 0.2)
        memory_pos = 0

        # ── Initialisation ────────────────────────────────────────
        # Uniform random (replaces MATLAB sobolset)
        popold  = lb + np.random.rand(pop_size, d) * (ub - lb)
        fitness = np.array([problem.evaluate(popold[i])
                            for i in range(pop_size)])
        nfes    = pop_size

        arc_max     = int(round(arc_rate * pop_size))
        archive_pop = np.empty((0, d))
        archive_fit = np.empty(0)

        # Operator probabilities
        prob = np.full(3, 1.0 / 3)

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            # Sort population best-first
            order   = np.argsort(fitness)
            fitness = fitness[order]
            pop     = popold[order].copy()
            popold  = pop.copy()

            # ── Sample F and CR from memory ───────────────────────
            mem_idx = np.random.randint(0, H, pop_size)
            mu_sf   = memory_sf[mem_idx]
            mu_cr   = memory_cr[mem_idx]

            # CR ~ N(mu_cr, 0.1), terminal -1 → 0
            cr = np.random.normal(mu_cr, 0.1)
            cr[mu_cr == -1] = 0.0
            cr = np.clip(cr, 0.0, 1.0)

            # F ~ Cauchy(mu_sf, 0.1)
            sf = np.zeros(pop_size)
            for i in range(pop_size):
                f = 0.0
                while f <= 0.0:
                    f = mu_sf[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                sf[i] = min(f, 1.0)

            # ── r1 (pop), r2 (pop+archive), r3 (pop) ─────────────
            pop_all = (np.vstack([pop, archive_pop])
                       if len(archive_pop) > 0 else pop.copy())
            NP_all  = len(pop_all)

            r0 = np.arange(pop_size)
            r1 = self._gen_r1(pop_size, r0)
            r2 = self._gen_r2(NP_all, r1, r0)
            r3 = self._gen_r3(pop_size, r1, r2, r0)

            # ── Operator assignment ───────────────────────────────
            bb   = np.random.rand(pop_size)
            l2   = prob[0] + prob[1]
            de_1 = bb <= prob[0]
            de_2 = (bb > prob[0]) & (bb <= l2)
            de_3 = bb > l2

            # ── pbest pool (Op1) ──────────────────────────────────
            pNP = max(2, int(round(p_best_rate * pop_size)))
            rand_idx = (np.floor(np.random.rand(pop_size) * pNP)
                        .astype(int))
            pbest    = pop[rand_idx]

            # ── q-best pool (Op3 + crossover) ─────────────────────
            t = nfes / max_fe
            q_best_rate = 2 * p_best_rate - p_best_rate * t
            attraction  = 0.5 + 0.5 * t

            qNP_mut  = max(2, int(round(q_best_rate * pop_size)))
            rand_idx3 = (np.floor(np.random.rand(pop_size) * qNP_mut)
                         .astype(int))
            qbest_mut = pop[rand_idx3]

            # ── Mutation ──────────────────────────────────────────
            vi   = np.zeros((pop_size, d))
            sf_c = sf[:, np.newaxis]

            # Op1: DE/current-to-pbest/1 with archive
            if np.any(de_1):
                idx = np.where(de_1)[0]
                vi[idx] = (pop[idx]
                           + sf_c[idx] * (pbest[idx]     - pop[idx]
                                          + pop[r1[idx]]
                                          - pop_all[r2[idx]]))

            # Op2: DE/current-to-rand/1 with archive
            if np.any(de_2):
                idx = np.where(de_2)[0]
                vi[idx] = (pop[idx]
                           + sf_c[idx] * (pop[r1[idx]] - pop_all[r2[idx]]))

            # Op3: DE/weighted-rand-to-qbest/1 with attraction
            if np.any(de_3):
                idx = np.where(de_3)[0]
                vi[idx] = sf_c[idx] * (pop[r1[idx]]
                                       + attraction * (qbest_mut[idx]
                                                       - pop[r3[idx]]))

            # ── Boundary: midpoint (boundConstraint.m) ────────────
            below = vi < lb
            above = vi > ub
            vi = np.where(below, (pop + lb) / 2, vi)
            vi = np.where(above, (pop + ub) / 2, vi)

            # ── q-best crossover ──────────────────────────────────
            # Standard binomial mask
            j_rand = np.random.randint(0, d, pop_size)
            mask   = np.random.rand(pop_size, d) > cr[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = False   # jrand always from vi

            # Build q-best parent pool from pop ∪ archive
            qNP_cr    = max(2, int(round(q_best_rate * NP_all)))
            rand_cr   = (np.floor(np.random.rand(NP_all) * qNP_cr)
                         .astype(int))
            pop_all_best = pop_all[rand_cr][:pop_size]

            # With prob q_cr_rate per row, use q-best as crossover parent
            bb_cr   = np.random.rand(pop_size) <= q_cr_rate
            q_parent = pop.copy()
            q_parent[bb_cr] = pop_all_best[bb_cr]

            # Apply mask: where mask=True take from q_parent, else from vi
            ui = vi.copy()
            ui[mask] = q_parent[mask]

            # ── Evaluate ──────────────────────────────────────────
            children_fitness = np.array([
                problem.evaluate(ui[i]) if not problem.is_done()
                else fitness[i]
                for i in range(pop_size)
            ])
            nfes += pop_size

            dif = np.abs(fitness - children_fitness)

            # ── Identify successes (strict <) ─────────────────────
            I_better = fitness > children_fitness
            good_F   = sf[I_better]
            good_CR  = cr[I_better]
            dif_val  = dif[I_better]

            # ── Operator probability update ───────────────────────
            rel_imp = np.maximum(0,
                (fitness - children_fitness) / (np.abs(fitness) + 1e-300))
            count_S = np.array([
                max(0.0, float(np.mean(rel_imp[de_1]))) if np.any(de_1) else 0.0,
                max(0.0, float(np.mean(rel_imp[de_2]))) if np.any(de_2) else 0.0,
                max(0.0, float(np.mean(rel_imp[de_3]))) if np.any(de_3) else 0.0,
            ])
            if np.any(count_S != 0):
                prob = np.maximum(0.1, np.minimum(0.9,
                                  count_S / (count_S.sum() + 1e-300)))
                prob /= prob.sum()
            else:
                prob = np.full(3, 1.0 / 3)

            # ── Archive update ────────────────────────────────────
            archive_pop, archive_fit = self._update_archive(
                archive_pop, archive_fit,
                popold[I_better], fitness[I_better],
                arc_max
            )

            # ── Selection (min of parent/child) ───────────────────
            both    = np.stack([fitness, children_fitness], axis=1)
            I_idx   = np.argmin(both, axis=1)
            fitness = both[np.arange(pop_size), I_idx]
            popold  = pop.copy()
            popold[I_idx == 1] = ui[I_idx == 1]

            # ── Memory update ─────────────────────────────────────
            if len(good_F) > 0:
                w = dif_val / (dif_val.sum() + 1e-300)
                memory_sf[memory_pos] = (
                    (w * good_F**2).sum() /
                    ((w * good_F).sum() + 1e-300)
                )
                if np.max(good_CR) == 0 or memory_cr[memory_pos] == -1:
                    memory_cr[memory_pos] = -1
                else:
                    memory_cr[memory_pos] = (
                        (w * good_CR**2).sum() /
                        ((w * good_CR).sum() + 1e-300)
                    )
                memory_pos = (memory_pos + 1) % H
            else:
                memory_cr[memory_pos] = 0.5
                memory_sf[memory_pos] = 0.5

            # ── Linear population size reduction ──────────────────
            plan = max(min_pop,
                       int(round(((min_pop - max_pop) / max_fe)
                                 * nfes + max_pop)))
            if pop_size > plan:
                reduction = min(pop_size - plan, pop_size - min_pop)
                if reduction > 0:
                    worst = np.argsort(fitness)[::-1][:reduction]
                    keep  = np.ones(pop_size, dtype=bool)
                    keep[worst] = False
                    popold   = popold[keep]
                    fitness  = fitness[keep]
                    pop_size = int(np.sum(keep))

                    arc_max = int(round(arc_rate * pop_size))
                    if len(archive_pop) > arc_max:
                        rnd = np.random.permutation(len(archive_pop))[:arc_max]
                        archive_pop = archive_pop[rnd]
                        archive_fit = archive_fit[rnd]

        return self._make_result(problem)

    # ── Helpers ──────────────────────────────────────────────────

    def _gen_r1(self, NP, r0):
        """r1 ≠ r0, uniform from [0, NP)."""
        r1 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(1000):
            pos = (r1 == r0)
            if not np.any(pos):
                break
            r1[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)
        return r1

    def _gen_r2(self, NP_all, r1, r0):
        """r2 ≠ r1, ≠ r0, from [0, NP_all)."""
        NP = len(r0)
        r2 = np.floor(np.random.rand(NP) * NP_all).astype(int)
        for _ in range(1000):
            pos = (r2 == r1) | (r2 == r0)
            if not np.any(pos):
                break
            r2[pos] = np.floor(
                np.random.rand(np.sum(pos)) * NP_all
            ).astype(int)
        return r2

    def _gen_r3(self, NP, r1, r2, r0):
        """r3 ≠ r0, ≠ r1, ≠ r2, from [0, NP)."""
        r3 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(1000):
            pos = (r3 == r0) | (r3 == r1) | (r3 == r2)
            if not np.any(pos):
                break
            r3[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)
        return r3

    def _update_archive(self, archive_pop, archive_fit,
                        new_pop, new_fit, arc_max):
        """updateArchive: append, deduplicate, randperm trim."""
        if arc_max == 0 or len(new_pop) == 0:
            return archive_pop, archive_fit
        if len(archive_pop) > 0:
            pop_all = np.vstack([archive_pop, new_pop])
            fit_all = np.concatenate([archive_fit, new_fit])
        else:
            pop_all = new_pop.copy()
            fit_all = new_fit.copy()
        _, IX = np.unique(pop_all, axis=0, return_index=True)
        pop_all = pop_all[np.sort(IX)]
        fit_all = fit_all[np.sort(IX)]
        if len(pop_all) > arc_max:
            rnd     = np.random.permutation(len(pop_all))[:arc_max]
            pop_all = pop_all[rnd]
            fit_all = fit_all[rnd]
        return pop_all, fit_all
