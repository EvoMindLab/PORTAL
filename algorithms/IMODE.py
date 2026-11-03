# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
IMODE — Improved Multi-operator Differential Evolution Algorithm
================================================================
Reference:
    Sallam, K. M., Elsayed, S. M., Chakrabortty, R. K., & Ryan, M. J. (2020).
    Improved multi-operator differential evolution algorithm for solving
    unconstrained problems.
    Proceedings of the 2020 IEEE Congress on Evolutionary Computation
    (CEC 2020), pp. 1–8.
    DOI: 10.1109/CEC48606.2020.9185577

Reference source:
    Official CEC2020 E24365 Updated codes (1).rar:
    IMODE.m, IMODE_main.m, Introd_Par.m, han_boun.m,
    updateArchive.m, gnR1R2.m
    The reference code was consulted for clarification.
    CEC 2020 winner.

Architecture note:
    IMODE_main.m sets Probs=[1,0], disabling CMA-ES entirely.
    The actual competition entry is the DE component (IMODE.m) only.
    The local search LS2 (fmincon) is omitted as it requires a gradient-
    based solver not available in this framework; it only fires in the
    final 15% of budget with probability 0.1.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    IMODE adapts both DE operators and their parameters simultaneously.

    THREE DE OPERATORS:
        Op1: DE/current-to-pbest/1 with archive (p=0.25)
            vi = x + F*(pbest - x + r1 - r2_arch)
        Op2: DE/current-to-pbest/1 pop only (p=0.25)
            vi = x + F*(pbest - x + r1 - r3)
        Op3: DE/rand-to-pbest/1 variant (p=0.50)
            vi = F*r1 + F*(pbest - r3)   [no current term]

    OPERATOR PROBABILITY ADAPTATION:
        count_S[k] = mean relative improvement for op k this generation
        prob = max(0.1, min(0.9, count_S/sum(count_S)))
        On zero improvement: reset to equal [1/3, 1/3, 1/3]

    CROSSOVER (per generation, 40% binomial / 60% exponential):
        Both use CR sorted in ascending order, assigned best-to-worst
        (best individual gets lowest CR = exploit, worst gets highest = explore)

    BOUNDARY: han_boun with hb = randi(3) each call:
        case 1: midpoint  (lb+parent)/2
        case 2: reflection 2*lb-parent (clipped)
        case 3: random resample

    POPULATION REDUCTION: Linear (same formula as L-SHADE)
        UpdPopSize = round(((MinPop - InitPop)/MaxFES)*nfes + InitPop)
        Worst individuals removed (population already sorted).

    ARCHIVE: updateArchive (deduplicate + randperm trim, same as SHADE).

    CR/F ADAPTATION: weighted Lehmer mean, same as SHADE 1.1.1.
        memory_f init = 0.2, memory_cr init = 0.2, memory_size = 20*D
        Reset to 0.5 on zero successes.

Parameters (from IMODE_main.m and Introd_Par.m):
    init_rate  : pop = init_rate * D^2     (default: 6, code: 6*n*n)
    min_pop    : minimum population        (default: 4)
    arch_rate  : archive multiplier        (default: 2.6)
    H_rate     : memory size = H_rate * D  (default: 20)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class IMODE(BaseAlgorithm):
    """
    IMODE: Improved Multi-operator Differential Evolution.
    PORTAL Python implementation informed by IMODE.m / IMODE_main.m.
    CEC 2020 winner.
    """

    def __init__(
        self,
        init_rate : int   = 6,     # pop = init_rate * D^2  (code: 6*n*n)
        min_pop   : int   = 4,     # Par.MinPopSize
        arch_rate : float = 2.6,   # arch_rate * PS1
        H_rate    : int   = 20,    # memory_size = H_rate * D
    ):
        super().__init__(name='IMODE')
        self.init_rate = init_rate
        self.min_pop   = min_pop
        self.arch_rate = arch_rate
        self.H_rate    = H_rate

    def run(self, problem: PortalProblem) -> dict:
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        init_pop = self.init_rate * d * d   # 6*D^2
        min_pop  = self.min_pop
        pop_size = init_pop
        H        = self.H_rate * d          # 20*D
        arch_rate = self.arch_rate

        if max_fe < init_pop:
            raise ValueError(
                f"IMODE needs at least {init_pop} evaluations for its initial "
                f"population, but the supplied budget is {max_fe}."
            )

        arc_max = int(round(arch_rate * pop_size))

        # ── Memory init: 0.2 ─────────────────────────────────────
        memory_f  = np.full(H, 0.2)
        memory_cr = np.full(H, 0.2)
        hist_pos  = 0   # 0-indexed (MATLAB: hist_pos=1, 1-indexed)

        # ── Initialisation ────────────────────────────────────────
        pop     = lb + np.random.rand(pop_size, d) * (ub - lb)
        fitness = np.array([problem.evaluate(pop[i]) for i in range(pop_size)])
        nfes    = pop_size

        # Sort population best-first (IMODE keeps sorted)
        order   = np.argsort(fitness)
        pop     = pop[order]
        fitness = fitness[order]

        archive_pop = np.empty((0, d))
        archive_fit = np.empty(0)

        # Operator probabilities [op1, op2, op3]
        prob = np.array([1/3, 1/3, 1/3])

        # F_arr and CR_arr are generated fresh each generation from memory;
        # no persistent initialisation needed. (Dead code removed — the
        # normrnd init in IMODE_main.m is immediately overwritten each gen.)
        F_arr  = np.zeros(pop_size)
        CR_arr = np.zeros(pop_size)

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            N = pop_size

            # ── Sample F and CR from memory ───────────────────────
            mem_idx = np.random.randint(0, H, N)
            mu_sf   = memory_f[mem_idx]
            mu_cr   = memory_cr[mem_idx]

            # CR ~ N(mu_cr, 0.1), terminal -1 → 0
            CR_arr = np.random.normal(mu_cr, 0.1)
            term   = (mu_cr == -1)
            CR_arr[term] = 0.0
            CR_arr = np.clip(CR_arr, 0.0, 1.0)

            # F ~ Cauchy(mu_sf, 0.1)
            F_arr = np.zeros(N)
            for i in range(N):
                f = 0.0
                while f <= 0.0:
                    f = mu_sf[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                F_arr[i] = min(f, 1.0)

            # Population already sorted best-first
            # Sort CR ascending (best gets low CR, worst gets high CR)
            CR_arr = np.sort(CR_arr)

            # ── r1, r2, r3 selection (gnR1R2 style) ───────────────
            pop_all = (np.vstack([pop, archive_pop])
                       if len(archive_pop) > 0 else pop)
            N_all = len(pop_all)

            r0 = np.arange(N)
            r1 = self._gnR1(N, r0)
            r2, r3 = self._gnR1R2R3(N, N_all, r0, r1)

            # ── Operator assignment ───────────────────────────────
            bb   = np.random.rand(N)
            l2   = prob[0] + prob[1]
            op_1 = bb <= prob[0]
            op_2 = (bb > prob[0]) & (bb <= l2)
            op_3 = bb > l2

            # ── pbest indices ─────────────────────────────────────
            # Op1 & Op2: top 25%
            pNP_25 = max(1, int(round(0.25 * N)))
            rand25 = np.floor(np.random.rand(N) * pNP_25).astype(int)
            rand25 = np.maximum(rand25, 0)
            pbest_25 = rand25   # indices into sorted pop (best first)

            # Op3: top 50%
            pNP_50 = max(2, int(round(0.50 * N)))
            rand50 = np.floor(np.random.rand(N) * pNP_50).astype(int)
            rand50 = np.maximum(rand50, 0)
            pbest_50 = rand50

            # ── Mutation ──────────────────────────────────────────
            vi = np.zeros((N, d))
            F_col = F_arr[:, np.newaxis]

            # Op1: vi = x + F*(pbest25 - x + r1 - r2_arch)
            if np.any(op_1):
                idx = np.where(op_1)[0]
                vi[idx] = (pop[idx]
                           + F_col[idx] * (pop[pbest_25[idx]] - pop[idx]
                                           + pop[r1[idx]]
                                           - pop_all[r2[idx]]))

            # Op2: vi = x + F*(pbest25 - x + r1 - r3_pop)
            if np.any(op_2):
                idx = np.where(op_2)[0]
                vi[idx] = (pop[idx]
                           + F_col[idx] * (pop[pbest_25[idx]] - pop[idx]
                                           + pop[r1[idx]]
                                           - pop[r3[idx]]))

            # Op3: vi = F*r1 + F*(pbest50 - r3)
            if np.any(op_3):
                idx = np.where(op_3)[0]
                vi[idx] = (F_col[idx] * pop[r1[idx]]
                           + F_col[idx] * (pop[pbest_50[idx]] - pop[r3[idx]]))

            # ── Boundary: han_boun (random case each call) ────────
            vi = self._han_boun(vi, lb, ub, pop, N)

            # ── Crossover: 40% binomial / 60% exponential ─────────
            if np.random.rand() < 0.4:
                # Binomial
                CR_col = CR_arr[:, np.newaxis]
                mask   = np.random.rand(N, d) > CR_col
                jrand  = np.floor(np.random.rand(N) * d).astype(int)
                mask[np.arange(N), jrand] = False
                ui = vi.copy()
                ui[mask] = pop[mask]
            else:
                # Exponential
                ui = pop.copy()
                start = np.random.randint(0, d, N)
                for i in range(N):
                    l = start[i] + 1
                    while np.random.rand() < CR_arr[i] and l < d:
                        l += 1
                    ui[i, start[i]:l] = vi[i, start[i]:l]

            # ── Evaluate ──────────────────────────────────────────
            fit_new = np.array([
                problem.evaluate(ui[i]) if not problem.is_done()
                else fitness[i]
                for i in range(N)
            ])
            nfes += N

            # ── Improvement tracking ──────────────────────────────
            diff      = np.abs(fitness - fit_new)
            I_better  = fit_new < fitness   # strict

            good_F    = F_arr[I_better]
            good_CR   = CR_arr[I_better]
            dif_val   = diff[I_better]

            # ── Operator probability update ───────────────────────
            # count_S[k] = mean relative improvement for op k
            rel_imp = np.maximum(0, (fitness - fit_new) / (np.abs(fitness) + 1e-300))
            count_S = np.zeros(3)
            for k, mask_k in enumerate([op_1, op_2, op_3]):
                if np.any(mask_k):
                    count_S[k] = max(0, np.mean(rel_imp[mask_k]))

            if np.any(count_S != 0):
                prob = np.maximum(0.1, np.minimum(0.9,
                                  count_S / (count_S.sum() + 1e-300)))
                prob /= prob.sum()
            else:
                prob = np.full(3, 1/3)

            # ── Archive update ────────────────────────────────────
            archive_pop, archive_fit = self._update_archive(
                archive_pop, archive_fit,
                pop[I_better], fitness[I_better],
                arc_max
            )

            # ── Selection ─────────────────────────────────────────
            pop      = np.where(I_better[:, np.newaxis], ui, pop)
            fitness  = np.where(I_better, fit_new, fitness)

            # ── Sort population best-first ─────────────────────────
            order   = np.argsort(fitness)
            pop     = pop[order]
            fitness = fitness[order]

            # ── Memory update ─────────────────────────────────────
            if len(good_F) > 0:
                w = dif_val / (dif_val.sum() + 1e-300)
                memory_f[hist_pos] = (
                    (w * good_F**2).sum() / ((w * good_F).sum() + 1e-300)
                )
                if np.max(good_CR) == 0 or memory_cr[hist_pos] == -1:
                    memory_cr[hist_pos] = -1
                else:
                    memory_cr[hist_pos] = (
                        (w * good_CR**2).sum()
                        / ((w * good_CR).sum() + 1e-300)
                    )
                hist_pos = (hist_pos + 1) % H
            else:
                # Reset to 0.5 when no successes (IMODE.m)
                memory_cr[hist_pos] = 0.5
                memory_f[hist_pos]  = 0.5

            # ── Linear population size reduction ──────────────────
            upd_size = max(min_pop,
                           int(round(((min_pop - init_pop) / max_fe)
                                     * nfes + init_pop)))

            if pop_size > upd_size:
                reduction = min(pop_size - upd_size, pop_size - min_pop)
                if reduction > 0:
                    # Remove worst (already sorted; worst = last)
                    pop      = pop[:-reduction]
                    fitness  = fitness[:-reduction]
                    pop_size = pop_size - reduction

                    arc_max = int(round(arch_rate * pop_size))
                    if len(archive_pop) > arc_max:
                        rnd = np.random.permutation(len(archive_pop))[:arc_max]
                        archive_pop = archive_pop[rnd]
                        archive_fit = archive_fit[rnd]

        return self._make_result(problem)

    # ── Helpers ──────────────────────────────────────────────────

    def _gnR1(self, NP, r0):
        """Generate r1: ≠ r0[i], uniform from [0,NP)."""
        r1 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(99999):
            pos = (r1 == r0)
            if not np.any(pos):
                break
            r1[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)
        return r1

    def _gnR1R2R3(self, NP, NP_all, r0, r1):
        """Generate r2 (from full pop+archive) and r3 (from pop), all distinct."""
        r2 = np.floor(np.random.rand(NP) * NP_all).astype(int)
        for _ in range(99999):
            pos = (r2 == r0) | (r2 == r1)
            if not np.any(pos):
                break
            r2[pos] = np.floor(np.random.rand(np.sum(pos)) * NP_all).astype(int)

        r3 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(99999):
            pos = (r3 == r0) | (r3 == r1) | (r3 == r2)
            if not np.any(pos):
                break
            r3[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)

        return r2, r3

    def _han_boun(self, vi, lb, ub, parent, N):
        """han_boun with random case selection (hb = randi(3))."""
        hb = np.random.randint(1, 4)   # 1, 2, or 3
        if hb == 1:
            # Midpoint
            below = vi < lb
            above = vi > ub
            vi = np.where(below, (parent + lb) / 2, vi)
            vi = np.where(above, (parent + ub) / 2, vi)
        elif hb == 2:
            # Reflection clipped
            below = vi < lb
            above = vi > ub
            reflected = 2 * lb - parent
            vi = np.where(below, np.clip(reflected, lb, ub), vi)
            reflected = 2 * ub - parent
            vi = np.where(above, np.clip(reflected, lb, ub), vi)
        else:
            # Random resample
            out   = (vi < lb) | (vi > ub)
            rand_ = lb + np.random.rand(N, len(lb)) * (ub - lb)
            vi    = np.where(out, rand_, vi)
        return vi

    def _update_archive(self, archive_pop, archive_fit, new_pop, new_fit, arc_max):
        """updateArchive.m: append, deduplicate, trim."""
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
            rnd = np.random.permutation(len(pop_all))[:arc_max]
            pop_all = pop_all[rnd]
            fit_all = fit_all[rnd]
        return pop_all, fit_all
