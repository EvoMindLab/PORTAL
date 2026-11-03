# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
L-SHADE — Improving the Search Performance of SHADE Using Linear
          Population Size Reduction
=================================================================
Reference:
    Tanabe, R., & Fukunaga, A. (2014).
    Improving the search performance of SHADE using linear population
    size reduction.
    Proceedings of the 2014 IEEE Congress on Evolutionary Computation
    (CEC 2014), pp. 1658–1665.
    DOI: 10.1109/CEC.2014.6900380

Reference source:
    LSHADE1.0.1_CEC2014_Octave-Matlab.zip — lshade.m
    This PORTAL-specific Python adaptation follows lshade.m (fixed 1.0.1 version).
    The utility functions boundConstraint.m, updateArchive.m, gnR1R2.m
    are identical to those in SHADE 1.1.1 and are reused from the
    SHADE implementation.

Source credit and license:
    Ryoji Tanabe and Alex Fukunaga; original JADE utilities by Jingqiao Zhang.
    The author repository declares GPL v3; its license and source credits are
    retained in LICENSES/Tanabe-GPL-3.0.txt and LICENSES/Tanabe-NOTICE.txt.
    See algorithms/PROVENANCE.md for the exact source/version evidence.

Key mechanism:
    L-SHADE = SHADE + Linear Population Size Reduction (LPSR).

    All mechanisms are identical to SHADE 1.1.1 except:

    LINEAR POPULATION SIZE REDUCTION (lshade.m lines ~90-110):
        max_pop_size = 18 * D  (initial population)
        min_pop_size = 4       (minimum population, never goes below)
        After each generation's memory update:
            plan = round(((min - max) / max_nfes) * nfes + max)
            if current_pop > plan:
                remove (current_pop - plan) WORST individuals
                (worst = highest fitness in minimisation)
                archive max size also shrinks: round(arc_rate * new_pop)
                if archive exceeds new max: randomly trim with randperm

    This linearly decreases the population from 18*D to 4 over the
    full budget, focusing resources on exploitation as the search matures.
    CEC 2014 winner.

Parameters (exactly matching lshade.m):
    p        : pbest fraction           (default: 0.11)
    arc_rate : archive size multiplier  (default: 1.4)
    H        : memory size              (default: 5, fixed in lshade.m)
    min_pop  : minimum population size  (default: 4)
    init_pop_rate : initial pop = rate*D(default: 18)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class L_SHADE(BaseAlgorithm):
    """
    L-SHADE: SHADE with Linear Population Size Reduction.
    PORTAL Python adaptation of lshade.m (version 1.0.1).
    CEC 2014 winner.
    """

    def __init__(
        self,
        p             : float = 0.11,  # pbest fraction (lshade.m: 0.11)
        arc_rate      : float = 1.4,   # archive = arc_rate * NP (lshade.m: 1.4)
        H             : int   = 5,     # memory size (lshade.m: 5, fixed)
        min_pop       : int   = 4,     # minimum population size
        init_pop_rate : int   = 18,    # initial pop = rate * D
    ):
        super().__init__(name='L-SHADE')
        self.p             = p
        self.arc_rate      = arc_rate
        self.H             = H
        self.min_pop       = min_pop
        self.init_pop_rate = init_pop_rate

    def run(self, problem: PortalProblem) -> dict:
        """
        Run L-SHADE on a PORTAL problem.

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
        p        = self.p
        arc_rate = self.arc_rate
        H        = self.H
        min_pop  = max(self.min_pop, 4)

        # Initial and minimum population sizes
        if problem.remaining_fe() < min_pop:
            while not problem.is_done():
                problem.evaluate(problem.random_solution())
            return self._make_result(problem)

        max_pop_size = min(self.init_pop_rate * d, problem.remaining_fe())
        pop_size     = max_pop_size

        # ── Historical memory — initialised to 0.5 ────────────────
        memory_sf  = np.full(H, 0.5)
        memory_cr  = np.full(H, 0.5)
        memory_pos = 0   # 0-indexed

        # ── Initialisation ────────────────────────────────────────
        popold = lb + np.random.rand(pop_size, d) * (ub - lb)
        pop    = popold.copy()
        fitness = problem.evaluate_batch(pop)

        # Archive
        arc_max     = int(round(arc_rate * pop_size))
        archive_pop = np.empty((0, d))
        archive_fit = np.empty(0)

        # nfes tracks evaluations for LPSR plan; use problem.fe_count

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            pop = popold.copy()
            sorted_index = np.argsort(fitness)   # best first

            # ── Sample parameters from memory ─────────────────────
            mem_rand_index = np.random.randint(0, H, pop_size)
            mu_sf = memory_sf[mem_rand_index]
            mu_cr = memory_cr[mem_rand_index]

            # CR ~ N(mu_cr, 0.1), terminal if mu_cr == -1
            cr = np.random.normal(mu_cr, 0.1)
            term_pos = (mu_cr == -1)
            cr[term_pos] = 0.0
            cr = np.clip(cr, 0.0, 1.0)

            # F ~ Cauchy(mu_sf, 0.1), resample if <=0, cap at 1
            sf = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            pos = sf <= 0
            while np.any(pos):
                sf[pos] = (mu_sf[pos]
                           + 0.1 * np.tan(np.pi * (
                               np.random.rand(np.sum(pos)) - 0.5)))
                pos = sf <= 0
            sf = np.minimum(sf, 1.0)

            # ── r1, r2 selection ──────────────────────────────────
            pop_all = (np.vstack([pop, archive_pop])
                       if len(archive_pop) > 0 else pop)
            NP_all  = len(pop_all)

            r0 = np.arange(pop_size)
            r1 = self._gnR1(pop_size, r0)
            r2 = self._gnR2(NP_all, r1, r0)

            # ── pbest selection ───────────────────────────────────
            pNP = max(int(round(p * pop_size)), 2)
            rand_idx   = np.floor(np.random.rand(pop_size) * pNP).astype(int)
            rand_idx   = np.maximum(rand_idx, 0)
            pbest_rows = sorted_index[rand_idx]
            pbest      = pop[pbest_rows]

            # ── Mutation ──────────────────────────────────────────
            sf_col = sf[:, np.newaxis]
            vi = (pop
                  + sf_col * (pbest     - pop)
                  + sf_col * (pop[r1]   - pop_all[r2]))

            # ── Boundary: midpoint ────────────────────────────────
            vi = self._bound_constraint(vi, pop, lb, ub)

            # ── Crossover ─────────────────────────────────────────
            cr_col = cr[:, np.newaxis]
            mask   = np.random.rand(pop_size, d) > cr_col
            jrand  = np.floor(np.random.rand(pop_size) * d).astype(int)
            mask[np.arange(pop_size), jrand] = False
            ui = vi.copy()
            ui[mask] = pop[mask]

            # ── Evaluate ──────────────────────────────────────────
            evaluated = problem.evaluate_batch(ui)
            children_fitness = fitness.copy()
            children_fitness[:len(evaluated)] = evaluated
            nfes = problem.fe_count   # use actual count, not estimate

            # ── Compute differences ───────────────────────────────
            dif = np.abs(fitness - children_fitness)
            I_better = fitness > children_fitness

            good_CR  = cr[I_better]
            good_F   = sf[I_better]
            dif_val  = dif[I_better]

            # ── Update archive ────────────────────────────────────
            archive_pop, archive_fit = self._update_archive(
                archive_pop, archive_fit,
                popold[I_better], fitness[I_better],
                arc_max
            )

            # ── Selection ─────────────────────────────────────────
            I_idx       = np.argmin(np.stack([fitness, children_fitness], axis=1), axis=1)
            fitness     = np.minimum(fitness, children_fitness)
            popold      = pop.copy()
            popold[I_idx == 1] = ui[I_idx == 1]

            # ── Update memory ─────────────────────────────────────
            if len(good_F) > 0:
                sum_dif = dif_val.sum()
                w = dif_val / (sum_dif + 1e-300)

                memory_sf[memory_pos] = (
                    (w * good_F**2).sum() / ((w * good_F).sum() + 1e-300)
                )

                if np.max(good_CR) == 0 or memory_cr[memory_pos] == -1:
                    memory_cr[memory_pos] = -1
                else:
                    memory_cr[memory_pos] = (
                        (w * good_CR**2).sum()
                        / ((w * good_CR).sum() + 1e-300)
                    )

                memory_pos = (memory_pos + 1) % H

            # ── Linear Population Size Reduction (LPSR) ───────────
            # plan_pop_size decreases linearly from max to min
            plan_pop_size = int(round(
                ((min_pop - max_pop_size) / max_fe) * problem.fe_count + max_pop_size
            ))

            if pop_size > plan_pop_size:
                reduction = pop_size - plan_pop_size
                # Safety: never go below min_pop
                reduction = min(reduction, pop_size - min_pop)

                if reduction > 0:
                    # Remove worst individuals (highest fitness)
                    worst_sorted = np.argsort(fitness)[::-1]   # worst first
                    remove_idx   = worst_sorted[:reduction]

                    # Keep mask
                    keep = np.ones(pop_size, dtype=bool)
                    keep[remove_idx] = False

                    popold  = popold[keep]
                    fitness = fitness[keep]
                    pop_size = int(np.sum(keep))

                    # Archive also shrinks
                    arc_max = int(round(arc_rate * pop_size))
                    if len(archive_pop) > arc_max:
                        rnd = np.random.permutation(len(archive_pop))[:arc_max]
                        archive_pop = archive_pop[rnd]
                        archive_fit = archive_fit[rnd]

        return self._make_result(problem)

    # ── Helpers (same as SHADE) ───────────────────────────────────

    def _gnR1(self, NP: int, r0: np.ndarray) -> np.ndarray:
        r1 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(99999):
            pos = (r1 == r0)
            if not np.any(pos):
                break
            r1[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)
        return r1

    def _gnR2(self, NP_all: int, r1: np.ndarray, r0: np.ndarray) -> np.ndarray:
        NP = len(r0)
        r2 = np.floor(np.random.rand(NP) * NP_all).astype(int)
        for _ in range(99999):
            pos = (r2 == r1) | (r2 == r0)
            if not np.any(pos):
                break
            r2[pos] = np.floor(np.random.rand(np.sum(pos)) * NP_all).astype(int)
        return r2

    def _bound_constraint(self, vi, pop, lb, ub):
        below = vi < lb
        above = vi > ub
        vi = np.where(below, (pop + lb) / 2, vi)
        vi = np.where(above, (pop + ub) / 2, vi)
        return vi

    def _update_archive(self, archive_pop, archive_fit, new_pop, new_fit, arc_max):
        if arc_max == 0 or len(new_pop) == 0:
            return archive_pop, archive_fit
        if len(archive_pop) > 0:
            pop_all = np.vstack([archive_pop, new_pop])
            fit_all = np.concatenate([archive_fit, new_fit])
        else:
            pop_all = new_pop.copy()
            fit_all = new_fit.copy()
        _, IX = np.unique(pop_all, axis=0, return_index=True)
        IX = np.sort(IX)
        pop_all = pop_all[IX]
        fit_all = fit_all[IX]
        if len(pop_all) > arc_max:
            rnd = np.random.permutation(len(pop_all))[:arc_max]
            pop_all = pop_all[rnd]
            fit_all = fit_all[rnd]
        return pop_all, fit_all
