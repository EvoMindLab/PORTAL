# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
SHADE — Success-History Based Parameter Adaptation for Differential Evolution
==============================================================================
Reference:
    Tanabe, R., & Fukunaga, A. (2013).
    Success-history based parameter adaptation for differential evolution.
    Proceedings of the 2013 IEEE Congress on Evolutionary Computation
    (CEC 2013), pp. 71–78.
    DOI: 10.1109/CEC.2013.6557555

Reference source:
    SHADE1.1.1_CEC2014_Matlab_Octave.zip — shade.m, boundConstraint.m,
    updateArchive.m, gnR1R2.m
    This PORTAL-specific Python adaptation follows SHADE 1.1.1, including its
    archive-update procedure.
    Original JADE utility functions by Jingqiao Zhang (boundConstraint.m,
    updateArchive.m, gnR1R2.m) are reused in SHADE.

Source credit and license:
    Ryoji Tanabe and Alex Fukunaga; original JADE utilities by Jingqiao Zhang.
    The author repository declares GPL v3; its license and source credits are
    retained in LICENSES/Tanabe-GPL-3.0.txt and LICENSES/Tanabe-NOTICE.txt.
    See algorithms/PROVENANCE.md for the exact source/version evidence.

Key mechanism:
    SHADE replaces JADE's single adaptive mean (mu_F, mu_CR) with a
    historical memory array of H=D entries, removing the need for a
    learning rate parameter c.

    HISTORICAL MEMORY (H = problem dimension D):
        memory_sf[1..H], memory_cr[1..H] — initialised to 0.5
        Special value -1 in memory_cr: "terminal CR" → CR=0 always

    PARAMETER SAMPLING (vectorised over NP, matching MATLAB):
        r = randint(H) per individual
        CR_i ~ N(memory_cr[r], 0.1), clipped [0,1]; 0 if r is terminal
        F_i  ~ Cauchy(memory_sf[r], 0.1), resampled if <=0, capped at 1

    MEMORY UPDATE (fitness-weighted Lehmer mean):
        goodF, goodCR = successful params where fitness > child_fitness
        weights = dif / sum(dif)  where dif = |parent_fit - child_fit|
        memory_sf[k]  = sum(w*F^2) / sum(w*F)   (weighted Lehmer)
        memory_cr[k]  = sum(w*CR^2)/ sum(w*CR)  (weighted Lehmer)
        Terminal case: if max(goodCR)==0 or current slot is -1: set -1

    ARCHIVE (size = arc_rate * NP = 2 * NP):
        Stores parents replaced by successful children.
        updateArchive: adds new entries, removes duplicates (unique),
        then randomly trims to max size using randperm.

    MUTATION: DE/current-to-pbest/1
        pNP = max(round(p * NP), 2)  (at least 2 pbest candidates)

    BOUNDARY: midpoint (boundConstraint.m — midpoint of parent and bound)

Parameters (exactly matching shade.m):
    pop_size : population size NP       (default: 100)
    p        : pbest fraction           (default: 0.1)
    arc_rate : archive size = arc_rate*NP (default: 2)
    H        : memory size (None = D)   (default: None → D)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class SHADE(BaseAlgorithm):
    """
    Success-History Based Parameter Adaptation for Differential Evolution.
    PORTAL Python adaptation of the SHADE 1.1.1 reference formulation.
    """

    def __init__(
        self,
        pop_size : int   = 100,   # population size (shade.m: 100)
        p        : float = 0.1,   # pbest fraction (shade.m: 0.1)
        arc_rate : float = 2.0,   # archive size = arc_rate * NP (shade.m: 2)
        H        : int   = None,  # memory size (shade.m: D); None → use D
    ):
        super().__init__(name='SHADE')
        self.pop_size = pop_size
        self.p        = p
        self.arc_rate = arc_rate
        self.H        = H

    def run(self, problem: PortalProblem) -> dict:
        """
        Run SHADE on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d   = problem.d
        lb  = problem.lb
        ub  = problem.ub

        # ── Algorithm parameters ──────────────────────────────────
        NP       = self.pop_size
        p        = self.p
        arc_max  = int(self.arc_rate * NP)   # archive max size
        H        = self.H if self.H is not None else d   # memory_size = D

        # pNP: at least 2 (MATLAB: max(round(p*NP), 2))
        pNP = max(int(round(p * NP)), 2)

        # ── Historical memory — initialised to 0.5 ────────────────
        memory_sf = np.full(H, 0.5)   # scaling factor memory
        memory_cr = np.full(H, 0.5)   # crossover rate memory
        memory_pos = 0                 # current write position (0-indexed)

        # ── Initialisation ────────────────────────────────────────
        popold = lb + np.random.rand(NP, d) * (ub - lb)
        pop    = popold.copy()
        fitness = np.array([problem.evaluate(pop[i]) for i in range(NP)])

        # Archive (initially empty)
        archive_pop = np.empty((0, d))
        archive_fit = np.empty(0)

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            pop = popold.copy()
            sorted_index = np.argsort(fitness)   # ascending: best first

            # ── Sample parameters from memory (vectorised) ────────
            mem_rand_index = np.random.randint(0, H, NP)
            mu_sf = memory_sf[mem_rand_index]   # shape (NP,)
            mu_cr = memory_cr[mem_rand_index]   # shape (NP,)

            # CR: N(mu_cr, 0.1), clipped [0,1]
            # Terminal slots (mu_cr == -1): CR = 0
            cr = np.random.normal(mu_cr, 0.1)
            term_pos = (mu_cr == -1)
            cr[term_pos] = 0.0
            cr = np.clip(cr, 0.0, 1.0)

            # F: Cauchy(mu_sf, 0.1), resample if <=0, cap at 1
            sf = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(NP) - 0.5))
            pos = sf <= 0
            while np.any(pos):
                sf[pos] = (mu_sf[pos]
                           + 0.1 * np.tan(np.pi * (
                               np.random.rand(np.sum(pos)) - 0.5)))
                pos = sf <= 0
            sf = np.minimum(sf, 1.0)

            # ── r1, r2 selection (gnR1R2 logic) ───────────────────
            pop_all = (np.vstack([pop, archive_pop])
                       if len(archive_pop) > 0 else pop)
            NP_all  = len(pop_all)

            r0 = np.arange(NP)
            r1 = self._gnR1(NP, r0)
            r2 = self._gnR2(NP_all, r1, r0)

            # ── pbest selection ───────────────────────────────────
            rand_idx   = np.floor(np.random.rand(NP) * pNP).astype(int)
            rand_idx   = np.maximum(rand_idx, 0)
            pbest_rows = sorted_index[rand_idx]
            pbest      = pop[pbest_rows]   # shape (NP, d)

            # ── Mutation: DE/current-to-pbest/1 ──────────────────
            sf_col = sf[:, np.newaxis]   # shape (NP, 1) for broadcasting
            vi = (pop
                  + sf_col * (pbest        - pop)
                  + sf_col * (pop[r1]      - pop_all[r2]))

            # ── Boundary: midpoint (boundConstraint.m) ────────────
            vi = self._bound_constraint(vi, pop, lb, ub)

            # ── Crossover: binomial ───────────────────────────────
            cr_col = cr[:, np.newaxis]
            mask   = np.random.rand(NP, d) > cr_col   # True → use parent
            # j_rand: force one dim from mutant per individual
            jrand         = np.floor(np.random.rand(NP) * d).astype(int)
            jrand_indices = (np.arange(NP), jrand)
            mask[jrand_indices] = False   # False → use mutant
            ui = vi.copy()
            ui[mask] = pop[mask]

            # ── Evaluate children ─────────────────────────────────
            children_fitness = np.array([
                problem.evaluate(ui[i]) if not problem.is_done()
                else fitness[i]
                for i in range(NP)
            ])

            # ── Compute fitness differences ───────────────────────
            dif = np.abs(fitness - children_fitness)

            # ── Identify successes (child strictly better than parent)
            I_better = fitness > children_fitness   # shape (NP,), bool

            good_CR  = cr[I_better]
            good_F   = sf[I_better]
            dif_val  = dif[I_better]

            # ── Update archive with replaced parents ───────────────
            archive_pop, archive_fit = self._update_archive(
                archive_pop, archive_fit,
                popold[I_better], fitness[I_better],
                arc_max
            )

            # ── Selection: greedy ─────────────────────────────────
            I_idx = np.argmin(
                np.stack([fitness, children_fitness], axis=1), axis=1
            )   # 0 = parent wins, 1 = child wins
            # Note: argmin returns 0 on ties → parent wins (strict <).
            # MATLAB shade.m also uses strict > for I_better, so this
            # matches the reference implementation rather than the paper
            # pseudocode which uses non-strict <=.
            fitness_new = np.minimum(fitness, children_fitness)
            popold_new  = popold.copy()
            popold_new[I_idx == 1] = ui[I_idx == 1]
            fitness = fitness_new
            popold  = popold_new

            # ── Update memory ─────────────────────────────────────
            num_success = np.sum(I_better)
            if num_success > 0:
                sum_dif = dif_val.sum()
                if sum_dif > 0:
                    w = dif_val / sum_dif
                else:
                    w = np.ones(num_success) / num_success

                # Weighted Lehmer mean for F
                memory_sf[memory_pos] = (
                    (w * good_F**2).sum() / ((w * good_F).sum() + 1e-300)
                )

                # Weighted Lehmer mean for CR (terminal logic)
                if np.max(good_CR) == 0 or memory_cr[memory_pos] == -1:
                    memory_cr[memory_pos] = -1
                else:
                    memory_cr[memory_pos] = (
                        (w * good_CR**2).sum()
                        / ((w * good_CR).sum() + 1e-300)
                    )

                memory_pos = (memory_pos + 1) % H

        return self._make_result(problem)

    # ── Helpers (translations of MATLAB utility functions) ────────

    def _gnR1(self, NP: int, r0: np.ndarray) -> np.ndarray:
        """
        Generate r1: each element from {0..NP-1}, r1[i] != r0[i].
        Translation of gnR1R2.m (r1 part).
        """
        r1 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(99999):
            pos = (r1 == r0)
            if not np.any(pos):
                break
            r1[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)
        return r1

    def _gnR2(self, NP_all: int, r1: np.ndarray,
              r0: np.ndarray) -> np.ndarray:
        """
        Generate r2: each element from {0..NP_all-1}, r2[i] != r1[i], r0[i].
        Translation of gnR1R2.m (r2 part).
        """
        NP = len(r0)
        r2 = np.floor(np.random.rand(NP) * NP_all).astype(int)
        for _ in range(99999):
            pos = (r2 == r1) | (r2 == r0)
            if not np.any(pos):
                break
            r2[pos] = np.floor(
                np.random.rand(np.sum(pos)) * NP_all
            ).astype(int)
        return r2

    def _bound_constraint(self, vi: np.ndarray, pop: np.ndarray,
                          lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """
        Midpoint boundary constraint (boundConstraint.m).
        vi_j = (parent_j + lb_j) / 2  if vi_j < lb_j
        vi_j = (parent_j + ub_j) / 2  if vi_j > ub_j
        """
        below = vi < lb
        above = vi > ub
        vi = np.where(below, (pop + lb) / 2, vi)
        vi = np.where(above, (pop + ub) / 2, vi)
        return vi

    def _update_archive(self, archive_pop, archive_fit,
                        new_pop, new_fit, arc_max):
        """
        Update archive (updateArchive.m):
        1. Append new solutions
        2. Remove duplicates (unique rows)
        3. If over limit, randomly trim with randperm
        """
        if arc_max == 0 or len(new_pop) == 0:
            return archive_pop, archive_fit

        # Append
        if len(archive_pop) > 0:
            pop_all = np.vstack([archive_pop, new_pop])
            fit_all = np.concatenate([archive_fit, new_fit])
        else:
            pop_all = new_pop.copy()
            fit_all = new_fit.copy()

        # Remove duplicates (unique rows)
        _, IX = np.unique(pop_all, axis=0, return_index=True)
        IX = np.sort(IX)
        pop_all = pop_all[IX]
        fit_all = fit_all[IX]

        # Trim to max size using random permutation
        if len(pop_all) > arc_max:
            rnd = np.random.permutation(len(pop_all))[:arc_max]
            pop_all = pop_all[rnd]
            fit_all = fit_all[rnd]

        return pop_all, fit_all
