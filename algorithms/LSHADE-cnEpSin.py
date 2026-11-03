# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
LSHADE-cnEpSin — Ensemble Sinusoidal DE with Covariance Matrix Adaptation
==========================================================================
Reference:
    Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2017).
    Ensemble sinusoidal differential covariance matrix adaptation with
    Euclidean neighbourhood for solving CEC2017 benchmark problems.
    Proceedings of the 2017 IEEE Congress on Evolutionary Computation
    (CEC 2017), pp. 372–379.
    DOI: 10.1109/CEC.2017.7969336

    Joint CEC2016 winner (with LSHADE-EpSin predecessor).

Reference source:
    LSHADE-cnEpSin_matlab.rar — LSHADE_cnEpSin.m
    The reference code was consulted for clarification.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    LSHADE-cnEpSin = L-SHADE + Ensemble Sinusoidal F (EpSin)
                                + Covariance Matrix Crossover (cn)

    ENSEMBLE SINUSOIDAL F ADAPTATION (EpSin):
        Two sinusoidal strategies for scaling factor F,
        selected by SaDE-style probability after LP=20 learning generations.
        Active only in the FIRST HALF of the run (nfes ≤ max_nfes/2).
        In the second half, standard Cauchy F from memory is used.

        Strategy 1 (decreasing amplitude, fixed frequency freq_init=0.5):
            sf = 0.5*(sin(2π·freq_init·gg + π)·((G_Max-gg)/G_Max) + 1)
            Amplitude shrinks → F moves toward 1 from oscillation

        Strategy 2 (increasing amplitude, adaptive frequency):
            sf = 0.5*(sin(2π·freq·gg)·(gg/G_Max) + 1)
            freq ~ Cauchy(mu_freq, 0.1), stored in memory_freq
            Amplitude grows → F explores more as run progresses

        After LP generations, strategy probabilities computed from
        sliding window of success/failure counts (same as SaDE).

    COVARIANCE MATRIX CROSSOVER (cn), probability pb=0.4:
        Each generation: with prob pb, rotate to Eigen coordinate system
        defined by the covariance matrix of SEL nearest neighbours of best.
            1. Find SEL = round(ps*NP) individuals nearest to best by Euclidean
            2. Compute covariance C of neighbourhood
            3. Eigendecompose [R,D] = eig(C); rotation matrix TM = R
            4. Rotate pop → Xr = pop·TM; rotate mutant → vi_r = vi·TM
            5. Crossover in rotated space
            6. Rotate back: ui = Ur·TM.T
        Otherwise (prob 1-pb=0.6): standard binomial crossover.

    All other mechanisms identical to L-SHADE 1.1.1:
        - DE/current-to-pbest/1 mutation with archive
        - Midpoint boundary (boundConstraint.m)
        - Linear population size reduction
        - updateArchive (deduplicate + randperm trim)
        - Weighted Lehmer memory update for sf, cr, freq
        - Terminal CR sentinel (-1)

Parameters (exactly matching LSHADE_cnEpSin.m):
    pop_rate  : pop = pop_rate * D     (default: 18)
    min_pop   : minimum population     (default: 4)
    H         : memory size            (default: 5)
    arc_rate  : archive multiplier     (default: 1.4)
    p         : pbest fraction         (default: 0.11)
    pb        : Eigen crossover prob   (default: 0.4)
    ps        : neighbourhood fraction (default: 0.5)
    freq_init : initial/fixed freq S1  (default: 0.5)
    LP        : learning period gens   (default: 20)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class LSHADE_cnEpSin(BaseAlgorithm):
    """
    LSHADE-cnEpSin: L-SHADE with Ensemble Sinusoidal F and Covariance
    Matrix crossover; PORTAL implementation informed by LSHADE_cnEpSin.m.
    """

    def __init__(
        self,
        pop_rate  : int   = 18,    # pop = pop_rate * D
        min_pop   : int   = 4,
        H         : int   = 5,     # memory size
        arc_rate  : float = 1.4,
        p         : float = 0.11,  # pbest fraction
        pb        : float = 0.4,   # Eigen crossover probability
        ps        : float = 0.5,   # neighbourhood fraction
        freq_init : float = 0.5,   # fixed frequency for strategy 1
        LP        : int   = 20,    # learning period (generations)
    ):
        super().__init__(name='LSHADE-cnEpSin')
        self.pop_rate  = pop_rate
        self.min_pop   = min_pop
        self.H         = H
        self.arc_rate  = arc_rate
        self.p         = p
        self.pb        = pb
        self.ps        = ps
        self.freq_init = freq_init
        self.LP        = LP

    def run(self, problem: PortalProblem) -> dict:
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        max_pop  = self.pop_rate * d
        min_pop  = self.min_pop
        pop_size = max_pop
        H        = self.H
        arc_rate = self.arc_rate
        p        = self.p
        pb       = self.pb
        ps       = self.ps
        freq_init = self.freq_init
        LP        = self.LP

        if max_fe < pop_size:
            raise ValueError(
                f"LSHADE-cnEpSin needs at least {pop_size} evaluations for "
                f"its initial population, but the supplied budget is {max_fe}."
            )

        # G_Max: approximate max generations (used in sin formula)
        G_Max = max(1, int(max_fe / pop_size))

        # ── Memory — all init 0.5 ─────────────────────────────────
        memory_sf   = np.full(H, 0.5)
        memory_cr   = np.full(H, 0.5)
        memory_freq = np.full(H, 0.5)
        memory_pos  = 0

        # ── Initialisation ────────────────────────────────────────
        popold  = lb + np.random.rand(pop_size, d) * (ub - lb)
        pop     = popold.copy()
        fitness = np.array([problem.evaluate(pop[i]) for i in range(pop_size)])
        nfes    = pop_size

        arc_max     = int(round(arc_rate * pop_size))
        archive_pop = np.empty((0, d))
        archive_fit = np.empty(0)

        # EpSin tracking arrays
        goodF1all = []   # successes for strategy 1 per generation
        goodF2all = []
        badF1all  = []
        badF2all  = []

        gg = 0   # generation counter for sin formula

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():
            gg += 1
            pop = popold.copy()

            sorted_index = np.argsort(fitness)

            # ── Sample CR and freq from memory ────────────────────
            mem_idx  = np.random.randint(0, H, pop_size)
            mu_cr    = memory_cr[mem_idx]
            mu_freq  = memory_freq[mem_idx]
            mu_sf    = memory_sf[mem_idx]

            # CR ~ N(mu_cr, 0.1), terminal -1 → 0
            cr = np.random.normal(mu_cr, 0.1)
            term = (mu_cr == -1)
            cr[term] = 0.0
            cr = np.clip(cr, 0.0, 1.0)

            # freq ~ Cauchy(mu_freq, 0.1)
            freq = np.zeros(pop_size)
            for i in range(pop_size):
                f = 0.0
                while f <= 0.0:
                    f = mu_freq[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                freq[i] = min(f, 1.0)

            # ── Sinusoidal F strategy selection ───────────────────
            flag1 = False
            flag2 = False

            if nfes <= max_fe / 2:
                # First half: use sinusoidal F
                if gg <= LP:
                    # Learning period: random 50/50
                    if np.random.rand() < 0.5:
                        flag1 = True
                    else:
                        flag2 = True
                else:
                    # SaDE-style probability from sliding window
                    ns1 = sum(goodF1all[max(0, gg-1-LP):gg-1])
                    nf1 = sum(badF1all[max(0, gg-1-LP):gg-1])
                    ns2 = sum(goodF2all[max(0, gg-1-LP):gg-1])
                    nf2 = sum(badF2all[max(0, gg-1-LP):gg-1])
                    sumS1 = ns1 / (ns1 + nf1 + 1e-300) + 0.01
                    sumS2 = ns2 / (ns2 + nf2 + 1e-300) + 0.01
                    if sumS1 >= sumS2:
                        flag1 = True
                    else:
                        flag2 = True

            if flag1:
                # Strategy 1: decreasing amplitude, fixed freq
                amp = np.sin(2 * np.pi * freq_init * gg + np.pi) * \
                      ((G_Max - gg) / G_Max)
                sf_val = 0.5 * (amp + 1.0)
                sf = np.full((pop_size, d), sf_val)
            elif flag2:
                # Strategy 2: increasing amplitude, adaptive freq
                amp = np.sin(2 * np.pi * freq[:, np.newaxis] * gg) * \
                      (gg / G_Max)
                sf = 0.5 * (amp + 1.0)
            else:
                # Second half: Cauchy F from memory
                sf_1d = np.zeros(pop_size)
                for i in range(pop_size):
                    f = 0.0
                    while f <= 0.0:
                        f = mu_sf[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                    sf_1d[i] = min(f, 1.0)
                sf = sf_1d[:, np.newaxis] * np.ones((pop_size, d))

            # ── r1, r2 selection ──────────────────────────────────
            pop_all = (np.vstack([pop, archive_pop])
                       if len(archive_pop) > 0 else pop)
            NP_all = len(pop_all)
            r0 = np.arange(pop_size)
            r1 = self._gnR1(pop_size, r0)
            r2 = self._gnR2(NP_all, r1, r0)

            # ── pbest selection ───────────────────────────────────
            pNP = max(2, int(round(p * pop_size)))
            rand_idx   = np.floor(np.random.rand(pop_size) * pNP).astype(int)
            rand_idx   = np.maximum(rand_idx, 0)
            pbest_rows = sorted_index[rand_idx]
            pbest      = pop[pbest_rows]

            # ── Mutation ──────────────────────────────────────────
            vi = (pop
                  + sf * (pbest     - pop)
                  + sf * (pop[r1]   - pop_all[r2]))

            # ── Boundary: midpoint (boundConstraint.m) ────────────
            below = vi < lb
            above = vi > ub
            vi = np.where(below, (pop + lb) / 2, vi)
            vi = np.where(above, (pop + ub) / 2, vi)

            # ── Crossover ─────────────────────────────────────────
            # jrand (J in MATLAB)
            j_rand    = np.random.randint(0, d, pop_size)
            cr_mask   = np.random.rand(pop_size, d) < cr[:, np.newaxis]
            # Ensure jrand dim is always included
            cr_mask[np.arange(pop_size), j_rand] = True

            if np.random.rand() < pb:
                # Type A: Eigen/covariance crossover
                # Neighbourhood of best (SEL nearest by Euclidean)
                SEL = max(2, int(round(ps * pop_size)))
                best = pop[sorted_index[0]]
                dists = np.sqrt(np.sum((pop - best)**2, axis=1))
                neigh_idx = np.argsort(dists)[:SEL]
                X_sel = pop[neigh_idx]   # shape (SEL, d)

                # Covariance matrix
                x_mean = X_sel.mean(axis=0)
                X_c = X_sel - x_mean
                C   = (X_c.T @ X_c) / (SEL - 1)
                C   = np.triu(C) + np.triu(C, 1).T   # enforce symmetry

                # Condition limit 1e20
                eigvals, R = np.linalg.eigh(C)
                if eigvals.max() > 1e20 * eigvals.min():
                    tmp = eigvals.max() / 1e20 - eigvals.min()
                    C   = C + tmp * np.eye(d)
                    _, R = np.linalg.eigh(C)

                TM  = R       # rotation matrix
                TM_ = R.T     # inverse rotation

                # Rotate to Eigen space
                Xr = pop @ TM    # (NP, d)
                vi_r = vi @ TM

                # Crossover in rotated space
                Ur = Xr.copy()
                Ur[cr_mask] = vi_r[cr_mask]

                # Rotate back
                ui = Ur @ TM_
                # Apply midpoint boundary to rotated-back ui
                # (rotation can push solutions outside bounds)
                below = ui < lb
                above = ui > ub
                ui = np.where(below, (pop + lb) / 2, ui)
                ui = np.where(above, (pop + ub) / 2, ui)

            else:
                # Type B: standard binomial
                ui = pop.copy()
                ui[cr_mask] = vi[cr_mask]

            # ── Evaluate ──────────────────────────────────────────
            children_fitness = np.array([
                problem.evaluate(ui[i]) if not problem.is_done()
                else fitness[i]
                for i in range(pop_size)
            ])
            nfes += pop_size

            dif = np.abs(fitness - children_fitness)

            # ── Identify successes ────────────────────────────────
            I_better  = fitness > children_fitness   # strict <
            good_CR   = cr[I_better]
            good_sf1d = sf[:, 0][I_better]   # first col is representative
            good_freq = freq[I_better]
            dif_val   = dif[I_better]

            bad_sf = sf[:, 0][~I_better]

            # ── EpSin strategy tracking ───────────────────────────
            if flag1:
                goodF1all.append(np.sum(I_better))
                badF1all.append(np.sum(~I_better))
                goodF2all.append(1)
                badF2all.append(1)
            elif flag2:
                goodF2all.append(np.sum(I_better))
                badF2all.append(np.sum(~I_better))
                goodF1all.append(1)
                badF1all.append(1)
            else:
                goodF1all.append(1)
                badF1all.append(1)
                goodF2all.append(1)
                badF2all.append(1)

            # ── Archive update ────────────────────────────────────
            archive_pop, archive_fit = self._update_archive(
                archive_pop, archive_fit,
                popold[I_better], fitness[I_better],
                arc_max
            )

            # ── Selection ─────────────────────────────────────────
            I_idx   = np.argmin(np.stack([fitness, children_fitness], axis=1), axis=1)
            fitness = np.minimum(fitness, children_fitness)
            popold  = pop.copy()
            popold[I_idx == 1] = ui[I_idx == 1]

            # ── Memory update ─────────────────────────────────────
            if len(good_sf1d) > 0:
                sum_dif = dif_val.sum()
                w = dif_val / (sum_dif + 1e-300)

                memory_sf[memory_pos] = (
                    (w * good_sf1d**2).sum() /
                    ((w * good_sf1d).sum() + 1e-300)
                )

                if np.max(good_CR) == 0 or memory_cr[memory_pos] == -1:
                    memory_cr[memory_pos] = -1
                else:
                    memory_cr[memory_pos] = (
                        (w * good_CR**2).sum() /
                        ((w * good_CR).sum() + 1e-300)
                    )

                if np.max(good_freq) == 0 or memory_freq[memory_pos] == -1:
                    memory_freq[memory_pos] = -1
                else:
                    memory_freq[memory_pos] = (
                        (w * good_freq**2).sum() /
                        ((w * good_freq).sum() + 1e-300)
                    )

                memory_pos = (memory_pos + 1) % H

            # ── Linear population size reduction ──────────────────
            plan = max(min_pop,
                       int(round(((min_pop - max_pop) / max_fe) * nfes + max_pop)))

            if pop_size > plan:
                reduction = min(pop_size - plan, pop_size - min_pop)
                if reduction > 0:
                    worst = np.argsort(fitness)[::-1][:reduction]
                    keep  = np.ones(pop_size, dtype=bool)
                    keep[worst] = False
                    popold   = popold[keep]
                    fitness  = fitness[keep]
                    pop_size = int(np.sum(keep))
                    SEL      = max(2, int(round(ps * pop_size)))

                    arc_max = int(round(arc_rate * pop_size))
                    if len(archive_pop) > arc_max:
                        rnd = np.random.permutation(len(archive_pop))[:arc_max]
                        archive_pop = archive_pop[rnd]
                        archive_fit = archive_fit[rnd]

        return self._make_result(problem)

    # ── Helpers ──────────────────────────────────────────────────

    def _gnR1(self, NP, r0):
        r1 = np.floor(np.random.rand(NP) * NP).astype(int)
        for _ in range(99999):
            pos = (r1 == r0)
            if not np.any(pos):
                break
            r1[pos] = np.floor(np.random.rand(np.sum(pos)) * NP).astype(int)
        return r1

    def _gnR2(self, NP_all, r1, r0):
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
