# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
SaDE — Self-Adaptive Differential Evolution with Strategy Adaptation
=====================================================================
Reference:
    Qin, A. K., Huang, V. L., & Suganthan, P. N. (2009).
    Differential evolution algorithm with strategy adaptation for
    global numerical optimization.
    IEEE Transactions on Evolutionary Computation, 13(2), 398–417.
    DOI: 10.1109/TEVC.2007.914537

Reference source:
    https://github.com/P-N-Suganthan/CODES
    File: SaDE.m (2009-IEEE-TEC-SaDE.zip)
    The reference code was consulted for clarification.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    SaDE adapts both the mutation strategy and the CR parameter by
    learning from past successes. F is sampled from N(0.5, 0.3)
    per strategy per generation (not fixed — from MATLAB code).

    FOUR STRATEGIES (from SaDE.m):
        S1: DE/rand/1/bin
            v = pm3 + F*(pm1 - pm2)
        S2: DE/current-to-best/2/bin
            v = x + F*(best-x) + F*(pm1-pm2) + F*(pm3-pm4)
        S3: DE/rand/2/bin
            v = pm5 + F*(pm1-pm2) + F*(pm3-pm4)
        S4: DE/current-to-rand/1 (no crossover except jrand)
            v = x + rand*(pm5-x) + F*(pm1-pm2)
            Note: first coefficient is scalar rand, not F

    STRATEGY SELECTION: Stochastic Universal Sampling (SUS)
        Allocates exactly floor/ceil(p_k * NP) individuals per strategy,
        then shuffles assignments randomly. More structured than simple
        roulette wheel, reducing variance in strategy allocation.

    STRATEGY PROBABILITY UPDATE (after learngen=50 generations):
        Sliding window of learngen generations of ns/nf counts:
            p_k = ns_k / (ns_k + nf_k) + 0.01
        Oldest generation's counts are dropped each generation.

    CR ADAPTATION (after learngen=50 generations):
        Each strategy k maintains aaaa{k}: a list of [CR, gen] pairs
        for all successful CRs. Each generation, entries from the
        oldest generation in the window are removed (sliding window).
        CRm_k = median of remaining successful CRs for strategy k.
        If no successes exist: CRm_k = random.
        For each individual: CR ~ N(CRm_k, 0.1) clipped to [0, 1].

    F ADAPTATION (from MATLAB, not fixed as paper text implies):
        Per strategy per generation: F ~ N(0.5, 0.3)

Parameters (exactly matching SaDE.m):
    pop_size  : population size NP   (default: 50)
    learngen  : learning period      (default: 50 generations)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class SaDE(BaseAlgorithm):
    """
    Self-Adaptive Differential Evolution.
    PORTAL Python implementation informed by the SaDE.m reference.
    Adapts strategy probabilities and CR from success history.
    F sampled from N(0.5, 0.3) per strategy per generation.
    """

    def __init__(
        self,
        pop_size : int = 50,    # population size (SaDE.m default: 50)
        learngen : int = 50,    # learning period in generations
    ):
        super().__init__(name='SaDE')
        self.pop_size = pop_size
        self.learngen = learngen

    def run(self, problem: PortalProblem) -> dict:
        """
        Run SaDE on a PORTAL problem.

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
        learngen = self.learngen
        K        = 4       # number of strategies
        CR_init  = 0.5     # initial CR median per strategy

        # ── Initialisation ────────────────────────────────────────
        pop = lb + np.random.rand(NP, d) * (ub - lb)
        fit = np.array([problem.evaluate(pop[i]) for i in range(NP)])

        best_idx = int(np.argmin(fit))

        # Strategy probabilities — equal at start
        pfit = np.ones(K)   # unnormalised; normalised when used

        # CR medians per strategy — initialised to CR_init
        ccm = np.full(K, CR_init)

        # aaaa{k}: list of [CR_value, generation_number] for strategy k
        aaaa = [[] for _ in range(K)]   # list of lists of [CR, gen]

        # ns, nf: sliding window arrays — shape (learngen, K) at most
        # MATLAB: grows by one row per gen, drops oldest when >= learngen
        ns_win = []   # list of shape-(K,) arrays
        nf_win = []

        gen = 0

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():
            gen += 1
            popold = pop.copy()

            # ── Build shuffled donor pools (MATLAB vectorised style)
            # pm1..pm5 are shuffled versions of popold
            a1 = np.random.permutation(NP)
            ind = np.random.permutation(4)
            a2 = a1[(np.arange(NP) + ind[0]) % NP]
            a3 = a2[(np.arange(NP) + ind[1]) % NP]
            a4 = a3[(np.arange(NP) + ind[2]) % NP]
            a5 = a4[(np.arange(NP) + ind[3]) % NP]

            pm1 = popold[a1]
            pm2 = popold[a2]
            pm3 = popold[a3]
            pm4 = popold[a4]
            pm5 = popold[a5]
            bm  = np.tile(popold[best_idx], (NP, 1))

            # ── CR update (after learngen, every generation) ──────
            if gen >= learngen:
                for k in range(K):
                    # Remove entries from the oldest generation in window
                    if len(aaaa[k]) > 0:
                        oldest_gen = aaaa[k][0][1]
                        aaaa[k] = [e for e in aaaa[k] if e[1] != oldest_gen]
                    if len(aaaa[k]) > 0:
                        ccm[k] = float(np.median([e[0] for e in aaaa[k]]))
                    else:
                        ccm[k] = float(np.random.rand())

            # ── Sample CR per strategy per individual ─────────────
            # cc shape: (NP, K) — each column is CR for one strategy
            cc = np.zeros((NP, K))
            for k in range(K):
                for i in range(NP):
                    tt = np.random.normal(ccm[k], 0.1)
                    while tt > 1 or tt < 0:
                        tt = np.random.normal(ccm[k], 0.1)
                    cc[i, k] = tt

            # ── Strategy assignment: Stochastic Universal Sampling ─
            stpool = self._sus(pfit, NP)   # shape (NP,), values in 0..K-1

            # Build index sets per strategy
            index = [np.where(stpool == k)[0] for k in range(K)]
            # aaa[k][i] = 1 if particle i uses strategy k
            aaa = [np.zeros(NP) for _ in range(K)]
            for k in range(K):
                aaa[k][index[k]] = 1

            # ── Build crossover masks (binomial, j_rand forced) ───
            # Each particle uses the CR of its assigned strategy
            cr_per_particle = np.array([cc[i, stpool[i]] for i in range(NP)])
            mui = np.random.rand(NP, d) < cr_per_particle[:, np.newaxis]
            # j_rand: force at least one dimension from mutant
            jrand = np.random.randint(0, d, NP)
            for i in range(NP):
                mui[i, jrand[i]] = True
            mpo = ~mui   # inverse mask

            # ── Generate trial vectors ────────────────────────────
            ui = np.zeros((NP, d))

            # Strategy 1: DE/rand/1/bin — base=pm3, diff=pm1-pm2
            if len(index[0]) > 0:
                idx = index[0]
                F1 = np.random.normal(0.5, 0.3, (len(idx), 1))
                F1 = np.tile(F1, (1, d))
                v  = pm3[idx] + F1 * (pm1[idx] - pm2[idx])
                ui[idx] = (popold[idx] * mpo[idx] + v * mui[idx])

            # Strategy 2: DE/current-to-best/2/bin
            if len(index[1]) > 0:
                idx = index[1]
                F2 = np.random.normal(0.5, 0.3, (len(idx), 1))
                F2 = np.tile(F2, (1, d))
                v  = (popold[idx]
                      + F2 * (bm[idx]  - popold[idx])
                      + F2 * (pm1[idx] - pm2[idx])
                      + F2 * (pm3[idx] - pm4[idx]))
                ui[idx] = (popold[idx] * mpo[idx] + v * mui[idx])

            # Strategy 3: DE/rand/2/bin — base=pm5
            if len(index[2]) > 0:
                idx = index[2]
                F3 = np.random.normal(0.5, 0.3, (len(idx), 1))
                F3 = np.tile(F3, (1, d))
                v  = (pm5[idx]
                      + F3 * (pm1[idx] - pm2[idx])
                      + F3 * (pm3[idx] - pm4[idx]))
                ui[idx] = (popold[idx] * mpo[idx] + v * mui[idx])

            # Strategy 4: DE/current-to-rand/1
            # ui = x + rand*(pm5-x) + F*(pm1-pm2)
            # Note: first coeff is scalar rand (MATLAB: rand), not F
            if len(index[3]) > 0:
                idx = index[3]
                F4  = np.random.normal(0.5, 0.3, (len(idx), 1))
                F4  = np.tile(F4, (1, d))
                r_s = np.random.rand()   # scalar rand (MATLAB: rand)
                v   = (popold[idx]
                       + r_s * (pm5[idx]  - popold[idx])
                       + F4  * (pm1[idx]  - pm2[idx]))
                ui[idx] = v   # no crossover for strategy 4
                # Note: r_s is a scalar rand shared across all S4 particles
                # this generation. MATLAB code (active line) uses scalar rand.
                # A commented-out line in SaDE.m shows rand(m,1) was
                # considered but the scalar version was the final choice.
                # The paper says K in [0,1] per individual, but we follow
                # the released code.

            # ── Boundary: resample violating dimensions ───────────
            out = (ui < lb) | (ui > ub)
            rand_fill = lb + np.random.rand(NP, d) * (ub - lb)
            ui = np.where(out, rand_fill, ui)

            # ── Selection and success tracking ─────────────────────
            lpcount = np.zeros(K, dtype=int)
            npcount = np.zeros(K, dtype=int)

            for i in range(NP):
                if problem.is_done():
                    break

                f_trial = problem.evaluate(ui[i])

                if f_trial <= fit[i]:
                    pop[i] = ui[i]
                    fit[i] = f_trial

                    # Record which strategy succeeded
                    for k in range(K):
                        if aaa[k][i] == 1:
                            lpcount[k] += 1
                            aaaa[k].append([cc[i, k], gen])

                    if f_trial <= fit[best_idx]:
                        best_idx = i
                else:
                    for k in range(K):
                        if aaa[k][i] == 1:
                            npcount[k] += 1

            # ── Update sliding window ns/nf ───────────────────────
            ns_win.append(lpcount.copy())
            nf_win.append(npcount.copy())

            # Drop oldest generation when window exceeds learngen
            if len(ns_win) > learngen:
                ns_win.pop(0)
                nf_win.pop(0)

            # ── Update strategy probabilities ─────────────────────
            if gen >= learngen:
                ns_total = np.sum(ns_win, axis=0)
                nf_total = np.sum(nf_win, axis=0)
                for k in range(K):
                    total = ns_total[k] + nf_total[k]
                    pfit[k] = (ns_total[k] / total + 0.01
                               if total > 0 else 0.01)

        return self._make_result(problem)

    # ── Stochastic Universal Sampling ─────────────────────────────

    def _sus(self, pfit: np.ndarray, NP: int) -> np.ndarray:
        """
        Stochastic Universal Sampling strategy assignment.
        Allocate mutation strategies by stochastic universal sampling.

        Returns stpool: shape (NP,), values in 0..K-1 (0-indexed).
        """
        K = len(pfit)
        normfit = pfit / pfit.sum()

        rr      = np.random.rand()
        spacing = 1.0 / NP
        # randnums: NP evenly spaced values starting at rr, wrapped mod 1
        randnums = np.sort(np.mod(
            rr + np.arange(NP) * spacing, 1.0
        ))

        stpool = np.zeros(NP, dtype=int)
        partsum = 0.0
        prev_count = 0
        for k in range(K):
            partsum += normfit[k]
            curr_count = int(np.sum(randnums < partsum))
            stpool[prev_count:curr_count] = k
            prev_count = curr_count

        # Shuffle assignments randomly
        np.random.shuffle(stpool)
        return stpool
