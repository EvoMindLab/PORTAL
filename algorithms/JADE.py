# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
JADE — Adaptive Differential Evolution with Optional External Archive
======================================================================
Reference:
    Zhang, J., & Sanderson, A. C. (2009).
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945–958.
    DOI: 10.1109/TEVC.2009.2014613

Source:
    Implemented directly from paper Sections III-A, III-B, III-C,
    and Table I. No official author code is publicly available.

Key mechanism:
    JADE introduces three innovations over classic DE:

    1. DE/current-to-pbest/1 mutation:
        v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)
        x_pbest: random from top 100p% of current population
        x_r1:    random from current population (≠ i)
        x_r2:    random from population UNION archive (≠ i, ≠ r1)
        p = 0.05 (paper Table I)

    2. Optional external archive:
        Stores recently replaced inferior solutions (max size NP).
        Used as donor pool for r2 selection, providing information
        about past search directions. When full, a random entry is
        removed to make room.

    3. Adaptive parameter control (paper Eqs. 1-2):
        F_i  ~ Cauchy(mu_F, 0.1), clipped to (0, 1]
            if F_i <= 0: resample; if F_i > 1: clip to 1
        CR_i ~ N(mu_CR, 0.1), clipped to [0, 1]

        After each generation, using only SUCCESSFUL (F_i, CR_i):
            mu_F  = (1-c)*mu_F  + c * meanL(S_F)
            mu_CR = (1-c)*mu_CR + c * mean(S_CR)
        where meanL is the Lehmer mean: sum(F^2)/sum(F)
        and c = 0.1 is the adaptation learning rate.

Parameters (defaults follow Zhang & Sanderson 2009, Table I):
    pop_size : population size NP  (default: 100)
    p        : pbest fraction      (default: 0.05)
    c        : adaptation rate     (default: 0.1)
    archive  : use external archive(default: True)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class JADE(BaseAlgorithm):
    """
    JADE: Adaptive Differential Evolution with Optional External Archive.
    DE/current-to-pbest/1 mutation with Cauchy F and Gaussian CR adaptation.
    """

    def __init__(
        self,
        pop_size : int   = 100,   # population size (paper: 100)
        p        : float = 0.05,  # pbest fraction (paper: 0.05)
        c        : float = 0.1,   # adaptation learning rate (paper: 0.1)
        archive  : bool  = True,  # use external archive (paper: yes)
    ):
        super().__init__(name='JADE')
        self.pop_size = pop_size
        self.p        = p
        self.c        = c
        self.archive  = archive

    def run(self, problem: PortalProblem) -> dict:
        """
        Run JADE on a PORTAL problem.

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
        NP      = self.pop_size
        p       = self.p
        c       = self.c

        # Number of pbest candidates (at least 1)
        n_pbest = max(1, int(np.floor(p * NP)))

        # ── Initialisation ────────────────────────────────────────
        pop = lb + np.random.rand(NP, d) * (ub - lb)
        fit = np.array([problem.evaluate(pop[i]) for i in range(NP)])

        # Adaptive parameter means (paper: initialised to 0.5)
        mu_F  = 0.5
        mu_CR = 0.5

        # External archive (initially empty)
        archive = np.empty((0, d))

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            S_F  = []   # successful F values this generation
            S_CR = []   # successful CR values this generation

            # Sorted indices: ascending fitness (best first)
            sorted_idx = np.argsort(fit)

            for i in range(NP):
                if problem.is_done():
                    break

                # ── Sample F from Cauchy(mu_F, 0.1) ──────────────
                # Paper: if F <= 0 resample; if F > 1 set to 1
                F_i = 0.0
                while F_i <= 0:
                    F_i = mu_F + 0.1 * np.random.standard_cauchy()
                F_i = min(F_i, 1.0)

                # ── Sample CR from N(mu_CR, 0.1) ──────────────────
                CR_i = float(np.clip(
                    np.random.normal(mu_CR, 0.1), 0.0, 1.0
                ))

                # ── Select x_pbest ────────────────────────────────
                # Random from top n_pbest individuals
                pbest_pool = sorted_idx[:n_pbest]
                pbest_idx  = pbest_pool[np.random.randint(n_pbest)]
                x_pbest    = pop[pbest_idx]

                # ── Select r1 from population (≠ i) ──────────────
                r1_pool = [j for j in range(NP) if j != i]
                r1      = np.random.choice(r1_pool)

                # ── Select r2 from pop UNION archive (≠ i, ≠ r1) ─
                combined = list(range(NP))
                if self.archive and len(archive) > 0:
                    # Archive indices offset by NP
                    combined_arr = np.vstack([pop, archive])
                    r2_pool = [j for j in range(len(combined_arr))
                               if j != i and j != r1]
                else:
                    combined_arr = pop
                    r2_pool = [j for j in range(NP)
                               if j != i and j != r1]

                r2   = np.random.choice(r2_pool)
                x_r2 = combined_arr[r2]

                # ── Mutation: DE/current-to-pbest/1 ──────────────
                mutant = (pop[i]
                          + F_i * (x_pbest  - pop[i])
                          + F_i * (pop[r1]  - x_r2))

                # ── Boundary: midpoint-to-parent for violating dims ─
                # Paper Section II: x_j = (x_j^(parent) + bound) / 2
                below  = mutant < lb
                above  = mutant > ub
                mutant = np.where(below, (lb + pop[i]) / 2, mutant)
                mutant = np.where(above, (ub + pop[i]) / 2, mutant)

                # ── Crossover: binomial ───────────────────────────
                j_rand = np.random.randint(d)
                mask   = np.random.rand(d) < CR_i
                mask[j_rand] = True
                trial  = np.where(mask, mutant, pop[i])

                # ── Selection ─────────────────────────────────────
                f_trial = problem.evaluate(trial)

                if f_trial < fit[i]:
                    # Add loser to archive (trimming done after full loop)
                    if self.archive:
                        archive = np.vstack([archive, pop[i]])

                    pop[i] = trial
                    fit[i] = f_trial

                    # Record successful parameters
                    S_F.append(F_i)
                    S_CR.append(CR_i)

            # ── Trim archive to max size NP (paper Table I line 26)
            # Done AFTER full generation loop — archive may briefly
            # exceed NP mid-generation; paper trims once at end
            if self.archive and len(archive) > NP:
                while len(archive) > NP:
                    del_idx = np.random.randint(len(archive))
                    archive = np.delete(archive, del_idx, axis=0)

            # ── Update mu_F and mu_CR (paper Eqs. 1-2) ───────────
            if len(S_F) > 0:
                S_F_arr  = np.array(S_F)
                S_CR_arr = np.array(S_CR)

                # Lehmer mean for F (paper Eq.1)
                mean_L_F = np.sum(S_F_arr**2) / (np.sum(S_F_arr) + 1e-300)
                mu_F     = (1 - c) * mu_F + c * mean_L_F

                # Arithmetic mean for CR (paper Eq.2)
                mu_CR = (1 - c) * mu_CR + c * np.mean(S_CR_arr)

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)
