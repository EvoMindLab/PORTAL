# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
Classic DE — Differential Evolution (DE/rand/1/bin)
=====================================================
Reference:
    Storn, R., & Price, K. (1997).
    Differential evolution – a simple and efficient heuristic for global
    optimization over continuous spaces.
    Journal of Global Optimization, 11(4), 341–359.
    DOI: 10.1023/A:1008202821328

Official code:
    http://www1.icsi.berkeley.edu/~storn/code.html
    File: devec2.m (MATLAB), de.c (C)
    Server not reachable at time of implementation.
    Implemented directly from paper equations — DE is unambiguous.

Key mechanism:
    DE maintains a population of NP real-valued vectors. Each generation,
    for every target vector x_i a trial vector u_i is created via:

    (1) Mutation — DE/rand/1:
        Pick three distinct random indices r1, r2, r3 ≠ i
        v_i = x_r1 + F * (x_r2 - x_r3)
        The difference vector (x_r2 - x_r3) is scaled by F and added
        to the base vector x_r1.

    (2) Crossover — binomial (bin):
        For each dimension j:
            u_i[j] = v_i[j]   if rand() < CR or j == j_rand
                   = x_i[j]   otherwise
        j_rand ensures at least one dimension comes from the mutant.

    (3) Selection — greedy:
        x_i = u_i   if f(u_i) <= f(x_i)
            = x_i   otherwise

    This is the canonical DE/rand/1/bin variant — the most widely used
    baseline in the DE literature.

    Boundary handling: reflection (paper does not specify; reflection
    is the standard practical choice — resample violating components
    from [lb, ub] uniformly).

Parameters (recommended defaults from Storn & Price 1997):
    pop_size : population size NP  (default: 10*d, paper: 5*d to 10*d)
    F        : scaling factor      (default: 0.5, paper: [0.4, 1.0])
    CR       : crossover rate      (default: 0.9, paper: [0.0, 1.0])
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class Classic_DE(BaseAlgorithm):
    """
    Classical Differential Evolution — DE/rand/1/bin.
    Canonical baseline for all DE variants.
    """

    def __init__(
        self,
        pop_size : int   = None,   # None = auto: 10*d
        F        : float = 0.5,    # scaling factor
        CR       : float = 0.9,    # crossover rate
    ):
        super().__init__(name='Classic DE')
        self.pop_size = pop_size
        self.F        = F
        self.CR       = CR

    @staticmethod
    def _sample_indices_excluding(excluded: np.ndarray, upper: int) -> np.ndarray:
        """Sample one index in [0, upper) per row, excluding row-specific ids."""
        values = np.random.randint(0, upper - 1, size=len(excluded))
        return values + (values >= excluded)

    def run(self, problem: PortalProblem) -> dict:
        """
        Run Classic DE on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d   = problem.d
        lb  = problem.lb        # shape (d,)
        ub  = problem.ub        # shape (d,)

        # ── Algorithm parameters ──────────────────────────────────
        NP = self.pop_size if self.pop_size is not None else 10 * d
        if NP < 4:
            raise ValueError("Classic DE requires pop_size >= 4 for DE/rand/1/bin.")
        if problem.remaining_fe() < 4:
            while not problem.is_done():
                problem.evaluate(problem.random_solution())
            return self._make_result(problem)
        NP = min(NP, problem.remaining_fe())

        F  = self.F
        CR = self.CR

        # ── Initialisation ────────────────────────────────────────
        pop = lb + np.random.rand(NP, d) * (ub - lb)
        fit = problem.evaluate_batch(pop)

        # ── Main loop ─────────────────────────────────────────────
        # Use the canonical synchronous DE generation: build all trial vectors
        # from the current population, batch-evaluate them, then apply greedy
        # selection. This keeps the FE budget identical while avoiding one
        # fitness call per target vector.
        while not problem.is_done():
            batch_size = min(NP, problem.remaining_fe())
            if batch_size <= 0:
                break

            target_idx = np.arange(batch_size)
            r1 = self._sample_indices_excluding(target_idx, NP)

            r2 = self._sample_indices_excluding(target_idx, NP)
            duplicate = r2 == r1
            while np.any(duplicate):
                r2[duplicate] = self._sample_indices_excluding(target_idx[duplicate], NP)
                duplicate = r2 == r1

            r3 = self._sample_indices_excluding(target_idx, NP)
            duplicate = (r3 == r1) | (r3 == r2)
            while np.any(duplicate):
                r3[duplicate] = self._sample_indices_excluding(target_idx[duplicate], NP)
                duplicate = (r3 == r1) | (r3 == r2)

            # ── Mutation: DE/rand/1 ───────────────────────────────
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # ── Boundary handling: reflection ─────────────────────
            below = mutant < lb
            above = mutant > ub
            mutant = np.where(below, 2*lb - mutant, mutant)
            mutant = np.where(above, 2*ub - mutant, mutant)
            still_out = (mutant < lb) | (mutant > ub)
            mutant = np.where(
                still_out,
                lb + np.random.rand(batch_size, d) * (ub - lb),
                mutant
            )

            # ── Crossover: binomial ───────────────────────────────
            mask = np.random.rand(batch_size, d) < CR
            j_rand = np.random.randint(d, size=batch_size)
            mask[np.arange(batch_size), j_rand] = True

            trial = np.where(mask, mutant, pop[target_idx])

            # ── Selection: greedy ─────────────────────────────────
            f_trial = problem.evaluate_batch(trial)
            evaluated_idx = target_idx[:len(f_trial)]
            improved = f_trial <= fit[evaluated_idx]
            if np.any(improved):
                pop[evaluated_idx[improved]] = trial[:len(f_trial)][improved]
                fit[evaluated_idx[improved]] = f_trial[improved]


        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)
