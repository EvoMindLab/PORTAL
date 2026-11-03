# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
jDE — Self-Adapting Control Parameters in Differential Evolution
================================================================
Reference:
    Brest, J., Greiner, S., Bošković, B., Mernik, M., & Žumer, V. (2006).
    Self-adapting control parameters in differential evolution:
    a comparative study on numerical benchmark problems.
    IEEE Transactions on Evolutionary Computation, 10(6), 646–657.
    DOI: 10.1109/TEVC.2006.890628

Source:
    Implemented directly from paper equations (Section II-B).
    No official author code is publicly available.

Key mechanism:
    Classic DE uses fixed F and CR throughout the run. jDE encodes
    F and CR directly into each individual vector, allowing them to
    self-adapt alongside the solution.

    Each individual i carries its own (F_i, CR_i) pair. At each
    generation, new parameter values are drawn with small probability
    (tau1, tau2); otherwise the parent's values are inherited:

        F_i_new  = F_lower + rand * F_upper   if rand < tau1
                 = F_i                         otherwise

        CR_i_new = rand                        if rand < tau2
                 = CR_i                        otherwise

    (paper Eq. 4-5, with tau1 = tau2 = 0.1)

    The trial vector is then generated using the new (F_i_new, CR_i_new)
    with the standard DE/rand/1/bin strategy. Selection is greedy:
    if the trial is better, both the solution AND its parameters
    (F_i_new, CR_i_new) replace the parent.

    This way, successful parameter values propagate to the next
    generation — the algorithm implicitly learns good F and CR values
    for the problem at hand without user tuning.

Parameters (defaults follow Brest et al. 2006, Table I):
    pop_size  : population size NP   (default: 10*d)
    F_lower   : lower bound for F    (default: 0.1)
    F_upper   : upper bound for F    (default: 0.9)
    tau1      : F adaptation prob    (default: 0.1)
    tau2      : CR adaptation prob   (default: 0.1)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class JDE(BaseAlgorithm):
    """
    Self-Adapting Differential Evolution (jDE).
    F and CR encoded per individual and adapted via random perturbation.
    """

    def __init__(
        self,
        pop_size : int   = None,   # None = auto: 10*d
        F_lower  : float = 0.1,    # lower bound for F (paper: 0.1)
        F_upper  : float = 0.9,    # upper bound for F (paper: 0.9)
        tau1     : float = 0.1,    # F adaptation probability
        tau2     : float = 0.1,    # CR adaptation probability
    ):
        super().__init__(name='jDE')
        self.pop_size = pop_size
        self.F_lower  = F_lower
        self.F_upper  = F_upper
        self.tau1     = tau1
        self.tau2     = tau2

    def run(self, problem: PortalProblem) -> dict:
        """
        Run jDE on a PORTAL problem.

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
        NP      = self.pop_size if self.pop_size is not None else 10 * d
        F_lower = self.F_lower
        F_upper = self.F_upper
        tau1    = self.tau1
        tau2    = self.tau2

        # ── Initialisation ────────────────────────────────────────
        pop = lb + np.random.rand(NP, d) * (ub - lb)
        fit = np.array([problem.evaluate(pop[i]) for i in range(NP)])

        # Initialise F and CR per individual (paper Section II-B)
        F  = np.full(NP, 0.5)   # initial F values
        CR = np.full(NP, 0.9)   # initial CR values

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            for i in range(NP):
                if problem.is_done():
                    break

                # ── Parameter self-adaptation (paper Eq. 4-5) ────
                # F: draw new value if rand < tau1
                F_new = (F_lower + np.random.rand() * F_upper
                         if np.random.rand() < tau1
                         else F[i])

                # CR: draw new value if rand < tau2
                CR_new = (np.random.rand()
                          if np.random.rand() < tau2
                          else CR[i])

                # ── Mutation: DE/rand/1 ───────────────────────────
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                mutant = pop[r1] + F_new * (pop[r2] - pop[r3])

                # Boundary: clamp violating components to bound value
                # (paper Sec. III-A: "set to bound value")
                mutant = np.clip(mutant, lb, ub)

                # ── Crossover: binomial ───────────────────────────
                j_rand = np.random.randint(d)
                mask   = np.random.rand(d) < CR_new
                mask[j_rand] = True

                trial = np.where(mask, mutant, pop[i])

                # ── Selection: greedy ─────────────────────────────
                # If trial wins, propagate BOTH solution AND parameters
                # (paper Sec. III-C: strictly better, i.e. f(u) < f(x))
                f_trial = problem.evaluate(trial)

                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    F[i]   = F_new    # successful F propagates
                    CR[i]  = CR_new   # successful CR propagates

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)
