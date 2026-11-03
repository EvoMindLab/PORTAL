# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
CMA-ES — Covariance Matrix Adaptation Evolution Strategy
=========================================================
Reference:
    Hansen, N., & Ostermeier, A. (2001).
    Completely derandomized self-adaptation in evolution strategies.
    Evolutionary Computation, 9(2), 159–195.
    DOI: 10.1162/106365601750190398

    Hansen, N. (2016).
    The CMA Evolution Strategy: A Tutorial.
    arXiv:1604.00772.

Official code:
    purecma.py — Nikolaus Hansen, 2010-2011, 2017.
    Released into the public domain.
    URL: http://github.com/CMA-ES/pycma
    Installed via: pip install cma

    This wrapper uses the CMAES ask/tell interface from purecma.py
    verbatim, adapting only the termination condition and boundary
    handling to integrate with the PORTAL framework.

Key mechanism:
    CMA-ES maintains a multivariate Gaussian N(m, σ²C) and adapts
    its mean m, step-size σ, and full covariance matrix C using
    information from the population.

    ASK:  sample λ = 4 + floor(3·ln(D)) candidates from N(m, σ²C)
          x_k = m + σ · B·D·z_k,  z_k ~ N(0, I)

    TELL (three coupled updates):
    1. MEAN:     m ← weighted sum of top-μ candidates
    2. STEP-SIZE (CSA — cumulative step-size adaptation):
         ps ← (1-cs)·ps + √(cs(2-cs)μeff) · C^{-1/2} · (m'-m)/σ
         σ  ← σ · exp(cs/damps · (||ps||/χ_N - 1))
    3. COVARIANCE (rank-1 + rank-μ update):
         pc ← (1-cc)·pc + hsig·√(cc(2-cc)μeff) · (m'-m)/σ
         C  ← (1-c1-cμ)·C + c1·pc·pcᵀ + cμ·Σ wk·dk·dkᵀ
         where dk = (x_{k:λ} - m_old)/σ

    BOUNDARY: clip to [lb, ub]. Solutions sampled outside bounds are
    clipped before evaluation. CMA-ES is designed for unconstrained
    problems; clipping is the standard pragmatic approach for bound
    constraints when the optimum is not on the boundary.

    TERMINATION (purecma.py criteria, checked each generation):
    - condition number of C > 1e14
    - range of fitness values < 1e-12 (tolfun)
    - σ·max(D)^0.5 < 1e-11 (tolx)
    These are used as restart triggers, not hard stops, since PORTAL
    controls the FE budget. On convergence, the algorithm restarts
    with a fresh σ and C from the current best solution until max_fe
    is exhausted.

Parameters (purecma defaults):
    sigma0   : initial step size (default: (ub-lb)/4)
    popsize  : λ, default '4 + int(3 * log(N))' (Hansen's formula)
    restarts : kept for backward compatibility; CMA-ES now restarts
               until max_fe is exhausted.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm

# Import Hansen's official purecma — ask/tell interface
from cma.more_algorithms.purecma import CMAES as _PureCMAES


class CMA_ES(BaseAlgorithm):
    """
    CMA-ES wrapper around Hansen's official purecma.py.
    Uses the ask/tell interface directly; adds PORTAL boundary
    handling (clip) and budget-aware restart on convergence.
    """

    def __init__(
        self,
        sigma0   = None,    # None -> (ub-lb)/4
        popsize  = None,    # None -> Hansen's '4 + int(3*log(N))'
        restarts = None,    # kept for compatibility; budget controls stopping
    ):
        super().__init__(name='CMA-ES')
        self.sigma0   = sigma0
        self.popsize  = popsize
        self.restarts = restarts

    def run(self, problem: PortalProblem) -> dict:
        d  = problem.d
        lb = problem.lb
        ub = problem.ub

        sigma0  = self.sigma0 if self.sigma0 is not None else float((ub - lb).mean() / 4.0)
        popsize = self.popsize  # None → purecma uses '4 + int(3*log(N))'

        while not problem.is_done():
            fe_before_restart = problem.fe_count

            # ── Initialise (or re-initialise after restart) ───────
            # Start from best known solution if restarting
            if problem.best_x is None:
                x0 = (lb + ub) / 2.0    # domain centre
            else:
                x0 = problem.best_x.copy()  # restart from best found
                sigma0 = float((ub - lb).mean() / 4.0)  # reset step size

            es = _PureCMAES(
                x0.tolist(),
                sigma0,
                popsize=str(popsize) if popsize else _PureCMAES.__init__.__defaults__[0],
                maxfevals=int(1e12),   # we use problem.is_done() for budget
                randn=lambda mu, sd: float(np.random.normal(mu, sd)),
            )

            # ── Inner loop: one CMA-ES run ─────────────────────────
            while not es.stop() and not problem.is_done():
                X_list = es.ask()   # list of list, λ candidates

                # Apply boundary: clip each candidate
                X_clipped = [
                    np.clip(np.array(x), lb, ub).tolist()
                    for x in X_list
                ]

                # Evaluate
                fit = []
                for x in X_clipped:
                    if problem.is_done():
                        fit.append(float('inf'))
                    else:
                        fit.append(problem.evaluate(np.array(x)))

                es.tell(X_clipped, fit)

            if problem.is_done():
                break

            # If a stop criterion fires before any evaluation, consume one
            # safe evaluation so the outer budget-controlled loop progresses.
            if problem.fe_count == fe_before_restart:
                problem.evaluate(np.clip(x0, lb, ub))

        return self._make_result(problem)
