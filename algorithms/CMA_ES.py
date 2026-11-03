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

    TERMINATION (purecma criteria, checked each generation):
    - condition number of C > 1e14
    - range of fitness values < 1e-12 (tolfun)
    - σ·max(D)^0.5 < 1e-11 (tolx)
    These are used as restart triggers, not hard stops, since PORTAL
    controls the FE budget. On convergence, the algorithm restarts
    with a fresh σ and C from the current best solution.

Parameters (purecma defaults):
    sigma0   : initial step size (default: (ub-lb)/4)
    popsize  : λ, default '4 + int(3 * log(N))' (Hansen's formula)
    restarts : max restarts on premature convergence (default: 9)

API notes (cma 4.4.x):
    - Import path: cma.purecma (not cma.more_algorithms.purecma)
    - CMAES(xstart, sigma, popsize='4 + int(3 * log(N))', ftarget=None,
            maxfevals=...)
    - popsize is passed as a string expression or integer
    - es.stop() returns {} (falsy) when running, non-empty dict when stopped
    - es.result is a plain tuple:
        (xbest, fbest, evals_xbest, evaluations, iterations, xmean, stds)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm

# Use cma's pure-Python CMA-ES implementation.
import cma as _cma_pkg
_PureCMAES = _cma_pkg.purecma.CMAES


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
        restarts = 9,       # max restarts on convergence
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

        # popsize: pass as string expression (purecma default) or integer
        # None → use purecma's default: '4 + int(3 * log(N))'
        popsize = self.popsize if self.popsize is not None else '4 + int(3 * log(N))'

        n_restarts = 0

        while not problem.is_done():
            # ── Initialise (or re-initialise after restart) ────────
            if n_restarts == 0:
                x0 = (lb + ub) / 2.0        # domain centre on first run
            else:
                x0 = problem.best_x.copy()  # restart from best found
                sigma0 = float((ub - lb).mean() / 4.0)  # reset step size

            es = _PureCMAES(
                x0.tolist(),
                sigma0,
                popsize=popsize,
                maxfevals=int(1e12),   # budget controlled by problem.is_done()
            )

            # ── Inner loop: one CMA-ES run ──────────────────────────
            while not es.stop() and not problem.is_done():
                X_list = es.ask()   # list of lists, λ candidates

                # Clip each candidate to [lb, ub]
                X_clipped = [
                    np.clip(np.array(x), lb, ub).tolist()
                    for x in X_list
                ]

                # Evaluate — honour budget
                fit = problem.evaluate_batch(np.asarray(X_clipped, dtype=float)).tolist()
                if len(fit) < len(X_clipped):
                    fit.extend([float('inf')] * (len(X_clipped) - len(fit)))

                es.tell(X_clipped, fit)

            # ── Restart logic ──────────────────────────────────────
            n_restarts += 1
            if n_restarts > self.restarts or problem.is_done():
                break

        return self._make_result(problem)
