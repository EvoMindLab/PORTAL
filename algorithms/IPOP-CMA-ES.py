# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
IPOP-CMA-ES — CMA-ES with Increasing Population Restarts
==========================================================
Reference:
    Auger, A., & Hansen, N. (2005).
    A restart CMA evolution strategy with increasing population size.
    Proceedings of the IEEE Congress on Evolutionary Computation
    (CEC 2005), pp. 1769–1776.
    DOI: 10.1109/CEC.2005.1554902

Official code:
    IPOP-CMA-ES is implemented in virtually all CMA-ES libraries as the
    standard restart strategy. The inner CMA-ES used here is from:

    CyberAgentAILab/cmaes (Python, MIT licence):
        URL: github.com/CyberAgentAILab/cmaes
        Version used: 0.13.0
        Class: cmaes.CMA (ask/tell interface)
        cmaes.CMA.should_stop() implements all IPOP convergence criteria.

    The IPOP restart loop is from the paper (Algorithm 1) and is standard
    across all CMA-ES libraries (pycma, cmaes, libcmaes, etc.).

Key mechanism:
    IPOP-CMA-ES is standard CMA-ES with one modification: whenever the
    inner CMA-ES run converges (stagnation, tolx, tolxup, condition number,
    etc.), a new run begins with a doubled population size.

    ALGORITHM (Auger & Hansen 2005, Algorithm 1):
        lam ← lam_0                           (default: 4 + floor(3·log(N)))
        repeat
            run CMA-ES with population lam until should_stop()
            lam ← 2 · lam                     (IPOP doubling)
        until budget exhausted

    RESTART INITIALISATION (paper Section 2):
        - Mean x₀: uniformly random in [lb, ub]
        - Step size σ₀: (ub - lb) / 5         (paper default: 0.2 × range)
        - Covariance: C = I (fresh start each run)
        The fresh random start is essential — IPOP is designed to escape
        local basins by restarting from a new random point with larger λ.

    CONVERGENCE CRITERIA (cmaes.CMA.should_stop()):
        - tolfun:       range of recent fitness values < 1e-12
        - tolx:         σ·diag(C) and σ·pc all < 1e-11
        - tolxup:       σ·max(D) > 1e4 (divergence)
        - noeffectcoord: mean unchanged after 0.2σ perturbation
        - noeffectaxis:  mean unchanged after 0.1σD perturbation
        - conditioncov:  cond(C) > 1e14

    BOUNDARY: out-of-bounds solutions passed to PORTAL evaluator, which
    clamps them. CMA-ES is designed for unconstrained problems.

Parameters:
    lam0     : initial population size (default: None → 4+floor(3·log(N)))
    sigma0   : initial step size (default: (ub-lb).mean()/5)
    max_restarts : safety cap on restarts (default: 100)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm
from cmaes import CMA


class IPOP_CMA_ES(BaseAlgorithm):
    """
    IPOP-CMA-ES: CMA-ES with increasing population restarts.
    Uses cmaes.CMA (CyberAgentAILab) for the inner CMA-ES runs.
    Restart loop follows Auger & Hansen (2005), Algorithm 1.
    """

    def __init__(
        self,
        lam0         = None,   # initial λ; None → cmaes default
        sigma0       = None,   # initial σ; None → (ub-lb).mean()/5
        max_restarts = 100,
    ):
        super().__init__(name='IPOP-CMA-ES')
        self.lam0         = lam0
        self.sigma0       = sigma0
        self.max_restarts = max_restarts

    def run(self, problem: PortalProblem) -> dict:
        d  = problem.d
        lb = problem.lb
        ub = problem.ub

        sigma0 = (self.sigma0 if self.sigma0 is not None
                  else float((ub - lb).mean() / 5.0))

        # Initial λ: cmaes uses its own default if None
        lam = self.lam0   # None → cmaes will use 4 + floor(3·log(N))

        n_restarts = 0

        while not problem.is_done():

            # ── Fresh random start for each run (paper Section 2) ──
            x0 = lb + np.random.rand(d) * (ub - lb)

            es = CMA(
                mean           = x0,
                sigma          = sigma0,
                population_size = lam,   # None → library default on first run
                bounds         = None,   # unconstrained; PORTAL handles bounds
            )

            # Track λ after first run (cmaes may resolve None to its default)
            if lam is None:
                lam = es.population_size

            # ── Inner CMA-ES loop ──────────────────────────────────
            while not problem.is_done():
                solutions = []
                for _ in range(es.population_size):
                    if problem.is_done():
                        break
                    x = es.ask()
                    f = problem.evaluate(x)
                    solutions.append((x, f))

                if not solutions:
                    break

                # Pad with dummy if budget ran out mid-generation
                while len(solutions) < es.population_size:
                    solutions.append((solutions[0][0], solutions[0][1]))

                es.tell(solutions)

                if es.should_stop():
                    break

            # ── IPOP doubling ──────────────────────────────────────
            lam = lam * 2
            n_restarts += 1

            if n_restarts >= self.max_restarts:
                break

        return self._make_result(problem)
