# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
CMSA-ES — Covariance Matrix Self-Adaptation Evolution Strategy
==============================================================
Reference:
    Beyer, H.-G., & Sendhoff, B. (2008).
    Covariance matrix adaptation revisited – the CMSA evolution strategy –.
    Parallel Problem Solving from Nature (PPSN X), LNCS 5199, pp. 123–132.
    DOI: 10.1007/978-3-540-87700-4_13

Official code:
    No official code published by the authors.
    Implemented directly from Algorithm (Fig. 2) and equations (1)-(2)
    in Beyer & Sendhoff (2008).

Key mechanism:
    CMSA-ES is a simpler alternative to CMA-ES that replaces the complex
    cumulative path-based CMA update with a direct sample-based covariance
    estimate, and replaces CSA step-size control with classical σ-self-
    adaptation (log-normal mutation of individual σ per offspring).

    DIFFERENCES FROM CMA-ES:
    ┌──────────────────────────────────────────────────────────────────┐
    │ CMA-ES                    │ CMSA-ES                             │
    │ CSA (cumulative paths ps) │ σ-SA (per-offspring log-normal)     │
    │ Rank-1 + rank-μ C update  │ Direct sample average hss^T>_μ      │
    │ 5 endogenous parameters   │ 2 endogenous parameters: τ, τ_c     │
    │ Weighted recombination    │ Uniform recombination (mean)        │
    └──────────────────────────────────────────────────────────────────┘

    GENERATION LOOP (Fig. 2):
        For l = 1..λ:
            σ_l = <σ> · exp(τ · N(0,1))               (R1) log-normal σ-SA
            s_l = √C · N(0,I)                           (R2) correlated direction
            z_l = σ_l · s_l                              (R3) scaled mutation
            y_l = y + z_l                                (R4) offspring
            f_l = f(y_l)                                 (R5) evaluate

        Select best μ offspring (truncation selection, uniform weights).
        y  ← y + <z>_μ                                  (R6) mean update
        C  ← (1 - 1/τ_c)·C + (1/τ_c)·<ss^T>_μ         (R7) C update

    PARAMETERS (Eqs. 1 and 2):
        τ   = 1/√(2N)           (learning rate for σ-SA, Eq. 1)
        τ_c = 1 + N(N+1)/(2μ)  (C learning time constant, Eq. 2)
        μ/λ = 1/4               (truncation ratio, paper Section 4.2)
        λ   = max(4, 4 + floor(3·log(N)))  (default, same as CMA-ES)

    COVARIANCE UPDATE via Cholesky:
        The paper suggests using Cholesky decomposition of C to avoid the
        O(N³) eigendecomposition. We use numpy's Cholesky: s_l = A · N(0,I)
        where A = chol(C) (lower triangular). C is updated via (R7) and
        re-decomposed lazily every generation.

    BOUNDARY: clip to [lb, ub] after mutation (pragmatic for bound constraints).

    RESTART: on numerical issues (C not positive definite), restart from
        current best with fresh C = I.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class CMSA_ES(BaseAlgorithm):
    """
    CMSA-ES: Covariance Matrix Self-Adaptation ES.
    Implemented from Fig. 2 and Eqs. (1)-(2) of Beyer & Sendhoff (2008).
    No official code exists; this is a direct paper translation.
    """

    def __init__(
        self,
        lam      = None,    # λ offspring; None → 4 + int(3*log(N))
        mu_ratio = 0.25,    # μ/λ truncation ratio (paper: 1/4)
        sigma0   = None,    # initial σ; None → (ub-lb)/4
        restarts = 9,
    ):
        super().__init__(name='CMSA-ES')
        self.lam      = lam
        self.mu_ratio = mu_ratio
        self.sigma0   = sigma0
        self.restarts = restarts

    def run(self, problem):
        d  = problem.d
        lb = problem.lb
        ub = problem.ub

        # ── Population sizes ──────────────────────────────────────
        lam = self.lam if self.lam else max(4, 4 + int(3 * np.log(d)))
        mu  = max(1, int(self.mu_ratio * lam))

        # ── Strategy parameters (Eqs. 1 and 2) ───────────────────
        tau   = 1.0 / np.sqrt(2.0 * d)               # Eq. 1
        tau_c = 1.0 + d * (d + 1) / (2.0 * mu)       # Eq. 2

        alpha_c = 1.0 / tau_c   # shorthand: 1/τ_c

        n_restarts = 0

        while not problem.is_done():

            # ── Initialise (or restart) ───────────────────────────
            if n_restarts == 0:
                y = (lb + ub) / 2.0
            else:
                y = problem.best_x.copy()

            sigma0 = (self.sigma0 if self.sigma0 is not None
                      else float((ub - lb).mean() / 4.0))
            sigma_pop = np.full(lam, sigma0)   # per-offspring σ (pool init)
            C = np.eye(d)                       # covariance matrix
            A = np.eye(d)                       # Cholesky factor: C = A A^T

            # ── Inner loop ────────────────────────────────────────
            while not problem.is_done():

                # Sample λ offspring
                sigma_l = np.zeros(lam)
                s_l     = np.zeros((lam, d))
                y_l     = np.zeros((lam, d))
                f_l     = np.full(lam, np.inf)

                sigma_mean = np.mean(sigma_pop)   # <σ> from previous gen

                for l in range(lam):
                    if problem.is_done():
                        break
                    # (R1) log-normal σ-SA
                    sigma_l[l] = sigma_mean * np.exp(tau * np.random.randn())
                    # (R2) correlated direction: s ~ N(0, C)
                    z_std = np.random.randn(d)
                    s_l[l] = A @ z_std       # s = A * N(0,I), C = A A^T
                    # (R3-R4) offspring
                    y_l[l] = y + sigma_l[l] * s_l[l]
                    y_l[l] = np.clip(y_l[l], lb, ub)
                    # (R5) evaluate
                    f_l[l] = problem.evaluate(y_l[l])

                # Truncation selection: best μ
                order  = np.argsort(f_l)
                sel    = order[:mu]

                # (R6) mean update: y ← y + <z>_μ
                z_sel     = (sigma_l[sel, np.newaxis] * s_l[sel])
                y         = y + z_sel.mean(axis=0)
                y         = np.clip(y, lb, ub)

                # Carry selected σ values to next generation
                sigma_pop = np.tile(sigma_l[sel], int(np.ceil(lam / mu)))[:lam]

                # (R7) covariance update: C ← (1 - 1/τ_c)C + (1/τ_c)<ss^T>_μ
                s_sel = s_l[sel]                   # (mu, d)
                S_avg = (s_sel.T @ s_sel) / mu     # mean outer product
                C     = (1 - alpha_c) * C + alpha_c * S_avg

                # Recompute Cholesky factor
                try:
                    C = 0.5 * (C + C.T)            # enforce symmetry
                    A = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # C lost positive definiteness — trigger restart
                    break

            n_restarts += 1
            if n_restarts > self.restarts or problem.is_done():
                break

        return self._make_result(problem)
