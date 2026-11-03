# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
NES — Natural Evolution Strategies (xNES variant)
===================================================
Reference:
    Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., &
    Schmidhuber, J. (2014).
    Natural Evolution Strategies.
    Journal of Machine Learning Research, 15(1), 949–980.
    URL: https://jmlr.org/papers/v15/wierstra14a.html

    Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J.
    (2010). Exponential Natural Evolution Strategies.
    Proceedings of GECCO 2010, pp. 393–400.
    DOI: 10.1145/1830483.1830557

Official code:
    PyBrain nes.py (Wierstra, Schaul, Sun Yi) — official authors' code.
    URL: github.com/pybrain/pybrain/blob/master/pybrain/optimization/
         distributionbased/nes.py
    The PyBrain implementation inherits from VanillaGradientEvolutionStrategies
    and is not standalone. This implementation follows Algorithm 3 (canonical
    NES) and Section 3.4 (xNES) from Wierstra et al. (2014) directly.

Key mechanism:
    NES maintains a Gaussian search distribution N(μ, Σ) and updates its
    parameters in the direction of higher expected fitness using the
    *natural* gradient (F⁻¹ ∇J), where F is the Fisher information matrix.
    Using the natural gradient rather than the plain gradient ensures the
    update is invariant to the parameterisation of the distribution.

    This implements xNES (exponential NES), which parameterises the
    distribution as N(μ, σ²·(A^T A)) where A ∈ GL(d) is maintained via
    its matrix exponential. This avoids the need to invert Σ directly and
    keeps the matrix updates on the manifold of positive definite matrices.

    ALGORITHM (xNES, Algorithm 3 + Section 3.4 of Wierstra et al. 2014):

    Parameterisation:
        θ = (μ, σ, B)
        Σ = σ² · B B^T    (B is the "normalised" square root factor)

    Each generation:
        1. Sample:  z_k ~ N(0, I),   x_k = μ + σ·B·z_k  (k=1..λ)
        2. Evaluate: f_k = f(x_k)  (negate for maximisation, we minimise)
        3. Fitness shaping (rank-based utilities, Eq. 21):
               u_k = max(0, log(λ/2 + 1) - log(rank(k))) / Z  - 1/λ
               where rank(k) is the rank of f_k ascending (best=1)
               Z normalises so Σ u_k = 1 over positive entries
        4. Natural gradient estimates (Section 3.4):
               ∇_μ J = (1/σ) Σ u_k · B^{-1} z_{k:λ}
               ∇_σ J = Σ u_k · (||B^{-1} z_{k:λ}||² - d) / (d·σ)
               ∇_B J = Σ u_k · (B^{-1} z_{k:λ} z_{k:λ}^T - I)
                     (using A = B·exp(·/2) formulation from paper)
        5. Exponential updates (xNES):
               μ ← μ + η_μ · σ · B · ∇_μ J
               σ ← σ · exp(η_σ/2 · ∇_σ J)
               B ← B · exp(η_B/2 · ∇_B J)

    DEFAULT LEARNING RATES (Table 3 / Section 4 of Wierstra et al. 2014):
        η_μ = 1
        η_σ = (3 + log(d)) / (5 · d · √d)
        η_B = η_σ

    FITNESS SHAPING: rank-based utilities (Section 3.1) make the algorithm
    invariant to monotone transformations of the fitness function.

    BOUNDARY: clip to [lb, ub] after sampling. NES is designed for
    unconstrained problems; clipping is the standard pragmatic approach.

    RESTART: on numerical issues (B singular), restart from best-found
    solution with fresh B = I.

Parameters:
    lam      : population size λ (default: 4 + int(3*log(d)))
    eta_mu   : learning rate for mean (default: 1.0)
    eta_sigma: learning rate for σ (default: (9+3log(d))/(5d√d)  [Table 1])
    eta_B    : learning rate for B (default: same as eta_sigma)
    sigma0   : initial step size (default: (ub-lb)/4)
    restarts : max restarts on convergence (default: 9)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class NES(BaseAlgorithm):
    """
    xNES: Exponential Natural Evolution Strategies.
    Implemented from Wierstra et al. (2014) JMLR, Section 3.4 / Algorithm 3.
    No standalone official code; PyBrain implementation requires its framework.
    """

    def __init__(
        self,
        lam       = None,    # None -> 4 + int(3*log(d))
        eta_mu    = 1.0,
        eta_sigma = None,    # None -> (9+3*log(d))/(5*d*sqrt(d))  [Table 1]
        eta_B     = None,    # None -> same as eta_sigma  [Table 1]
        sigma0    = None,    # None -> (ub-lb)/4
        restarts  = 9,
    ):
        super().__init__(name='NES')
        self.lam       = lam
        self.eta_mu    = eta_mu
        self.eta_sigma = eta_sigma
        self.eta_B     = eta_B
        self.sigma0    = sigma0
        self.restarts  = restarts

    def run(self, problem):
        d  = problem.d
        lb = problem.lb
        ub = problem.ub

        # ── Population size ───────────────────────────────────────
        lam = self.lam if self.lam else 4 + int(3 * np.log(d))

        # ── Learning rates (Table 3 / Section 4) ─────────────────
        eta_mu    = self.eta_mu
        eta_sigma = (self.eta_sigma if self.eta_sigma is not None
                     else (9 + 3 * np.log(d)) / (5.0 * d * np.sqrt(d)))  # Table 1: (9+3logN)/(5N√N)
        eta_B     = self.eta_B if self.eta_B is not None else eta_sigma

        # ── Fitness shaping utilities (rank-based, Section 3.1) ───
        # Pre-compute utility weights for λ individuals
        # u_k = max(0, log(λ/2+1) - log(rank)) / Z  - 1/λ
        ranks = np.arange(1, lam + 1, dtype=float)
        raw   = np.maximum(0.0, np.log(lam / 2.0 + 1) - np.log(ranks))
        Z     = raw.sum()
        utils = raw / Z - 1.0 / lam   # shaped utilities, sum ≈ 0

        n_restarts = 0

        while not problem.is_done():

            # ── Initialise (or restart) ───────────────────────────
            if n_restarts == 0:
                mu = (lb + ub) / 2.0
            else:
                mu = problem.best_x.copy()

            sigma0 = (self.sigma0 if self.sigma0 is not None
                      else float((ub - lb).mean() / 4.0))
            sigma = sigma0
            B     = np.eye(d)   # normalised sqrt factor; Σ = σ²·B·B^T

            # ── Inner loop ────────────────────────────────────────
            while not problem.is_done():

                # 1. Sample λ individuals
                # Algorithm 5: s_k ~ N(0,I), x_k = μ + σ·B^T·s_k → distribution N(μ, σ²·B·B^T)
                # Row form: x_k = mu + sigma * (s_k @ B) since (B^T s_k)^T = s_k^T B
                S_noise = np.random.randn(lam, d)       # s_k ~ N(0,I)
                X       = mu + sigma * (S_noise @ B)    # x_k = μ + σ·B^T·s_k (row convention)
                X       = np.clip(X, lb, ub)

                # 2. Evaluate (negate: NES maximises, we minimise)
                fits = np.array([
                    problem.evaluate(X[k]) if not problem.is_done()
                    else np.inf
                    for k in range(lam)
                ])

                # 3. Rank best-first (ascending fitness = best first)
                order    = np.argsort(fits)
                u_sorted = utils   # utils[0] = weight for rank-1

                # 4. Natural gradient estimates in the z-space
                # ∇_μ:  Σ u_k · z_{k:λ}   (in transformed space)
                # ∇_σ:  Σ u_k · (||z_{k:λ}||² - d) / d
                # ∇_B:  Σ u_k · (z z^T - I)
                grad_mu    = np.zeros(d)
                grad_sigma = 0.0
                grad_M     = np.zeros((d, d))   # ∇_M J (scale+shape combined)

                for i, k in enumerate(order):
                    u  = u_sorted[i]
                    s  = S_noise[k]   # the noise vector for this individual
                    # s_k is the N(0,I) noise; xNES gradients in s-space
                    grad_mu    += u * s
                    grad_sigma += u * (np.dot(s, s) - d)
                    grad_M     += u * (np.outer(s, s) - np.eye(d))

                grad_sigma /= d

                # Decouple scale from shape (Algorithm 5):
                # ∇_B J = ∇_M J − ∇_σ J · I
                grad_B = grad_M - grad_sigma * np.eye(d)

                # 5. Exponential parameter updates (Algorithm 5 / xNES)
                mu    = mu + eta_mu * sigma * (B.T @ grad_mu)  # B^T in row convention
                sigma = sigma * np.exp(eta_sigma / 2.0 * grad_sigma)

                # Matrix exponential update: B ← B · expm(η_B/2 · ∇_B)
                try:
                    from scipy.linalg import expm as scipy_expm
                    B = B @ scipy_expm(eta_B / 2.0 * grad_B)
                except ImportError:
                    B = B @ (np.eye(d) + eta_B / 2.0 * grad_B)

                # Safety: clip sigma to avoid collapse
                sigma = max(sigma, 1e-10)

                # Check for numerical issues
                if not np.isfinite(B).all() or not np.isfinite(mu).all():
                    break

            n_restarts += 1
            if n_restarts > self.restarts or problem.is_done():
                break

        return self._make_result(problem)
