# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
(μ/μ, λ)-ES — Canonical Evolution Strategy
============================================
Reference:
    Rechenberg, I. (1973).
    Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien
    der biologischen Evolution. Frommann-Holzboog, Stuttgart.

    Schwefel, H.-P. (1977).
    Numerische Optimierung von Computer-Modellen mittels der
    Evolutionsstrategie. Birkhäuser, Basel.

    Beyer, H.-G., & Schwefel, H.-P. (2002).
    Evolution strategies: A comprehensive introduction.
    Natural Computing, 1(1), 3–52.
    DOI: 10.1023/A:1015059928466

    Implemented from first principles following the standard textbook
    description (Beyer & Schwefel 2002, Section 2).

Key mechanism:
    (μ/μ, λ)-ES is the foundational real-valued evolutionary strategy.
    "μ/μ" means all μ parents contribute equally to one recombined
    parent (intermediate/whole arithmetic recombination).
    "λ" offspring are generated, the best μ are selected (comma
    selection — parents are discarded each generation).

    RECOMBINATION (intermediate):
        x_rec = (1/μ) Σ x_i        (component-wise mean of parents)
        σ_rec = (1/μ) Σ σ_i        (same for step sizes)

    SELF-ADAPTIVE STEP-SIZE MUTATION (individual σ per dimension):
        σ'_j = σ_j · exp(τ'·N(0,1) + τ·N_j(0,1))
        x'_j = x_rec_j + σ'_j · N_j(0,1)

        where:
            τ  = 1 / sqrt(2·D)          (global learning rate)
            τ' = 1 / sqrt(2·sqrt(D))    (individual learning rate)
            σ_min = 1e-10               (step-size lower bound)

    SELECTION (comma):
        Best μ of λ offspring become the new parents.
        No elitism — parents are never carried over.

    BOUNDARY: clip to [lb, ub] after mutation.

Parameters (standard textbook defaults):
    mu_     : number of parents μ    (default: 15)
    lambda_ : number of offspring λ  (default: 100)
    sigma0  : initial step size      (default: (ub-lb)/6 per dim)
    sigma_min : minimum step size    (default: 1e-10)

    Typical ratio λ/μ ≈ 7 (Beyer & Schwefel recommend λ ≥ 7μ for
    comma selection). Defaults mu=15, lambda=100 give λ/μ ≈ 6.7.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class ES(BaseAlgorithm):
    """
    (mu/mu, lambda)-ES: Canonical Evolution Strategy with self-adaptive
    per-dimension step sizes and intermediate recombination.
    Implemented from Beyer & Schwefel (2002).
    """

    def __init__(
        self,
        mu_       = 15,
        lambda_   = 100,
        sigma0    = None,   # None -> (ub-lb)/6 per dim
        sigma_min = 1e-10,
    ):
        super().__init__(name='ES')
        self.mu_       = mu_
        self.lambda_   = lambda_
        self.sigma0    = sigma0
        self.sigma_min = sigma_min

    def run(self, problem):
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub

        mu      = self.mu_
        lam     = self.lambda_
        sig_min = self.sigma_min

        # ── Learning rates (Beyer & Schwefel 2002, Eq. 19) ───────
        tau_prime = 1.0 / np.sqrt(2.0 * d)            # global
        tau       = 1.0 / np.sqrt(2.0 * np.sqrt(d))   # individual

        # ── Initialisation ────────────────────────────────────────
        sig_init = (np.full(d, float(self.sigma0))
                    if self.sigma0 is not None
                    else (ub - lb) / 6.0)

        parents_x = lb + np.random.rand(mu, d) * (ub - lb)
        parents_s = np.tile(sig_init, (mu, 1))
        parents_f = np.array([problem.evaluate(parents_x[i])
                               for i in range(mu)])

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            # ── Intermediate recombination ─────────────────────────
            x_rec = parents_x.mean(axis=0)   # (d,)
            s_rec = parents_s.mean(axis=0)   # (d,)

            # ── Generate lambda offspring ─────────────────────────
            off_x = np.empty((lam, d))
            off_s = np.empty((lam, d))
            off_f = np.empty(lam)

            global_draws = np.random.randn(lam)
            indiv_draws  = np.random.randn(lam, d)
            x_draws      = np.random.randn(lam, d)

            for k in range(lam):
                if problem.is_done():
                    break
                # Step-size mutation
                s_new = (s_rec
                         * np.exp(tau_prime * global_draws[k]
                                  + tau * indiv_draws[k]))
                s_new = np.maximum(s_new, sig_min)

                # Object-variable mutation
                x_new = x_rec + s_new * x_draws[k]
                x_new = np.clip(x_new, lb, ub)

                off_s[k] = s_new
                off_x[k] = x_new
                off_f[k] = problem.evaluate(x_new)

            # ── Comma selection: best mu of lambda ────────────────
            best_idx  = np.argsort(off_f)[:mu]
            parents_x = off_x[best_idx]
            parents_s = off_s[best_idx]
            parents_f = off_f[best_idx]

        return self._make_result(problem)
