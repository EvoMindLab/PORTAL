# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""Matrix Adaptation Evolution Strategy (MA-ES).

Reference
---------
H.-G. Beyer and B. Sendhoff, "Simplify Your Covariance Matrix
Adaptation Evolution Strategy," IEEE Transactions on Evolutionary
Computation, vol. 21, no. 5, pp. 746-759, 2017.
doi: 10.1109/TEVC.2017.2680320

The distribution update follows Fig. 3 of the paper and was cross-checked
against the authors' MATLAB implementation and the independently maintained
Evolutionary-Intelligence implementations in D-LM-MA and pypop7.

The paper formulates MA-ES for unconstrained optimization. PORTAL recipes are
bound-constrained, so this wrapper applies an explicit, reported boundary
policy before evaluation. The default is coordinate-wise reflection, which
keeps offspring feasible without collapsing all overshoots onto a box face.
``clip`` is retained for comparison, and ``none`` is available only for
reproducing the unconstrained algorithm.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class MA_ES(BaseAlgorithm):
    """The small-population ``(mu/mu_w, lambda)`` MA-ES."""

    def __init__(
        self,
        lam: int | None = None,
        sigma0: float | None = None,
        restarts: int = 9,
        boundary_handling: str = "reflect",
    ):
        super().__init__(name="MA-ES")
        self.lam = lam
        self.sigma0 = sigma0
        # Numerical-recovery restarts only. They are not part of the basic
        # MA-ES update and normally remain unused.
        self.restarts = int(restarts)
        self.boundary_handling = str(boundary_handling).lower()

    def _repair(self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        if self.boundary_handling == "reflect":
            span = ub - lb
            offset = np.mod(x - lb, 2.0 * span)
            return np.where(offset <= span, lb + offset, ub - (offset - span))
        if self.boundary_handling == "clip":
            return np.clip(x, lb, ub)
        if self.boundary_handling == "none":
            return x
        raise ValueError(
            "boundary_handling must be 'reflect', 'clip', or 'none'; "
            f"received {self.boundary_handling!r}."
        )

    def run(self, problem: PortalProblem) -> dict:
        d = int(problem.d)
        lb = np.asarray(problem.lb, dtype=float)
        ub = np.asarray(problem.ub, dtype=float)
        if d <= 0 or lb.shape != (d,) or ub.shape != (d,):
            raise ValueError("MA-ES received inconsistent problem dimensions or bounds.")
        if np.any(ub <= lb):
            raise ValueError("Every upper bound must exceed its lower bound.")

        lam = int(self.lam) if self.lam is not None else 4 + int(3 * np.log(d))
        if lam < 2:
            raise ValueError("MA-ES requires lam >= 2.")
        mu = lam // 2

        # Positive log weights from the reference implementation. Using
        # (lambda + 1)/2 matters when a user supplies an odd lambda.
        ranks = np.arange(1, mu + 1, dtype=float)
        weights = np.log((lam + 1.0) / 2.0) - np.log(ranks)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        alpha_cov = 2.0
        c_s = (mu_eff + 2.0) / (mu_eff + d + 5.0)
        c_1 = alpha_cov / ((d + 1.3) ** 2 + mu_eff)
        c_w = min(
            1.0 - c_1,
            alpha_cov * (mu_eff + 1.0 / mu_eff - 2.0)
            / ((d + 2.0) ** 2 + alpha_cov * mu_eff / 2.0),
        )
        d_sigma = 1.0 + c_s + 2.0 * max(
            0.0, np.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0
        )
        path_decay = 1.0 - c_s
        path_gain = np.sqrt(mu_eff * c_s * (2.0 - c_s))
        e_chi = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

        sigma_initial = (
            float(self.sigma0)
            if self.sigma0 is not None
            else float(np.mean(ub - lb) / 4.0)
        )
        if not np.isfinite(sigma_initial) or sigma_initial <= 0.0:
            raise ValueError("sigma0 must be finite and positive.")

        numerical_restarts = 0
        mean = lb + np.random.rand(d) * (ub - lb)

        while not problem.is_done():
            sigma = sigma_initial
            transform = np.eye(d)
            path = np.zeros(d)

            while not problem.is_done():
                z = np.random.standard_normal((lam, d))
                directions = z @ transform.T
                candidates = self._repair(mean + sigma * directions, lb, ub)

                fitness = np.full(lam, np.inf)
                evaluated = 0
                for k in range(lam):
                    if problem.is_done():
                        break
                    fitness[k] = problem.evaluate(candidates[k])
                    evaluated += 1

                # A partial final generation cannot affect any subsequent
                # search decision and must not update the distribution using
                # unevaluated offspring.
                if evaluated < lam:
                    return self._make_result(problem)

                order = np.argsort(fitness)
                selected = order[:mu]
                z_w = np.sum(weights[:, None] * z[selected], axis=0)
                d_w = np.sum(weights[:, None] * directions[selected], axis=0)

                mean = mean + sigma * d_w
                path = path_decay * path + path_gain * z_w

                zz_w = np.einsum("i,ij,ik->jk", weights, z[selected], z[selected])
                update = (
                    c_1 * (np.outer(path, path) - np.eye(d))
                    + c_w * (zz_w - np.eye(d))
                )
                transform = transform + 0.5 * transform @ update
                sigma *= np.exp(c_s / d_sigma * (np.linalg.norm(path) / e_chi - 1.0))

                if (
                    np.isfinite(mean).all()
                    and np.isfinite(transform).all()
                    and np.isfinite(sigma)
                    and sigma > 0.0
                ):
                    continue

                numerical_restarts += 1
                if numerical_restarts > self.restarts:
                    return self._make_result(problem)
                mean = (
                    np.clip(problem.best_x, lb, ub)
                    if problem.best_x is not None
                    else lb + np.random.rand(d) * (ub - lb)
                )
                break

        return self._make_result(problem)
