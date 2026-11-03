# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
OrthoMADS Pattern Search
========================
References:
    Audet, C., & Dennis, J. E. (2006).
    Mesh adaptive direct search algorithms for constrained optimization.
    SIAM Journal on Optimization, 17(1), 188-217.
    DOI: 10.1137/040603371

    Abramson, M. A., Audet, C., Dennis, J. E., & Le Digabel, S. (2009).
    OrthoMADS: A deterministic MADS instance with orthogonal directions.
    SIAM Journal on Optimization, 20(2), 948-966.
    DOI: 10.1137/080716980

This wrapper implements a bounded, budget-controlled OrthoMADS-style
instance of Mesh Adaptive Direct Search. It polls a positive spanning set
formed from deterministic orthogonal directions in normalized coordinates.
Using orthogonal poll bases instead of fixed coordinate axes makes the
method suitable for rotated landscapes while keeping a named, citable
pattern-search lineage.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class OrthoMADSPatternSearch(BaseAlgorithm):
    """Budget-controlled OrthoMADS-style pattern search."""

    def __init__(
        self,
        initial_mesh_size: float = 0.25,
        min_mesh_size: float = 1e-6,
        expand: float = 4.0,
        shrink: float = 0.25,
    ):
        super().__init__(name='OrthoMADS Pattern Search')
        self.initial_mesh_size = initial_mesh_size
        self.min_mesh_size = min_mesh_size
        self.expand = expand
        self.shrink = shrink

    def run(self, problem: PortalProblem) -> dict:
        d = problem.d
        lb = problem.lb
        ub = problem.ub
        span = ub - lb

        if problem.is_done():
            return self._make_result(problem)

        y = np.random.rand(d)
        x = lb + y * span
        f = problem.evaluate(x)

        mesh_size = max(self.initial_mesh_size, self.min_mesh_size)
        iteration = 0

        while not problem.is_done():
            basis = self._orthomads_basis(d, iteration)
            directions = np.vstack([basis.T, -basis.T])
            directions = directions[np.lexsort(directions.T[::-1])]

            best_y = y
            best_f = f

            y_trials = []
            for direction in directions:
                y_trial = np.clip(y + mesh_size * direction, 0.0, 1.0)
                disp = y_trial - y
                if np.linalg.norm(disp) <= 0.0:
                    continue
                y_trials.append(y_trial)

            if y_trials and not problem.is_done():
                y_trials = np.asarray(y_trials, dtype=float)
                x_trials = lb + y_trials * span
                f_trials = problem.evaluate_batch(x_trials)
            else:
                y_trials = np.empty((0, d), dtype=float)
                f_trials = np.empty(0, dtype=float)

            for y_trial, f_trial in zip(y_trials[:len(f_trials)], f_trials):
                if f_trial < best_f:
                    best_y = y_trial
                    best_f = float(f_trial)

            if best_f < f:
                y = best_y
                f = best_f
                mesh_size = min(0.5, mesh_size * self.expand)
            else:
                mesh_size *= self.shrink

            iteration += 1

            if mesh_size < self.min_mesh_size and not problem.is_done():
                y = np.random.rand(d)
                x = lb + y * span
                f = problem.evaluate(x)
                mesh_size = max(self.initial_mesh_size, self.min_mesh_size)
                iteration = 0

        return self._make_result(problem)

    def _orthomads_basis(self, d: int, iteration: int) -> np.ndarray:
        seed = self._halton_vector(d, iteration + 1)
        seed = 2.0 * seed - 1.0
        if np.linalg.norm(seed) <= 0.0:
            seed[0] = 1.0

        householder = self._householder_from_vector(seed)
        perm = np.roll(np.eye(d), shift=iteration % d, axis=1)
        return householder @ perm

    def _householder_from_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= 0.0:
            return np.eye(vector.size)
        u = vector / norm
        return np.eye(vector.size) - 2.0 * np.outer(u, u)

    def _halton_vector(self, d: int, index: int) -> np.ndarray:
        primes = self._first_primes(d)
        return np.array([self._radical_inverse(index, p) for p in primes])

    def _radical_inverse(self, index: int, base: int) -> float:
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def _first_primes(self, n: int):
        primes = []
        candidate = 2
        while len(primes) < n:
            for p in primes:
                if candidate % p == 0:
                    break
                if p * p > candidate:
                    break
            else:
                primes.append(candidate)
                candidate += 1
                continue

            if all(candidate % p != 0 for p in primes if p * p <= candidate):
                primes.append(candidate)
            candidate += 1
        return primes
