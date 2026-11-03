# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""Success Rate-based Adaptive Differential Evolution (L-SRTDE).

Reference
---------
V. Stanovov and E. Semenkin, "Success Rate-based Adaptive Differential
Evolution L-SRTDE for CEC 2024 Competition," 2024 IEEE Congress on
Evolutionary Computation, pp. 1-8, 2024.
doi: 10.1109/CEC60901.2024.10611907

This is an independent Python implementation of Algorithm 1 in the paper,
cross-checked against the authors' public C++ implementation. It preserves
the two-population mechanism, circular injection of successful trials,
success-rate adaptation of F and pbest pressure, rank-weighted donor
selection, repaired crossover-rate memory, and linear population-size
reduction.

Two interface adaptations are deliberate. The original CEC code hard-codes
[-100, 100], whereas this implementation uses the bounds supplied by PORTAL.
The original generation loop can pass its evaluation limit; this version
stops exactly at ``problem.max_fe`` so all compared algorithms receive the
same budget.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class L_SRTDE(BaseAlgorithm):
    """L-SRTDE with the parameterization reported for CEC 2024."""

    def __init__(
        self,
        init_rate: int = 20,
        min_pop: int = 4,
        memory_size: int = 5,
        selective_pressure: float = 3.0,
    ):
        super().__init__(name="L-SRTDE")
        self.init_rate = int(init_rate)
        self.min_pop = int(min_pop)
        self.memory_size = int(memory_size)
        self.selective_pressure = float(selective_pressure)

    def run(self, problem: PortalProblem) -> dict:
        d = int(problem.d)
        lb = np.asarray(problem.lb, dtype=float)
        ub = np.asarray(problem.ub, dtype=float)
        max_fe = int(problem.max_fe)

        n_initial = self.init_rate * d
        if d <= 0 or lb.shape != (d,) or ub.shape != (d,):
            raise ValueError("L-SRTDE received inconsistent dimensions or bounds.")
        if np.any(ub <= lb):
            raise ValueError("Every upper bound must exceed its lower bound.")
        if self.init_rate <= 0:
            raise ValueError("init_rate must be positive.")
        if self.min_pop < 4 or self.min_pop > n_initial:
            raise ValueError("min_pop must lie in [4, init_rate * dimension].")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive.")
        if self.selective_pressure < 0.0:
            raise ValueError("selective_pressure must be nonnegative.")
        if max_fe < n_initial:
            raise ValueError(
                f"L-SRTDE needs at least {n_initial} evaluations for its "
                f"initial population, but the supplied budget is {max_fe}."
            )

        initial_pos = lb + np.random.rand(n_initial, d) * (ub - lb)
        initial_fit = np.asarray(
            [problem.evaluate(individual) for individual in initial_pos],
            dtype=float,
        )

        # x_top contains the best-so-far pool plus temporary successful trials.
        # x_new is the circularly updated target population. Both begin with
        # the same individuals; x_new is ordered by initial fitness exactly as
        # in the authors' implementation.
        top_pos = initial_pos.copy()
        top_fit = initial_fit.copy()
        initial_order = np.argsort(initial_fit, kind="stable")
        active_pos = initial_pos[initial_order].copy()
        active_fit = initial_fit[initial_order].copy()

        memory_cr = np.ones(self.memory_size, dtype=float)
        memory_index = 0
        success_rate = 0.5
        circular_index = 0

        while not problem.is_done():
            n = len(active_pos)
            top_order = np.argsort(top_fit, kind="stable")
            active_order = np.argsort(active_fit, kind="stable")

            rank_weights = np.exp(
                -self.selective_pressure * np.arange(n, dtype=float) / n
            )
            rank_weights /= rank_weights.sum()

            mean_f = 0.4 + 0.25 * np.tanh(5.0 * success_rate)
            pbest_count = max(2, int(n * 0.7 * np.exp(-7.0 * success_rate)))
            pbest_count = min(pbest_count, n)

            successful_pos: list[np.ndarray] = []
            successful_fit: list[float] = []
            successful_cr: list[float] = []
            fitness_improvements: list[float] = []

            for _ in range(n):
                if problem.is_done():
                    break

                target_index = int(np.random.randint(n))

                pbest_candidates = top_order[:pbest_count]
                pbest_index = self._choice_excluding(
                    pbest_candidates,
                    {target_index},
                )

                rank_position = self._weighted_rank_excluding(
                    active_order,
                    rank_weights,
                    {target_index, pbest_index},
                )
                rank_donor_index = int(active_order[rank_position])

                random_donor_index = self._choice_excluding(
                    top_order,
                    {target_index, pbest_index, rank_donor_index},
                )

                f_scale = -1.0
                # Algorithm 1 specifies the open interval (0, 1).
                while f_scale <= 0.0 or f_scale >= 1.0:
                    f_scale = float(np.random.normal(mean_f, 0.02))

                cr_memory_slot = int(np.random.randint(self.memory_size))
                crossover_rate = float(
                    np.clip(
                        np.random.normal(memory_cr[cr_memory_slot], 0.05),
                        0.0,
                        1.0,
                    )
                )

                donor = (
                    active_pos[target_index]
                    + f_scale * (top_pos[pbest_index] - active_pos[target_index])
                    + f_scale
                    * (active_pos[rank_donor_index] - top_pos[random_donor_index])
                )

                crossover_mask = np.random.rand(d) < crossover_rate
                crossover_mask[int(np.random.randint(d))] = True
                trial = np.where(crossover_mask, donor, active_pos[target_index])

                violated = (trial < lb) | (trial > ub)
                if np.any(violated):
                    n_violated = int(np.count_nonzero(violated))
                    trial[violated] = (
                        lb[violated]
                        + np.random.rand(n_violated) * (ub[violated] - lb[violated])
                    )

                actual_cr = float(np.mean(crossover_mask))
                trial_fit = float(problem.evaluate(trial))
                target_fit = float(active_fit[target_index])

                if trial_fit <= target_fit:
                    successful_pos.append(trial.copy())
                    successful_fit.append(trial_fit)
                    successful_cr.append(actual_cr)
                    # Algorithm 1, line 34: improvement is measured against
                    # the target before circular replacement.
                    fitness_improvements.append(abs(target_fit - trial_fit))

                    active_pos[circular_index] = trial
                    active_fit[circular_index] = trial_fit
                    circular_index = (circular_index + 1) % n

                # Exact PORTAL budget handling. The final evaluated trial is
                # already recorded by PortalProblem, so no partial-generation
                # state update is required for the returned result.
                if problem.is_done():
                    return self._make_result(problem)

            success_rate = len(successful_pos) / n
            next_n = int(
                (self.min_pop - n_initial) / max_fe * problem.fe_count
                + n_initial
            )
            next_n = min(n, max(self.min_pop, next_n))

            # Merge x_top and x_temp, then retain the best next_n vectors.
            if successful_pos:
                merged_pos = np.vstack((top_pos, np.asarray(successful_pos)))
                merged_fit = np.concatenate(
                    (top_fit, np.asarray(successful_fit, dtype=float))
                )
            else:
                merged_pos = top_pos
                merged_fit = top_fit
            best = np.argsort(merged_fit, kind="stable")[:next_n]
            top_pos = merged_pos[best].copy()
            top_fit = merged_fit[best].copy()

            # Remove the worst x_new entries while preserving the circular
            # population's relative order rather than sorting it physically.
            if next_n < n:
                best_active = np.argsort(active_fit, kind="stable")[:next_n]
                keep = np.zeros(n, dtype=bool)
                keep[best_active] = True
                active_pos = active_pos[keep].copy()
                active_fit = active_fit[keep].copy()
                circular_index %= next_n

            if successful_cr:
                new_memory = self._weighted_lehmer_mean(
                    np.asarray(successful_cr, dtype=float),
                    np.asarray(fitness_improvements, dtype=float),
                )
                memory_cr[memory_index] = 0.5 * (
                    memory_cr[memory_index] + new_memory
                )
                memory_index = (memory_index + 1) % self.memory_size

        return self._make_result(problem)

    @staticmethod
    def _choice_excluding(candidates: np.ndarray, excluded: set[int]) -> int:
        allowed = np.asarray(
            [int(index) for index in candidates if int(index) not in excluded],
            dtype=int,
        )
        if len(allowed) == 0:
            raise RuntimeError("L-SRTDE donor constraints left no valid candidate.")
        return int(allowed[np.random.randint(len(allowed))])

    @staticmethod
    def _weighted_rank_excluding(
        ordered_indices: np.ndarray,
        rank_weights: np.ndarray,
        excluded: set[int],
    ) -> int:
        allowed_positions = np.asarray(
            [
                rank
                for rank, index in enumerate(ordered_indices)
                if int(index) not in excluded
            ],
            dtype=int,
        )
        if len(allowed_positions) == 0:
            raise RuntimeError("L-SRTDE rank selection has no valid candidate.")
        probabilities = rank_weights[allowed_positions]
        probabilities = probabilities / probabilities.sum()
        return int(np.random.choice(allowed_positions, p=probabilities))

    @staticmethod
    def _weighted_lehmer_mean(values: np.ndarray, improvements: np.ndarray) -> float:
        """Weighted Lehmer mean used for repaired-CR memory."""
        total_improvement = float(np.sum(improvements))
        if total_improvement <= np.finfo(float).tiny:
            return 1.0
        weights = improvements / total_improvement
        denominator = float(np.sum(weights * values))
        if abs(denominator) <= 1e-8:
            return 1.0
        return float(np.sum(weights * values**2) / denominator)
