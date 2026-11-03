# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""Triple Archives Particle Swarm Optimization (TAPSO).

Reference
---------
X. Xia, L. Gui, F. Yu, H. Wu, B. Wei, Y.-L. Zhang, and Z.-H. Zhan,
"Triple Archives Particle Swarm Optimization," IEEE Transactions on
Cybernetics, vol. 50, no. 12, pp. 4862-4875, 2020.
doi: 10.1109/TCYB.2019.2943928

This implementation follows Algorithms 1-4 and Eqs. (3)-(11) in the paper.
In particular:

* the elite and profiteer archives persist between generations;
* the profiteer score uses consecutive *current* positions and fitnesses;
* exemplar crossover is followed by per-coordinate mutation;
* the outstanding archive has maximum size N; and
* outstanding exemplars replace inferior personal-best records without new
  objective-function evaluations.

The paper says that initial velocities are random but does not specify their
distribution, and it does not state a boundary-repair rule. For bounded PORTAL
problems, velocities are initialized uniformly in half the coordinate span in
both directions and moved positions are clipped before evaluation. These two
wrapper policies are documented adaptations, not changes to TAPSO's archive
or learning mechanisms.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class TAPSO(BaseAlgorithm):
    """TAPSO with the parameter values used by Xia et al."""

    def __init__(
        self,
        pop_size: int = 40,
        w: float = 0.7298,
        crossover_probability: float = 0.5,
        mutation_probability: float = 0.02,
        archive_size: int | None = None,
    ):
        super().__init__(name="TAPSO")
        self.pop_size = int(pop_size)
        self.w = float(w)
        self.crossover_probability = float(crossover_probability)
        self.mutation_probability = float(mutation_probability)
        self.archive_size = archive_size

    def run(self, problem: PortalProblem) -> dict:
        d = int(problem.d)
        lb = np.asarray(problem.lb, dtype=float)
        ub = np.asarray(problem.ub, dtype=float)
        n = self.pop_size
        m = int(self.archive_size) if self.archive_size is not None else max(1, n // 4)

        if d <= 0 or lb.shape != (d,) or ub.shape != (d,):
            raise ValueError("TAPSO received inconsistent dimensions or bounds.")
        if np.any(ub <= lb):
            raise ValueError("Every upper bound must exceed its lower bound.")
        if n < 2:
            raise ValueError("TAPSO requires pop_size >= 2.")
        if not 1 <= m <= n:
            raise ValueError("archive_size must lie in [1, pop_size].")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must lie in [0, 1].")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1].")
        if problem.max_fe < n:
            raise ValueError(
                f"TAPSO needs at least pop_size={n} evaluations for initialization; "
                f"the supplied budget is {problem.max_fe}."
            )

        span = ub - lb
        pos = lb + np.random.rand(n, d) * span
        # The original experiments use symmetric bounds. Uniform
        # [-span/2, span/2] reproduces that natural random initialization and
        # remains meaningful for nonsymmetric PORTAL bounds.
        vel = (np.random.rand(n, d) - 0.5) * span
        fit = np.asarray([problem.evaluate(x) for x in pos], dtype=float)

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()
        best_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[best_idx].copy()
        gbest_fit = float(pbest_fit[best_idx])

        # Consecutive current-population states used by Eq. (3).
        previous_pos = pos.copy()
        previous_fit = fit.copy()

        # Persistent elite, profiteer, and outstanding archives.
        elite_pos = np.empty((0, d), dtype=float)
        elite_fit = np.empty(0, dtype=float)
        profiteer_pos = np.empty((0, d), dtype=float)
        profiteer_rate = np.empty(0, dtype=float)
        outstanding_pos = np.empty((0, d), dtype=float)
        outstanding_fit = np.empty(0, dtype=float)

        while not problem.is_done():
            distance = np.linalg.norm(previous_pos - pos, axis=1)
            # exp(distance) overflows for distances above about 710; the
            # corresponding improvement rate is numerically zero anyway.
            improvement_rate = (previous_fit - fit) / np.exp(np.minimum(distance, 700.0))

            elite_pos, elite_fit = self._update_ranked_archive(
                elite_pos,
                elite_fit,
                pos,
                fit,
                m,
                higher_is_better=False,
            )
            profiteer_pos, profiteer_rate = self._update_ranked_archive(
                profiteer_pos,
                profiteer_rate,
                pos,
                improvement_rate,
                m,
                higher_is_better=True,
            )

            elite_prob = self._rank_probabilities(len(elite_pos))
            profiteer_prob = self._rank_probabilities(len(profiteer_pos))

            # Save the current generation before particles move. These arrays
            # become the previous state for Eq. (3) in the next generation.
            generation_pos = pos.copy()
            generation_fit = fit.copy()

            for i in range(n):
                if problem.is_done():
                    break

                exemplar = self._breed_exemplar(
                    elite_pos,
                    profiteer_pos,
                    elite_prob,
                    profiteer_prob,
                    lb,
                    ub,
                )
                exemplar_fit = float(problem.evaluate(exemplar))
                old_pbest_fit = float(pbest_fit[i])

                # If the budget ends after evaluating the exemplar, that
                # evaluation still legitimately contributes to the reported
                # best-so-far value, but no incomplete particle move is made.
                if problem.is_done():
                    return self._make_result(problem)

                r1 = np.random.rand(d)
                r2 = np.random.rand(d)
                if exemplar_fit < gbest_fit:
                    # Diffident model, Eq. (9).
                    vel[i] = self.w * vel[i] + r1 * (exemplar - pos[i])
                    gbest_pos = exemplar.copy()
                    gbest_fit = exemplar_fit
                elif exemplar_fit < pbest_fit[i]:
                    # Mild model, Eq. (10).
                    vel[i] = (
                        self.w * vel[i]
                        + r1 * (exemplar - pos[i])
                        + r2 * (gbest_pos - pos[i])
                    )
                else:
                    # Confident model, Eq. (11).
                    vel[i] = self.w * vel[i] + r1 * (pbest_pos[i] - pos[i])

                pos[i] = np.clip(pos[i] + vel[i], lb, ub)
                fit[i] = float(problem.evaluate(pos[i]))

                if fit[i] < pbest_fit[i]:
                    pbest_pos[i] = pos[i].copy()
                    pbest_fit[i] = fit[i]
                if fit[i] < gbest_fit:
                    gbest_pos = pos[i].copy()
                    gbest_fit = float(fit[i])

                # Algorithm 4 compares E_i with PB_i from before this move.
                if exemplar_fit < old_pbest_fit:
                    outstanding_pos, outstanding_fit = self._update_outstanding(
                        outstanding_pos,
                        outstanding_fit,
                        exemplar,
                        exemplar_fit,
                        n,
                    )

            previous_pos = generation_pos
            previous_fit = generation_fit

            # Algorithm 3 changes only PB records and consumes no FEs.
            pbest_pos, pbest_fit = self._reuse_outstanding(
                outstanding_pos,
                outstanding_fit,
                pbest_pos,
                pbest_fit,
            )
            best_idx = int(np.argmin(pbest_fit))
            if pbest_fit[best_idx] < gbest_fit:
                gbest_pos = pbest_pos[best_idx].copy()
                gbest_fit = float(pbest_fit[best_idx])

        return self._make_result(problem)

    @staticmethod
    def _rank_probabilities(size: int) -> np.ndarray:
        """Equation (6): linearly decreasing rank probabilities."""
        ranks = np.arange(size, 0, -1, dtype=float)
        return ranks / ranks.sum()

    @staticmethod
    def _roulette(probabilities: np.ndarray) -> int:
        cumulative = np.cumsum(probabilities)
        return min(int(np.searchsorted(cumulative, np.random.rand(), side="right")), len(probabilities) - 1)

    def _breed_exemplar(
        self,
        elite_pos: np.ndarray,
        profiteer_pos: np.ndarray,
        elite_prob: np.ndarray,
        profiteer_prob: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Algorithm 1: rank selection, crossover, and mutation."""
        d = len(lb)
        exemplar = np.empty(d, dtype=float)
        for j in range(d):
            elite_index = self._roulette(elite_prob)
            profiteer_index = self._roulette(profiteer_prob)
            if np.random.rand() < self.crossover_probability:
                exemplar[j] = profiteer_pos[profiteer_index, j]
            else:
                exemplar[j] = elite_pos[elite_index, j]
            if np.random.rand() < self.mutation_probability:
                exemplar[j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
        return exemplar

    @staticmethod
    def _update_ranked_archive(
        old_pos: np.ndarray,
        old_score: np.ndarray,
        current_pos: np.ndarray,
        current_score: np.ndarray,
        capacity: int,
        *,
        higher_is_better: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge, rank, deduplicate, and truncate A_e or A_p."""
        positions = np.vstack((old_pos, current_pos))
        scores = np.concatenate((old_score, current_score))
        order = np.argsort(-scores if higher_is_better else scores, kind="stable")

        kept_positions: list[np.ndarray] = []
        kept_scores: list[float] = []
        seen: set[bytes] = set()
        for index in order:
            key = np.ascontiguousarray(positions[index]).tobytes()
            if key in seen:
                continue
            seen.add(key)
            kept_positions.append(positions[index].copy())
            kept_scores.append(float(scores[index]))
            if len(kept_positions) == capacity:
                break

        return np.asarray(kept_positions, dtype=float), np.asarray(kept_scores, dtype=float)

    @staticmethod
    def _update_outstanding(
        archive_pos: np.ndarray,
        archive_fit: np.ndarray,
        exemplar: np.ndarray,
        exemplar_fit: float,
        capacity: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 4, lines 13-18: tournament update of A_o."""
        if len(archive_fit) < capacity:
            if len(archive_fit) == 0:
                return exemplar.reshape(1, -1).copy(), np.asarray([exemplar_fit], dtype=float)
            return (
                np.vstack((archive_pos, exemplar)),
                np.append(archive_fit, exemplar_fit),
            )

        a, b = np.random.choice(len(archive_fit), size=2, replace=False)
        poorer = a if archive_fit[a] >= archive_fit[b] else b
        archive_pos = archive_pos.copy()
        archive_fit = archive_fit.copy()
        archive_pos[poorer] = exemplar
        archive_fit[poorer] = exemplar_fit
        return archive_pos, archive_fit

    @staticmethod
    def _reuse_outstanding(
        archive_pos: np.ndarray,
        archive_fit: np.ndarray,
        pbest_pos: np.ndarray,
        pbest_fit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 3: pair A_o and replace the worst PB records."""
        n_pairs = len(archive_fit) // 2
        if n_pairs == 0:
            return pbest_pos, pbest_fit

        shuffled = np.random.permutation(len(archive_fit))
        winners = []
        for pair_index in range(n_pairs):
            a = int(shuffled[2 * pair_index])
            b = int(shuffled[2 * pair_index + 1])
            winners.append(a if archive_fit[a] <= archive_fit[b] else b)

        # PBs are ordered from worst to best as required by Algorithm 3.
        inferior_order = np.argsort(pbest_fit)[::-1]
        new_pos = pbest_pos.copy()
        new_fit = pbest_fit.copy()
        for rank, winner in enumerate(winners):
            particle = int(inferior_order[rank])
            if archive_fit[winner] < new_fit[particle]:
                new_pos[particle] = archive_pos[winner]
                new_fit[particle] = archive_fit[winner]
        return new_pos, new_fit
