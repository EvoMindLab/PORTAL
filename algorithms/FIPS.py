# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
FIPS — Fully Informed Particle Swarm
=====================================
Reference:
    Mendes, R., Kennedy, J., & Neves, J. (2004).
    The fully informed particle swarm: simpler, maybe better.
    IEEE Transactions on Evolutionary Computation, 8(3), 204–210.
    DOI: 10.1109/TEVC.2004.826074

Key mechanism:
    In standard PSO each particle learns from two attractors:
    its personal best and the single global best neighbour.

    FIPS replaces this with a fully informed update: every particle
    learns from ALL neighbours simultaneously. Each neighbour's pbest
    contributes to the velocity update, weighted by its fitness rank.
    There is no single "best" attractor — the pull is shared across
    the entire neighbourhood.

    Velocity update (simplified):
        v = w * v + SUM over all neighbours j of:
                phi_j * (pbest_j - x)
        where phi_j = c * rand() / N   (c split equally across N neighbours)

    This wider information sharing improves diversity and avoids the
    collapse toward a single attractor that can stall standard PSO.

Parameters (defaults follow Mendes et al. 2004):
    pop_size    : swarm size              (default: min(10*d, 50))
    w           : inertia weight          (default: 0.729)
    phi         : total acceleration      (default: 4.1)
    topology    : neighbourhood topology  ('global' or 'ring')
                  global — every particle is neighbour of all others
                  ring   — each particle has 2 neighbours (left, right)
    v_max_r     : v_max as ratio of range (default: 0.2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class FIPS(BaseAlgorithm):
    """
    Fully Informed Particle Swarm (FIPS).
    Each particle learns from all neighbours, not just the single best.
    """

    def __init__(
        self,
        pop_size : int   = None,     # None = auto: min(10*d, 50)
        w        : float = 0.729,    # inertia weight
        phi      : float = 4.1,      # total acceleration coefficient
        topology : str   = 'global', # 'global' or 'ring'
        v_max_r  : float = 0.2,      # v_max = v_max_r * (ub - lb)
    ):
        super().__init__(name='FIPS')
        self.pop_size = pop_size
        self.w        = w
        self.phi      = phi
        self.topology = topology
        self.v_max_r  = v_max_r

    def run(self, problem: PortalProblem) -> dict:
        """
        Run FIPS on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d   = problem.d
        lb  = problem.lb        # shape (d,)
        ub  = problem.ub        # shape (d,)

        # ── Algorithm parameters ──────────────────────────────────
        N     = self.pop_size if self.pop_size is not None else min(10 * d, 50)
        w     = self.w
        phi   = self.phi
        v_max = self.v_max_r * (ub - lb)   # shape (d,)

        # ── Build neighbourhood list ──────────────────────────────
        # neighbours[i] = list of particle indices that inform particle i
        # global: every particle informs every other particle
        # ring:   each particle is informed by itself + left + right neighbour
        neighbours = self._build_neighbours(N, self.topology)

        # ── Initialisation ────────────────────────────────────────
        # Positions: uniform random in [lb, ub]
        pos = lb + np.random.rand(N, d) * (ub - lb)

        # Velocities: uniform random in [-v_max, +v_max]
        vel = -v_max + np.random.rand(N, d) * (2 * v_max)

        # Evaluate initial population
        fit = np.array([problem.evaluate(pos[i]) for i in range(N)])

        # Personal bests
        pbest_pos = pos.copy()   # shape (N, d)
        pbest_fit = fit.copy()   # shape (N,)

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            for i in range(N):
                if problem.is_done():
                    break

                nb = neighbours[i]     # indices of neighbours for particle i
                n_nb = len(nb)         # number of neighbours

                # ── FIPS velocity update ──────────────────────────
                # Each neighbour j contributes:
                #   (phi / n_nb) * rand() * (pbest_j - pos_i)
                # All contributions are summed and added to inertia term.
                social_term = np.zeros(d)
                for j in nb:
                    phi_j = (phi / n_nb) * np.random.rand(d)
                    social_term += phi_j * (pbest_pos[j] - pos[i])

                vel[i] = w * vel[i] + social_term

                # Clamp velocity
                vel[i] = np.clip(vel[i], -v_max, v_max)

                # ── Position update ───────────────────────────────
                pos[i] = pos[i] + vel[i]

                # Clamp position to search bounds
                pos[i] = np.clip(pos[i], lb, ub)

                # ── Evaluate ──────────────────────────────────────
                f_new = problem.evaluate(pos[i])

                # ── Update personal best ──────────────────────────
                if f_new < pbest_fit[i]:
                    pbest_fit[i] = f_new
                    pbest_pos[i] = pos[i].copy()

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)

    # ── Helper ────────────────────────────────────────────────────

    def _build_neighbours(self, N: int, topology: str) -> list:
        """
        Build the neighbourhood list for each particle.

        Args:
            N        : swarm size
            topology : 'global' or 'ring'

        Returns:
            neighbours : list of length N, each entry is a list of indices
        """
        if topology == 'global':
            # Every particle is informed by all others (including itself)
            all_idx = list(range(N))
            return [all_idx for _ in range(N)]

        elif topology == 'ring':
            # Each particle is informed by itself, left neighbour, right neighbour
            return [
                [(i - 1) % N, i, (i + 1) % N]
                for i in range(N)
            ]

        else:
            raise ValueError(
                f"Unknown topology: '{topology}'. Use 'global' or 'ring'."
            )
