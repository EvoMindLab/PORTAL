# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
CLPSO — Comprehensive Learning Particle Swarm Optimizer
========================================================
Reference:
    Liang, J. J., Qin, A. K., Suganthan, P. N., & Baskar, S. (2006).
    Comprehensive learning particle swarm optimizer for global
    optimization of multimodal functions.
    IEEE Transactions on Evolutionary Computation, 10(3), 281–295.
    DOI: 10.1109/TEVC.2005.857610

Reference source:
    https://github.com/P-N-Suganthan/CODES
    File: CLPSO_new_func.m
    The reference code was consulted for clarification.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    Each dimension of each particle independently selects its own
    exemplar from another particle's pbest (comprehensive learning).
    When a particle stagnates for 5 iterations without improvement,
    its exemplar table is refreshed via tournament selection.
    A special ai flag allows m random dimensions to pull toward
    gbest instead of pbest_f when exemplars are refreshed.

Parameters (exactly matching CLPSO_new_func.m):
    pop_size : swarm size               (default: 30, paper uses 30-40)
    max_gen  : max generations          (used alongside max_fe)
    c        : acceleration constant    (default: 1.49445 — from MATLAB)
    m_flag   : dims using gbest on reset(default: 0 — from MATLAB m=0)
    v_max_r  : v_max as ratio of range  (default: 0.2)
    stay_thr : stagnation threshold     (default: 5 — from MATLAB)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class CLPSO(BaseAlgorithm):
    """
    Comprehensive Learning Particle Swarm Optimizer.
    PORTAL Python implementation informed by MATLAB reference CLPSO_new_func.m.
    """

    def __init__(
        self,
        pop_size : int   = 30,       # swarm size (paper: 30-40)
        c        : float = 1.49445,  # acceleration constant (from MATLAB)
        m_flag   : int   = 0,        # dims using gbest on exemplar reset
        v_max_r  : float = 0.2,      # v_max = v_max_r * (ub - lb)
        stay_thr : int   = 5,        # stagnation threshold (from MATLAB)
    ):
        super().__init__(name='CLPSO')
        self.pop_size = pop_size
        self.c        = c
        self.m_flag   = m_flag
        self.v_max_r  = v_max_r
        self.stay_thr = stay_thr

    def run(self, problem: PortalProblem) -> dict:
        """
        Run CLPSO on a PORTAL problem.

        Args:
            problem : PortalProblem (loaded and reset by the caller)

        Returns:
            Standard result dict from self._make_result(problem)
        """
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb        # shape (d,)
        ub     = problem.ub        # shape (d,)
        max_fe = problem.max_fe

        # ── Algorithm parameters ──────────────────────────────────
        ps       = self.pop_size
        if ps < 2:
            raise ValueError("CLPSO requires pop_size >= 2.")
        if problem.remaining_fe() < 2:
            while not problem.is_done():
                problem.evaluate(problem.random_solution())
            return self._make_result(problem)
        ps       = min(ps, problem.remaining_fe())

        cc       = self.c          # same value for both components (MATLAB: cc=[1.49445 1.49445])
        m        = self.m_flag     # number of dims using gbest on reset (MATLAB: m=0)
        stay_thr = self.stay_thr   # stagnation threshold (MATLAB: 5)
        mv       = self.v_max_r * (ub - lb)   # shape (d,)

        # ── Learning probability Pc ───────────────────────────────
        # Learning-probability schedule, cross-checked against the reference:
        #   t = 0:1/(ps-1):1;  t = 5.*t;
        #   Pc = 0.0 + (0.5-0.0)*(exp(t)-exp(t(1)))/(exp(t(ps))-exp(t(1)))
        t  = np.linspace(0, 1, ps) * 5.0
        Pc = 0.0 + (0.5 - 0.0) * (np.exp(t) - np.exp(t[0])) / (np.exp(t[-1]) - np.exp(t[0]))

        # ── Inertia weight schedule ───────────────────────────────
        # MATLAB: iwt = 0.9 - (1:me)*(0.7/me)
        # We do not know me (max_gen) in advance so we use fe-based progress
        # iwt goes from 0.9 down to 0.2 linearly over the budget
        w_max = 0.9
        w_min = 0.2   # 0.9 - 0.7 = 0.2, matching MATLAB formula

        # ── Initialisation ────────────────────────────────────────
        pos = lb + (ub - lb) * np.random.rand(ps, d)
        vel = -mv + 2.0 * mv * np.random.rand(ps, d)

        # Evaluate initial population
        e = problem.evaluate_batch(pos)

        pbest    = pos.copy()       # personal best positions
        pbestval = e.copy()         # personal best fitness values

        gbest_idx = np.argmin(pbestval)
        gbest     = pbest[gbest_idx].copy()
        gbestval  = pbestval[gbest_idx]

        # ── Exemplar table initialisation ─────────────────────────
        # f_pbest[k, j] = index of particle whose pbest is used
        #                 for dimension j of particle k
        # ai[k, j]      = 1 if dimension j uses gbest instead of pbest_f
        stay_num = np.zeros(ps, dtype=int)
        ai       = np.zeros((ps, d), dtype=int)
        f_pbest  = np.tile(np.arange(ps).reshape(-1, 1), (1, d))  # default: self

        for k in range(ps):
            f_pbest[k], ai[k] = self._new_exemplar(k, ps, d, m, Pc[k], pbestval)

        # ── Main loop ─────────────────────────────────────────────
        # Use a synchronous CLPSO generation: refresh exemplars, move all
        # currently budgeted particles, batch-evaluate, then update pbest/gbest.
        while not problem.is_done():

            # Inertia weight: linear decay based on FE progress
            w = w_max - (w_max - w_min) * (problem.fe_count / max_fe)

            batch_size = min(ps, problem.remaining_fe())
            if batch_size <= 0:
                break

            idx = np.arange(batch_size)

            # ── Refresh exemplars if stagnated ────────────────────
            for k in idx:
                if stay_num[k] >= stay_thr:
                    stay_num[k] = 0
                    f_pbest[k], ai[k] = self._new_exemplar(
                        k, ps, d, m, Pc[k], pbestval
                    )

            # ── Build pbest_f: exemplar values per dimension ──────
            dims = np.arange(d)
            pbest_f = pbest[f_pbest[idx], dims]

            # ── Velocity update ───────────────────────────────────
            r1 = np.random.rand(batch_size, d)
            r2 = np.random.rand(batch_size, d)
            aa = (
                cc * (1 - ai[idx]) * r1 * (pbest_f - pos[idx])
                + cc * ai[idx] * r2 * (gbest - pos[idx])
            )
            vel[idx] = np.clip(w * vel[idx] + aa, -mv, mv)

            # ── Position update and boundary repair ───────────────
            pos[idx] = pos[idx] + vel[idx]
            out = (pos[idx] < lb) | (pos[idx] > ub)
            if np.any(out):
                pos[idx] = np.clip(pos[idx], lb, ub)
                vel_block = vel[idx]
                vel_block[out] = 0.0
                vel[idx] = vel_block

            # ── Evaluate ──────────────────────────────────────────
            f_new = problem.evaluate_batch(pos[idx])
            evaluated_idx = idx[:len(f_new)]

            # ── Update personal and global bests ──────────────────
            improved = f_new < pbestval[evaluated_idx]
            stay_num[evaluated_idx[~improved]] += 1
            if np.any(improved):
                improved_idx = evaluated_idx[improved]
                pbest[improved_idx] = pos[improved_idx].copy()
                pbestval[improved_idx] = f_new[improved]
                stay_num[improved_idx] = 0

                gbest_idx = int(np.argmin(pbestval))
                if pbestval[gbest_idx] < gbestval:
                    gbest = pbest[gbest_idx].copy()
                    gbestval = float(pbestval[gbest_idx])

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)

    # ── Helper ────────────────────────────────────────────────────

    def _new_exemplar(
        self,
        k        : int,
        ps       : int,
        d        : int,
        m        : int,
        pc_k     : float,
        pbestval : np.ndarray,
    ):
        """
        Build exemplar row for particle k.
        Construct exemplars using the published comprehensive-learning policy.

        Returns:
            f_pbest_k : shape (d,) — exemplar indices per dimension
            ai_k      : shape (d,) — 1 where gbest is used instead
        """
        # Start with self as exemplar for all dims
        f_pbest_k = np.full(d, k, dtype=int)
        ai_k      = np.zeros(d, dtype=int)

        # m random dimensions will use gbest (ai=1)
        # MATLAB: ar=randperm(D); ai(k,ar(1:m(k)))=1
        if m > 0:
            ar = np.random.permutation(d)
            ai_k[ar[:m]] = 1

        # For non-gbest dimensions, use tournament selection
        # MATLAB:
        #   fi1 = ceil(ps*rand(1,D))   — random particle indices
        #   fi2 = ceil(ps*rand(1,D))
        #   fi  = fi1 if pbestval(fi1)<pbestval(fi2) else fi2
        #   bi  = ceil(rand(1,D) - 1 + Pc(k))  — binary mask
        #   f_pbest(k,:) = bi.*fi + (1-bi).*f_pbest(k,:)
        fi1 = np.random.randint(0, ps, size=d)
        fi2 = np.random.randint(0, ps, size=d)
        fi  = np.where(pbestval[fi1] < pbestval[fi2], fi1, fi2)

        # bi: 1 with probability Pc(k), 0 otherwise
        # MATLAB: ceil(rand-1+Pc) = 1 if rand < Pc, else 0
        bi = (np.random.rand(d) < pc_k).astype(int)

        # Force at least one dimension to learn from others
        # MATLAB: if bi==zeros(1,D), rc=randperm(D); bi(rc(1))=1; end
        if not np.any(bi):
            rc = np.random.permutation(d)
            bi[rc[0]] = 1

        f_pbest_k = bi * fi + (1 - bi) * f_pbest_k

        other_dims = np.where((ai_k == 0) & (f_pbest_k == k))[0]
        if ps > 1 and len(other_dims) > 0 and len(other_dims) == d - np.sum(ai_k):
            j = int(np.random.choice(other_dims))
            others = np.array([idx for idx in range(ps) if idx != k])
            if len(others) >= 2:
                a, b = np.random.choice(others, 2, replace=False)
                f_pbest_k[j] = a if pbestval[a] < pbestval[b] else b
            else:
                f_pbest_k[j] = others[0]

        return f_pbest_k, ai_k
