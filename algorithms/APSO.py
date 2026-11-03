# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
APSO — Adaptive Particle Swarm Optimization
=============================================
Reference:
    Zhan, Z.-H., Zhang, J., Li, Y., & Chung, H. S.-H. (2009).
    Adaptive particle swarm optimization.
    IEEE Transactions on Systems, Man, and Cybernetics — Part B:
    Cybernetics, 39(6), 1362–1381.
    DOI: 10.1109/TSMCB.2009.2015956

Source:
    Implemented from paper equations and algorithms (primary source).
    Cross-checked against Python reimplementation by ZongSingHuang
    (https://github.com/ZongSingHuang/Adaptive-Particle-Swarm-Optimization).
    ELS passes sigma directly as the normal distribution's standard
    deviation, following paper Eq.18.

Key mechanism:
    APSO adds two components on top of standard PSO:

    1. Evolutionary State Estimation (ESE):
       At each generation, an evolutionary factor f is computed from
       the distances between particles and the global best:
           f = (d_g - d_min) / (d_max - d_min)
       where d_i = mean Euclidean distance from particle i to all others,
       d_g = distance of the global best particle, d_min/d_max = extremes.

       f is fed into four fuzzy membership functions (paper Fig. 3) to
       classify the swarm into one of four states:
           S1 = Exploration   (f ~ 0.6–0.7, particles spread out)
           S2 = Exploitation  (f ~ 0.3–0.4, converging on good region)
           S3 = Convergence   (f ~ 0.0–0.1, clustered near optimum)
           S4 = Jumping Out   (f ~ 0.9–1.0, gbest isolated from swarm)

       A fuzzy rule base (paper Table I) maps (previous state, current
       state) to the final state, providing smooth state transitions.

    2. Adaptive Parameter Control:
       Based on the identified state, c1 and c2 are adjusted each
       generation (paper Eq. 14–17):
           S1 (Exploration):  c1 += delta1,  c2 -= delta2
           S2 (Exploitation): c1 += 0.5*d1,  c2 -= 0.5*d2
           S3 (Convergence):  c1 += 0.5*d1,  c2 += 0.5*d2  + ELS
           S4 (Jumping Out):  c1 -= delta1,  c2 += delta2
       where delta ~ Uniform(0.05, 0.1). c1,c2 are clipped to [1.5,2.5]
       and renormalised if c1+c2 falls outside [3.0, 4.0].

       The inertia weight is set by a sigmoid function of f (paper Eq.13):
           w = 1 / (1 + 1.5 * exp(-2.6 * f))
       clipped to [w_min, w_max].

    3. Elitist Learning Strategy (ELS):
       When state = Convergence, gbest is perturbed in one randomly
       chosen dimension by a Gaussian with decreasing std (paper Eq.18):
           x'_gbest_d = x_gbest_d + (ub-lb) * N(0, sigma_g)
           sigma_g = 1 - 0.9 * (g / G_max)
       If the perturbed solution is better, it replaces gbest;
       otherwise it replaces the worst particle in the swarm.

Parameters (defaults follow Zhan et al. 2009):
    pop_size : swarm size               (default: 20, paper uses 20-40)
    w_max    : maximum inertia weight   (default: 0.9)
    w_min    : minimum inertia weight   (default: 0.4)
    c1_init  : initial c1               (default: 2.0)
    c2_init  : initial c2               (default: 2.0)
    v_max_r  : v_max as ratio of range  (default: 0.2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class APSO(BaseAlgorithm):
    """
    Adaptive Particle Swarm Optimization.
    Automatically adapts w, c1, c2 based on real-time evolutionary
    state estimation using fuzzy classification.
    """

    def __init__(
        self,
        pop_size : int   = 20,    # paper uses 20-40
        w_max    : float = 0.9,
        w_min    : float = 0.4,
        c1_init  : float = 2.0,
        c2_init  : float = 2.0,
        v_max_r  : float = 0.2,
    ):
        super().__init__(name='APSO')
        self.pop_size = pop_size
        self.w_max    = w_max
        self.w_min    = w_min
        self.c1_init  = c1_init
        self.c2_init  = c2_init
        self.v_max_r  = v_max_r

    def run(self, problem: PortalProblem) -> dict:
        """
        Run APSO on a PORTAL problem.

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
        N     = self.pop_size
        w_max = self.w_max
        w_min = self.w_min
        c1    = self.c1_init
        c2    = self.c2_init
        v_max = self.v_max_r * (ub - lb)   # shape (d,)

        # ── Initialisation ────────────────────────────────────────
        pos = lb + np.random.rand(N, d) * (ub - lb)
        vel = -v_max + np.random.rand(N, d) * (2 * v_max)

        fit = np.array([problem.evaluate(pos[i]) for i in range(N)])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()

        g_idx     = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_fit = pbest_fit[g_idx]

        # State tracking for fuzzy rule base
        # States: 0=Exploration(S1), 1=Exploitation(S2),
        #         2=Convergence(S3),  3=JumpingOut(S4)
        prev_state = 0   # start in Exploration

        # Generation counter for ELS sigma (paper Eq.18)
        g      = 0
        G_max  = max_fe // N   # approximate total generations

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            g += 1

            # ── Step 1: Evolutionary State Estimation ─────────────
            f = self._evolutionary_factor(pos, gbest_pos, N, d)

            # ── Step 2: Classify state via fuzzy membership ────────
            state = self._identify_state(f, prev_state)
            prev_state = state

            # ── Step 3: Adapt c1, c2 based on state ───────────────
            c1, c2 = self._update_c(c1, c2, state)

            # ── Step 4: Elitist Learning Strategy (Convergence) ───
            if state == 2:
                gbest_pos, gbest_fit, pos, fit = self._els(
                    gbest_pos, gbest_fit, pos, fit,
                    lb, ub, d, g, G_max, problem
                )
                if problem.is_done():
                    break

            # ── Step 5: Adapt inertia weight (paper Eq.13) ────────
            w = 1.0 / (1.0 + 1.5 * np.exp(-2.6 * f))
            w = np.clip(w, w_min, w_max)

            # ── Step 6: Velocity and position update ──────────────
            for i in range(N):
                if problem.is_done():
                    break

                r1 = np.random.rand(d)
                r2 = np.random.rand(d)

                vel[i] = (w * vel[i]
                          + c1 * r1 * (pbest_pos[i] - pos[i])
                          + c2 * r2 * (gbest_pos    - pos[i]))

                vel[i] = np.clip(vel[i], -v_max, v_max)
                pos[i] = np.clip(pos[i] + vel[i], lb, ub)

                f_new = problem.evaluate(pos[i])
                fit[i] = f_new

                if f_new < pbest_fit[i]:
                    pbest_fit[i] = f_new
                    pbest_pos[i] = pos[i].copy()

                    if f_new < gbest_fit:
                        gbest_fit = f_new
                        gbest_pos = pos[i].copy()

        # ── Return standard result ────────────────────────────────
        return self._make_result(problem)

    # ── Evolutionary State Estimation ─────────────────────────────

    def _evolutionary_factor(self, pos, gbest_pos, N, d):
        """
        Compute evolutionary factor f (paper Eq. 1-3).

        d_i  = mean Euclidean distance from particle i to all others
        d_g  = distance of the particle currently holding gBest
        f    = (d_g - d_min) / (d_max - d_min)

        Note: gbest_pos is passed explicitly because the particle
        holding gBest may have moved since gBest was last updated.
        Using argmin(current fit) would identify the wrong particle
        when gBest was found in a previous generation.
        """
        # Mean distance from each particle to all others (paper Eq.1)
        dist = np.zeros(N)
        for i in range(N):
            diff  = pos[i] - pos
            dists = np.sqrt(np.sum(diff**2, axis=1))
            dist[i] = np.sum(dists) / (N - 1)

        d_max = dist.max()
        d_min = dist.min()

        # Distance of gbest position to all particles (paper intent)
        # gbest_pos is the actual global best position found so far
        diff_g = gbest_pos - pos
        dists_g = np.sqrt(np.sum(diff_g**2, axis=1))
        d_g = np.sum(dists_g) / (N - 1)

        # Evolutionary factor (paper Eq.3)
        denom = d_max - d_min
        if denom < 1e-300:
            f = 0.0
        else:
            f = (d_g - d_min) / denom

        return float(f)

    # ── Fuzzy State Classification ────────────────────────────────

    def _identify_state(self, f: float, prev_state: int) -> int:
        """
        Classify swarm state using fuzzy membership functions
        and rule base (paper Fig.3 and Table I).

        States: 0=S1 Exploration, 1=S2 Exploitation,
                2=S3 Convergence,  3=S4 JumpingOut
        """
        # Fuzzy membership functions (paper Fig.3)
        # uS1: Exploration
        if   0.0 <= f <= 0.4: uS1 = 0.0
        elif 0.4 <  f <= 0.6: uS1 = 5*f - 2
        elif 0.6 <  f <= 0.7: uS1 = 1.0
        elif 0.7 <  f <= 0.8: uS1 = -10*f + 8
        else:                  uS1 = 0.0

        # uS2: Exploitation
        if   0.0 <= f <= 0.2: uS2 = 0.0
        elif 0.2 <  f <= 0.3: uS2 = 10*f - 2
        elif 0.3 <  f <= 0.4: uS2 = 1.0
        elif 0.4 <  f <= 0.6: uS2 = -5*f + 3
        else:                  uS2 = 0.0

        # uS3: Convergence
        if   0.0 <= f <= 0.1: uS3 = 1.0
        elif 0.1 <  f <= 0.3: uS3 = -5*f + 1.5
        else:                  uS3 = 0.0

        # uS4: Jumping Out
        if   0.0 <= f <= 0.7: uS4 = 0.0
        elif 0.7 <  f <= 0.9: uS4 = 5*f - 3.5
        else:                  uS4 = 1.0

        # Determine current state label from membership values
        # Priority: S3 > S2 > S1 > S4 when overlapping
        if uS3 > 0:
            current = 'S3&S2' if uS2 > 0 else 'S3'
        elif uS2 > 0:
            current = 'S2&S1' if uS1 > 0 else 'S2'
        elif uS1 > 0:
            current = 'S1&S4' if uS4 > 0 else 'S1'
        elif uS4 > 0:
            current = 'S4'
        else:
            current = 'S1'   # fallback

        # Rule base (paper Table I)
        # Rows = previous state, Columns = current state
        #              S3   S3&S2  S2   S2&S1  S1   S1&S4  S4
        rule_base = {
            0: {'S3':2, 'S3&S2':1, 'S2':1, 'S2&S1':0, 'S1':0, 'S1&S4':0, 'S4':3},  # prev=S1
            1: {'S3':2, 'S3&S2':1, 'S2':1, 'S2&S1':1, 'S1':0, 'S1&S4':0, 'S4':3},  # prev=S2
            2: {'S3':2, 'S3&S2':2, 'S2':1, 'S2&S1':1, 'S1':0, 'S1&S4':3, 'S4':3},  # prev=S3
            3: {'S3':2, 'S3&S2':2, 'S2':1, 'S2&S1':0, 'S1':0, 'S1&S4':3, 'S4':3},  # prev=S4
        }

        return rule_base[prev_state][current]

    # ── Adaptive Parameter Control ────────────────────────────────

    def _update_c(self, c1: float, c2: float, state: int):
        """
        Update c1 and c2 based on state (paper Eq.14-17).
        Clamp to [1.5, 2.5] and renormalise if sum outside [3.0, 4.0].
        """
        delta1 = np.random.uniform(0.05, 0.1)
        delta2 = np.random.uniform(0.05, 0.1)

        if state == 0:    # Exploration
            c1 += delta1
            c2 -= delta2
        elif state == 1:  # Exploitation
            c1 += 0.5 * delta1
            c2 -= 0.5 * delta2
        elif state == 2:  # Convergence
            c1 += 0.5 * delta1
            c2 += 0.5 * delta2
        elif state == 3:  # Jumping Out
            c1 -= delta1
            c2 += delta2

        # Clamp each to [1.5, 2.5]
        c1 = np.clip(c1, 1.5, 2.5)
        c2 = np.clip(c2, 1.5, 2.5)

        # Renormalise only when sum > 4.0 (paper Eq.12)
        # Paper only specifies downward normalisation to sum=4.0.
        # When sum < 3.0 (both c values at lower bound 1.5), no
        # upward normalisation is specified — leave as-is.
        if c1 + c2 > 4.0:
            total = c1 + c2
            c1 = 4.0 * c1 / total
            c2 = 4.0 * c2 / total

        return float(c1), float(c2)

    # ── Elitist Learning Strategy ─────────────────────────────────

    def _els(self, gbest_pos, gbest_fit, pos, fit,
             lb, ub, d, g, G_max, problem):
        """
        Elitist Learning Strategy (paper Eq.18-19).

        Perturbs gbest in one random dimension with Gaussian noise.
        If improved: replaces gbest.
        Otherwise:   replaces worst particle in swarm.
        """
        # Decreasing std (paper Eq.18)
        sigma_g = 1.0 - 0.9 * (g / max(G_max, 1))
        sigma_g = max(sigma_g, 1e-10)

        # Pick random dimension
        dim_idx = np.random.randint(d)

        # Perturb gbest in that dimension
        candidate = gbest_pos.copy()
        candidate[dim_idx] = (gbest_pos[dim_idx]
                              + (ub[dim_idx] - lb[dim_idx])
                              * np.random.normal(0, sigma_g))

        # Clamp to bounds
        candidate = np.clip(candidate, lb, ub)

        # Evaluate
        f_new = problem.evaluate(candidate)

        if f_new < gbest_fit:
            # Replace gbest
            gbest_pos = candidate.copy()
            gbest_fit = f_new
        else:
            # Replace worst particle in swarm (paper Fig.7: unconditional)
            # Paper always injects candidate into worst slot to maintain
            # diversity — no fitness condition is specified.
            worst_idx = np.argmax(fit)
            pos[worst_idx] = candidate.copy()
            fit[worst_idx] = f_new

        return gbest_pos, gbest_fit, pos, fit
