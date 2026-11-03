# License scope: algorithms/README.md; excluded from PORTAL's root MIT grant.
"""
EA4eig — Ensemble of four EAs with Eigenspace Decomposition
=============================================================
Reference:
    Bujok, P. (2022).
    EA4 algorithm with eigenspace decomposition for CEC2022
    benchmark problems.
    Proceedings of the 2022 IEEE Congress on Evolutionary
    Computation (CEC 2022).

Reference source:
    Official CEC2022 bundle, CEC2022_EA4eig:
    Run_EA4eig.m, roulete.m, nahvyb.m,
    nahvyb_expt.m, zrcad.m, cauchy_rnd.m
    The reference code was consulted for clarification.
    CEC 2022 competition entry.

Provenance and licensing:
    Paper-based PORTAL implementation, as confirmed by maintainer Danial
    Yazdani on November 3, 2026. Reference code was consulted for clarification.
    See algorithms/PROVENANCE.md; the scientific and source credits are retained.

Key mechanism:
    EA4eig maintains ONE shared population P and routes each
    generation to one of four component algorithms, selected by
    a roulette wheel whose weights are reinforced by success.

    FOUR COMPONENT ALGORITHMS:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. CoBiDE  – DE/rand/1 + per-individual Cauchy F/CR    │
    │ 2. IDE     – Improved DE with rank-based F, two stages  │
    │ 3. CMA-ES  – full (μ/μ_w,λ)-CMA-ES                    │
    │ 4. jSO     – jSO-style DE/current-to-pbest/1           │
    └─────────────────────────────────────────────────────────┘
    All four share the same population P and optionally use the
    same eigenvector coordinate system (computed from the top
    50% of P, probability peig=0.4 per generation).

    ROULETTE SELECTION (roulete.m):
        ni[k] = success count for algorithm k, init=n0=2
        prob[k] ∝ ni[k] / sum(ni)
        If min(ni)/sum(ni) < δ=1/(5·h): reset all ni=n0

    BOUNDARY:
        CoBiDE/jSO: reflection (zrcad.m) — iterative mirroring
        IDE:        random resample for out-of-bounds dims
        CMA-ES:     midpoint (out-of-bounds → (old+bound)/2)

    POPULATION SIZE REDUCTION (linear):
        optN = round(((Nmin−N_init)/maxFES)·FES + N_init)
        N_init=100, Nmin=10; worst removed after sorting.

Parameters (exactly matching Run_EA4eig.m):
    N_init   : initial population size  (default: 100)
    N_min    : minimum population size  (default: 10)
    peig     : eigenvector crossover prob (default: 0.4)
    CBps     : fraction used for covariance (default: 0.5)
    n0       : initial roulette count   (default: 2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portal_problem import PortalProblem
from base_algorithm import BaseAlgorithm


class EA4eig(BaseAlgorithm):
    """
    EA4eig: Ensemble of four EAs with Eigenspace Decomposition.
    PORTAL Python implementation informed by the Run_EA4eig.m reference.
    CEC 2022 competition entry.
    """

    def __init__(
        self,
        N_init : int   = 100,
        N_min  : int   = 10,
        peig   : float = 0.4,
        CBps   : float = 0.5,
        n0     : int   = 2,
    ):
        super().__init__(name='EA4eig')
        self.N_init = N_init
        self.N_min  = N_min
        self.peig   = peig
        self.CBps   = CBps
        self.n0     = n0

    # ── Helpers ──────────────────────────────────────────────────

    def _cauchy(self, x0, gamma):
        return x0 + gamma * np.tan(np.pi * (np.random.rand() - 0.5))

    def _zrcad(self, y, lb, ub):
        """Reflection boundary (zrcad.m). Works element-wise."""
        y = y.copy()
        for _ in range(100):
            above = y > ub
            below = y < lb
            if not (np.any(above) or np.any(below)):
                break
            y[above] = 2 * ub[above] - y[above]
            y[below] = 2 * lb[below] - y[below]
        return np.clip(y, lb, ub)

    def _nahvyb(self, N, k):
        """Sample k distinct indices from [0, N) uniformly."""
        return np.random.choice(N, k, replace=False)

    def _nahvyb_expt(self, N, k, excl):
        """Sample k distinct indices from [0,N), excluding those in excl."""
        pool = np.setdiff1d(np.arange(N), np.atleast_1d(excl))
        return np.random.choice(pool, min(k, len(pool)), replace=False)

    def _roulete(self, ni):
        """Roulette selection proportional to ni (roulete.m).
        Paper Eq.15 adds n0 offset: prob[h]=(ni[h]+n0)/Σ(nj+n0).
        MATLAB omits the offset; uses reset when p_min<δ instead.
        This matches the MATLAB exactly.
        """
        s = ni.sum()
        p_min = ni.min() / s
        cp = np.cumsum(ni) / s
        r = np.random.rand()
        return int(np.searchsorted(cp, r)), p_min

    def _eigenvectors(self, pop, CBps):
        """Compute eigenvectors from top CBps fraction of pop."""
        n_sel = max(2, int(round(len(pop) * CBps)))
        sub = pop[:n_sel]   # pop already sorted best-first
        C = np.cov(sub.T)
        if C.ndim == 0:
            C = np.array([[float(C)]])
        _, R = np.linalg.eigh(C)
        return R

    def _eig_crossover(self, R, y, v, cr):
        """Crossover in eigenvector coordinate system."""
        d = len(y)
        yeig = R.T @ y
        veig = R.T @ v
        change = np.where(np.random.rand(d) < cr)[0]
        if len(change) == 0:
            change = np.array([np.random.randint(d)])
        yeig[change] = veig[change]
        result = R @ yeig
        # Note: inverse rotation can push out of bounds; caller must clip.
        return result

    def run(self, problem: PortalProblem) -> dict:
        # ── Problem info ──────────────────────────────────────────
        d      = problem.d
        lb     = problem.lb
        ub     = problem.ub
        max_fe = problem.max_fe

        N_init = self.N_init
        N_min  = self.N_min
        peig   = self.peig
        CBps   = self.CBps
        n0     = self.n0
        h      = 4   # number of component algorithms

        # ── Initialisation ────────────────────────────────────────
        N   = N_init
        pop = lb + np.random.rand(N, d) * (ub - lb)
        fit = np.array([problem.evaluate(pop[i]) for i in range(N)])
        nfes = N

        # Sort best-first
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]

        # ── Per-individual F and CR (CoBiDE / IDE) ────────────────
        CBF  = np.zeros(N)
        CBCR = np.zeros(N)
        for i in range(N):
            CBF[i]  = self._init_cbf()
            CBCR[i] = self._init_cbcr()

        # ── Roulette counts ───────────────────────────────────────
        ni    = np.full(h, float(n0))
        delta = 1.0 / (5 * h)

        # ── jSO state ─────────────────────────────────────────────
        Asize_max = int(round(N * 2.6))
        H_jso = 5
        MF  = np.full(H_jso, 0.3)
        MCR = np.full(H_jso, 0.8)
        MF[H_jso - 1]  = 0.9
        MCR[H_jso - 1] = 0.9
        k_jso = 0
        A     = np.empty((0, d))
        Asize = 0
        pmax  = 0.25
        pmin  = pmax / 2

        # ── IDE state ─────────────────────────────────────────────
        gmax = max(1, int(round(max_fe / N_init)))
        T    = max(1, gmax // 10)
        GT   = gmax // 2
        gt   = GT
        g    = 0
        Tcurr = 0

        # ── CMA-ES state ──────────────────────────────────────────
        sigma   = (ub[0] - lb[0]) / 2.0
        oldPop  = pop.T.copy()
        myeps   = 1e-15

        mu_cma   = N // 2
        weights  = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
        weights /= weights.sum()
        mueff    = weights.sum()**2 / (weights**2).sum()

        cc   = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs   = (mueff + 2) / (d + mueff + 5)
        c1   = 2 / ((d + 1.3)**2 + mueff)
        cmu  = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        pc   = np.zeros(d)
        ps   = np.zeros(d)
        B    = np.eye(d)
        D    = np.ones(d)
        CC   = np.eye(d)
        invsqrtC = np.eye(d)
        eigeneval = 0
        chiN = d**0.5 * (1 - 1/(4*d) + 1/(21*d**2))

        # ── Main loop ─────────────────────────────────────────────
        while not problem.is_done():

            # Roulette selection
            hh, p_min = self._roulete(ni)
            if p_min < delta:
                ni[:] = float(n0)

            # Optionally compute eigenvectors (shared)
            use_eig = (np.random.rand() < peig)
            R = None
            if use_eig:
                n_sel = max(2, int(round(N * CBps)))
                if n_sel >= 2:
                    R = self._eigenvectors(pop[:N], CBps)

            # ── Case 1: CoBiDE ────────────────────────────────────
            if hh == 0:
                for i in range(N):
                    if problem.is_done():
                        break
                    excl = [i]
                    idx  = self._nahvyb_expt(N, 3, excl)
                    r1, r2, r3 = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                    v = r1 + CBF[i] * (r2 - r3)
                    v = self._zrcad(v, lb, ub)

                    y = pop[i].copy()
                    if use_eig and R is not None:
                        y_new = np.clip(self._eig_crossover(R, y, v, CBCR[i]), lb, ub)
                    else:
                        change = np.where(np.random.rand(d) < CBCR[i])[0]
                        if len(change) == 0:
                            change = np.array([np.random.randint(d)])
                        y_new = y.copy()
                        y_new[change] = v[change]
                    y_new = self._zrcad(y_new, lb, ub)

                    fy = problem.evaluate(y_new)
                    nfes += 1
                    if fy <= fit[i]:
                        pop[i] = y_new
                        fit[i] = fy
                        ni[hh] += 1
                    else:
                        CBF[i]  = self._init_cbf()
                        CBCR[i] = self._init_cbcr()

            # ── Case 2: IDE ───────────────────────────────────────
            elif hh == 1:
                # Sort best-first, carry CBF/CBCR
                order   = np.argsort(fit[:N])
                pop[:N] = pop[:N][order]
                fit[:N] = fit[:N][order]
                CBF[:N] = CBF[:N][order]
                CBCR[:N] = CBCR[:N][order]

                IDEps = 0.1 + 0.9 * 10**(5 * (g / max(gmax, 1) - 1))
                pd    = 0.1 * IDEps
                SRT   = 0.1 if g >= gt else 0.0

                Q = pop[:N].copy()
                for i in range(N):
                    excl = [i]
                    idx  = self._nahvyb_expt(N, 4, excl)
                    o_idx = i if g <= gt else idx[0]
                    r1_i, r2_i, r3_i = idx[1], idx[2], idx[3]

                    xo  = pop[o_idx].copy()
                    xr1 = pop[r1_i].copy()
                    xr2 = pop[r2_i].copy()
                    xr3 = pop[r3_i].copy()

                    # Perturbation of xr3
                    indperturb = np.where(np.random.rand(d) < pd)[0]
                    if len(indperturb) > 0:
                        xr3[indperturb] = (
                            lb[indperturb]
                            + np.random.rand(len(indperturb))
                            * (ub[indperturb] - lb[indperturb])
                        )

                    # Rank-based Fo
                    Fo = (o_idx + 1) / N + 0.1 * np.random.randn()
                    while Fo <= 0 or Fo > 1:
                        Fo = (o_idx + 1) / N + 0.1 * np.random.randn()

                    # Superior/Inferior logic
                    high_ind = max(1, int(IDEps * N))
                    if o_idx >= high_ind and r1_i >= high_ind:
                        cands = np.setdiff1d(np.arange(high_ind),
                                             [o_idx, r1_i, r2_i, r3_i, i])
                        if len(cands) > 0:
                            r1_i  = int(np.random.choice(cands))
                            xr1   = pop[r1_i].copy()

                    if g > gt and np.random.rand() < 0.5:
                        Q[i] = pop[i] + Fo * (xr1 - xo) + Fo * (xr2 - xr3)
                    else:
                        Q[i] = xo + Fo * (xr1 - xo) + Fo * (xr2 - xr3)

                # Crossover
                for i in range(N):
                    v = Q[i]
                    y = pop[i].copy()
                    if use_eig and R is not None:
                        y_new = np.clip(self._eig_crossover(R, y, v, CBCR[i]), lb, ub)
                    else:
                        CR = (i + 1) / N + 0.1 * np.random.randn()
                        CR = float(np.clip(CR, 0.0, 1.0))
                        j_rand = np.random.randint(d)
                        y_new  = y.copy()
                        for j in range(d):
                            if np.random.rand() <= CR or j == j_rand:
                                if Q[i, j] < lb[j] or Q[i, j] > ub[j]:
                                    y_new[j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                                else:
                                    y_new[j] = Q[i, j]
                    Q[i] = y_new

                # Evaluate and select
                q_fit  = np.array([problem.evaluate(Q[i])
                                   if not problem.is_done() else fit[i]
                                   for i in range(N)])
                nfes  += N
                succ   = np.where(q_fit <= fit[:N])[0]
                ni[hh] += len(succ)
                SR     = len(succ) / N
                pop[:N][succ] = Q[succ]
                fit[:N][succ] = q_fit[succ]

                # Stage transition logic
                if g < gt:
                    if SR <= SRT:
                        Tcurr += 1
                    else:
                        Tcurr = 0
                    if Tcurr >= T:
                        gt = g

                order   = np.argsort(fit[:N])
                pop[:N] = pop[:N][order]
                fit[:N] = fit[:N][order]
                CBF[:N] = CBF[:N][order]
                CBCR[:N] = CBCR[:N][order]
                g += 1

            # ── Case 3: CMA-ES ────────────────────────────────────
            elif hh == 2:
                order   = np.argsort(fit[:N])
                pop[:N] = pop[:N][order]
                fit[:N] = fit[:N][order]
                CBF[:N] = CBF[:N][order]
                CBCR[:N] = CBCR[:N][order]

                mu_cma = max(2, N // 2)
                if len(weights) != mu_cma:
                    weights = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
                    weights /= weights.sum()
                    mueff = weights.sum()**2 / (weights**2).sum()

                xmean = pop[:mu_cma].T @ weights

                new_pop  = np.empty((d, N))
                new_fits = np.empty(N)
                for kk in range(N):
                    if problem.is_done():
                        break
                    x = xmean + sigma * B @ (D * np.random.randn(d))
                    # Midpoint boundary
                    below = x < lb
                    above = x > ub
                    if np.any(below):
                        x[below] = (oldPop[below, kk % oldPop.shape[1]]
                                    + lb[below]) / 2
                    if np.any(above):
                        x[above] = (oldPop[above, kk % oldPop.shape[1]]
                                    + ub[above]) / 2
                    x = np.clip(x, lb, ub)

                    fx = problem.evaluate(x)
                    nfes += 1
                    new_pop[:, kk]  = x
                    new_fits[kk]    = fx

                    # Replace worst in P if improvement
                    worst_i = int(np.argmax(fit[:N]))
                    if fx < fit[worst_i]:
                        pop[worst_i] = x
                        fit[worst_i] = fx
                        ni[hh] += 1

                # CMA-ES update
                fit_order = np.argsort(new_fits)
                xold  = xmean.copy()
                xmean = new_pop[:, fit_order[:mu_cma]] @ weights
                oldPop = new_pop

                ps = ((1 - cs) * ps
                      + np.sqrt(cs * (2 - cs) * mueff)
                      * invsqrtC @ (xmean - xold) / sigma)
                hsig = (np.sum(ps**2) / (1 - (1-cs)**(2*nfes/N)) / d
                        < 2 + 4/(d+1))
                pc = ((1 - cc) * pc
                      + hsig * np.sqrt(cc * (2-cc) * mueff)
                      * (xmean - xold) / sigma)

                artmp = (1/sigma) * (new_pop[:, fit_order[:mu_cma]]
                                     - xold[:, np.newaxis])
                CC = ((1 - c1 - cmu) * CC
                      + c1 * (np.outer(pc, pc)
                              + (1 - hsig) * cc * (2-cc) * CC)
                      + cmu * artmp @ np.diag(weights) @ artmp.T)

                sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
                if sigma > 1e300 or sigma < 1e-300 or np.isnan(sigma):
                    sigma = (ub[0] - lb[0]) / 2.0

                if nfes - eigeneval > N / (c1 + cmu) / d / 10:
                    eigeneval = nfes
                    CC = np.triu(CC) + np.triu(CC, 1).T
                    eigvals, B = np.linalg.eigh(CC)
                    D = np.sqrt(np.maximum(eigvals, 1e-20))
                    invsqrtC = B @ np.diag(1.0/D) @ B.T

                # Early stop if ill-conditioned
                if D.max() > 1e7 * D.min():
                    sigma = (ub[0] - lb[0]) / 2.0
                    CC = np.eye(d)
                    B  = np.eye(d)
                    D  = np.ones(d)
                    invsqrtC = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)

            # ── Case 4: jSO ───────────────────────────────────────
            elif hh == 3:
                pp = pmax - (pmax - pmin) * (nfes / max_fe)
                Q  = pop[:N].copy()
                Fpole  = np.full(N, -1.0)
                CRpole = np.full(N, -1.0)
                SCR = []
                SF  = []
                deltafce = np.full(N, -1.0)
                suc = 0

                for i in range(N):
                    rr = int(np.random.randint(H_jso))
                    CR = MCR[rr] + np.sqrt(0.1) * np.random.randn()
                    CR = float(np.clip(CR, 0.0, 1.0))
                    t  = nfes / max_fe
                    if t < 0.25:
                        CR = max(CR, 0.7)
                    elif t < 0.5:
                        CR = max(CR, 0.6)

                    F = -1.0
                    while F <= 0:
                        F = 0.1 * np.tan(np.random.rand() * np.pi - np.pi/2) + MF[rr]
                    F = min(F, 1.0)
                    if t < 0.6 and F > 0.7:
                        F = 0.7

                    Fpole[i]  = F
                    CRpole[i] = CR

                    p = max(2, int(np.ceil(pp * N)))
                    xi = pop[i]
                    # pbest
                    ktery = np.random.randint(p)
                    xpbest = pop[ktery]

                    # r1 from pop ≠ i
                    excl = [i]
                    r1_idx = int(self._nahvyb_expt(N, 1, excl)[0])
                    r1 = pop[r1_idx]
                    excl.append(r1_idx)

                    # r2 from pop+archive ≠ i,r1
                    combined = (np.vstack([pop[:N], A])
                                if Asize > 0 else pop[:N])
                    NA = N + Asize
                    excl_arr = np.array(excl)
                    pool = np.setdiff1d(np.arange(NA), excl_arr)
                    r2_idx = int(np.random.choice(pool))
                    r2 = combined[r2_idx]

                    t = nfes / max_fe
                    if t < 0.2:
                        Fw = 0.7 * F
                    elif t < 0.4:
                        Fw = 0.8 * F
                    else:
                        Fw = 1.2 * F

                    v = xi + Fw * (xpbest - xi) + F * (r1 - r2)

                    if use_eig and R is not None:
                        y_new = np.clip(self._eig_crossover(R, xi.copy(), v, CBCR[i]), lb, ub)
                    else:
                        y = xi.copy()
                        change = np.where(np.random.rand(d) < CR)[0]
                        if len(change) == 0:
                            change = np.array([np.random.randint(d)])
                        y[change] = v[change]
                        y_new = y
                    Q[i] = self._zrcad(y_new, lb, ub)

                # Evaluate jSO offspring
                q_fit = np.array([problem.evaluate(Q[i])
                                  if not problem.is_done() else fit[i]
                                  for i in range(N)])
                nfes += N

                for i in range(N):
                    if q_fit[i] < fit[i]:
                        deltafce[i] = fit[i] - q_fit[i]
                        suc += 1
                        # archive
                        if Asize < Asize_max:
                            A = np.vstack([A, pop[i]])
                            Asize += 1
                        else:
                            rnd = np.random.randint(Asize)
                            A[rnd] = pop[i]
                        SCR.append(CRpole[i])
                        SF.append(Fpole[i])
                    if q_fit[i] <= fit[i]:
                        pop[i] = Q[i]
                        fit[i] = q_fit[i]

                # Memory update (Janez averaging)
                if suc > 0:
                    ni[hh] += suc
                    valid = deltafce[deltafce >= 0]
                    if valid.sum() > 0:
                        w = valid / valid.sum()
                        sf_arr  = np.array(SF)
                        scr_arr = np.array(SCR)
                        old_cr  = MCR[k_jso]
                        old_f   = MF[k_jso]
                        new_f   = (w * sf_arr**2).sum() / ((w * sf_arr).sum() + 1e-300)
                        if scr_arr.max() == 0 or MCR[k_jso] == -1:
                            MCR[k_jso] = -1
                        else:
                            new_cr = (w * scr_arr**2).sum() / ((w * scr_arr).sum() + 1e-300)
                            MCR[k_jso] = (new_cr + old_cr) / 2
                        MF[k_jso] = (new_f + old_f) / 2
                        k_jso = (k_jso + 1) % H_jso

            # ── Linear population size reduction ──────────────────
            opt_N = max(N_min,
                        int(round(((N_min - N_init) / max_fe) * nfes + N_init)))
            if N > opt_N:
                diff = min(N - opt_N, N - N_min)
                if diff > 0:
                    order   = np.argsort(fit[:N])
                    pop[:N] = pop[:N][order]
                    fit[:N] = fit[:N][order]
                    CBF[:N] = CBF[:N][order]
                    CBCR[:N] = CBCR[:N][order]
                    N -= diff
                    Asize_max = int(round(N * 2.6))
                    while Asize > Asize_max:
                        rnd = np.random.randint(Asize)
                        A = np.delete(A, rnd, axis=0)
                        Asize -= 1
                    # Resize CMA-ES weights
                    mu_cma = max(2, N // 2)
                    weights = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
                    weights /= weights.sum()
                    mueff = weights.sum()**2 / (weights**2).sum()

        return self._make_result(problem)

    def _init_cbf(self):
        """Initialise per-individual F (CoBiDE)."""
        while True:
            if np.random.rand() < 0.5:
                f = self._cauchy(0.65, 0.1)
            else:
                f = self._cauchy(1.0, 0.1)
            if f > 0:
                return min(f, 1.0)

    def _init_cbcr(self):
        """Initialise per-individual CR (CoBiDE)."""
        if np.random.rand() < 0.5:
            cr = self._cauchy(0.1, 0.1)
        else:
            cr = self._cauchy(0.95, 0.1)
        return float(np.clip(cr, 0.0, 1.0))
