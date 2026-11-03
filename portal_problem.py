"""
PORTAL Problem Wrapper
======================
The bridge between PORTAL instances and optimization algorithms.

This is the ONLY file algorithms interact with.
They never import fitness.py or load_instance.py directly.

Usage:
    problem = PortalProblem('Instances/test_batch_1.json')
    f = problem.evaluate(x)          # call fitness
    done = problem.is_done()         # check budget
    problem.reset()                  # reset for next run
"""

import numpy as np

from portal_core.fitness import fitness, fitness_batch
from portal_core.load_instance import load_instance
from portal_core.realization import (
    realize,
    METHOD_CENTER_BETA,
    METHOD_DIVERSE,
    PORTAL_SEED_TRAIN,
    PORTAL_SEED_TEST,
)


REALIZATION_NONE = 'none'
REALIZATION_TYPE1 = 'type1'
REALIZATION_TYPE2 = 'type2'


class PortalProblem:
    """
    Standard black-box problem interface for PORTAL instances.

    Algorithms only see:
        - problem.d           : dimensionality
        - problem.lb          : lower bounds, shape (d,)
        - problem.ub          : upper bounds, shape (d,)
        - problem.max_fe      : total evaluation budget
        - problem.evaluate(x) : call fitness, returns scalar
        - problem.is_done()   : True when budget exhausted
        - problem.reset()     : reset for a fresh run
    """

    def __init__(
        self,
        instance_path: str,
        max_fe: int = None,
        record_convergence: bool = True,
        evaluation_device: str = None,
        realization_type: str = REALIZATION_NONE,
        realization_base_seed='train',
        realization_sample_id: int = 1,
    ):
        """
        Load a PORTAL instance and set up the problem.

        Args:
            instance_path : path to the .json instance file
            max_fe        : evaluation budget (default: 10_000 * d)
            record_convergence : store one (FE, best_f) point per evaluation.
                                 Batch curation can disable this when it only
                                 needs the final best value.
            evaluation_device : 'auto', 'numpy', 'cpu', 'cuda', or 'mps'.
                                'auto' tries torch in cuda -> cpu -> mps order
                                and falls back to NumPy only if torch is not
                                usable.
        """
        # ── Load the PORTAL instance ──────────────────────────────
        self.instance_id  = instance_path
        # Persistent PORTAL instances use one unified schema for both
        # single- and multi-block landscapes. Legacy composite envelopes are
        # intentionally not auto-dispatched in the normal algorithm path.
        self.instance = load_instance(instance_path)
        self.is_composite = False

        # Apply optional realization at runtime (Type 0/1/2 semantics).
        self._apply_realization_config(
            realization_type=realization_type,
            base_seed=realization_base_seed,
            sample_id=realization_sample_id,
        )

        # ── Problem dimensions ────────────────────────────────────
        self.d  = self.instance['d']
        self.lb = np.full(self.d, float(self.instance['bounds_min']))
        self.ub = np.full(self.d, float(self.instance['bounds_max']))

        # ── Evaluation budget ─────────────────────────────────────
        # Standard CEC convention: 10,000 × d
        self.max_fe = max_fe if max_fe is not None else 10_000 * self.d

        # ── Run tracking (reset between runs) ─────────────────────
        self.fe_count    = 0
        self.best_f      = np.inf
        self.best_x      = None
        self.record_convergence = bool(record_convergence)
        self.evaluation_device = evaluation_device
        self.convergence = []          # list of (fe_count, best_f) tuples

    def _resolve_realization_base_seed(self, base_seed):
        """Map user-facing realization seed selectors to integer seeds."""
        if isinstance(base_seed, str):
            key = base_seed.strip().lower()
            if key == 'train':
                return int(PORTAL_SEED_TRAIN)
            if key == 'test':
                return int(PORTAL_SEED_TEST)
            return int(base_seed)
        return int(base_seed)

    def _apply_realization_config(self, realization_type, base_seed, sample_id):
        """Apply a configured realization mode to a unified instance."""
        rtype = str(realization_type).strip().lower()
        if rtype == REALIZATION_NONE:
            return
        if rtype == REALIZATION_TYPE1:
            method = METHOD_CENTER_BETA
        elif rtype == REALIZATION_TYPE2:
            method = METHOD_DIVERSE
        else:
            raise ValueError(
                f"Unsupported realization_type={realization_type!r}. "
                f"Use {REALIZATION_NONE!r}, {REALIZATION_TYPE1!r}, or {REALIZATION_TYPE2!r}."
            )

        base_seed_value = self._resolve_realization_base_seed(base_seed)
        sid = int(sample_id)

        self.instance = realize(
            self.instance,
            base_seed_value,
            sid,
            method=method,
        )

    # ──────────────────────────────────────────────────────────────
    # Core interface
    # ──────────────────────────────────────────────────────────────

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate one candidate solution.

        This is the ONLY method algorithms should call for fitness.

        Args:
            x : solution vector, shape (d,) or (d,1)

        Returns:
            f : scalar fitness value (float)

        Raises:
            RuntimeError : if budget is already exhausted
        """
        if self.fe_count >= self.max_fe:
            raise RuntimeError(
                f"Budget exhausted: {self.fe_count}/{self.max_fe} FEs used."
            )

        x = np.asarray(x, dtype=float).flatten()
        # np.squeeze handles Form B components that return shape (1,)
        f = float(np.squeeze(fitness(x, self.instance)))
        self.instance['_last_evaluation_backend'] = 'numpy:scalar'

        self.fe_count += 1

        # Track best solution found so far
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()

        # Record convergence point when requested.
        if self.record_convergence:
            self.convergence.append((self.fe_count, self.best_f))

        return f

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of candidate solutions, consuming one FE per row.
        If the requested batch exceeds the remaining budget, only the prefix
        that fits in the budget is evaluated and returned.
        """
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif (
            samples.ndim == 2
            and samples.shape[1] != self.d
            and samples.shape[0] == self.d
        ):
            samples = samples.reshape(1, -1)
        samples = samples.reshape(samples.shape[0], -1)
        if samples.shape[1] != self.d:
            raise ValueError(
                f"Solution dimension mismatch. Expected {self.d}, got {samples.shape[1]}"
            )
        if self.fe_count >= self.max_fe or samples.shape[0] == 0:
            return np.empty(0, dtype=float)

        count = min(samples.shape[0], self.remaining_fe())
        batch = samples[:count]
        if count == 1:
            return np.asarray([self.evaluate(batch[0])], dtype=float)

        values = np.asarray(
            fitness_batch(
                batch,
                self.instance,
                evaluation_device=self.evaluation_device,
            ),
            dtype=float,
        ).reshape(-1)

        if not self.record_convergence:
            self.fe_count += int(count)
            batch_best_idx = int(np.argmin(values))
            batch_best = float(values[batch_best_idx])
            if batch_best < self.best_f:
                self.best_f = batch_best
                self.best_x = batch[batch_best_idx].copy()
            return values.astype(float, copy=False)

        for sample, value in zip(batch, values):
            f = float(np.squeeze(value))
            self.fe_count += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = sample.copy()
            if self.record_convergence:
                self.convergence.append((self.fe_count, self.best_f))

        return values.astype(float, copy=False)

    def is_done(self) -> bool:
        """Return True when the evaluation budget is fully consumed."""
        return self.fe_count >= self.max_fe

    def reset(self):
        """
        Reset all counters for a fresh independent run.
        The loaded instance is NOT reloaded — same landscape, new run.
        """
        self.fe_count    = 0
        self.best_f      = np.inf
        self.best_x      = None
        self.convergence = []

    # ──────────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────────

    def random_solution(self) -> np.ndarray:
        """Return a uniformly random solution within bounds."""
        return self.lb + np.random.rand(self.d) * (self.ub - self.lb)

    def random_population(self, n: int) -> np.ndarray:
        """
        Return n random solutions within bounds.

        Args:
            n : population size

        Returns:
            pop : shape (n, d)
        """
        return self.lb + np.random.rand(n, self.d) * (self.ub - self.lb)

    def remaining_fe(self) -> int:
        """Return remaining function evaluation budget."""
        return self.max_fe - self.fe_count

    def __repr__(self):
        return (
            f"PortalProblem("
            f"d={self.d}, "
            f"K={self.instance.get('K', self.instance.get('J'))}, "
            f"max_fe={self.max_fe}, "
            f"fe_used={self.fe_count}, "
            f"best_f={self.best_f:.6e}"
            f")"
        )
