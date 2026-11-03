"""Per-step recording subclass of ``PortalProblem``.

Wraps every ``evaluate`` / ``evaluate_batch`` call to append a Frame to a tape
that the Education tab plays back.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from portal_problem import PortalProblem   # noqa: E402


@dataclass
class Frame:
    """One playback frame — particles + fitnesses at one batch evaluation."""
    positions: np.ndarray   # shape (P, d)
    fitnesses: np.ndarray   # shape (P,)
    fe_count: int           # total FEs consumed after this frame
    best_x: Optional[np.ndarray]
    best_f: float


class RecordingProblem(PortalProblem):
    """``PortalProblem`` that records each evaluation as a Frame."""

    def __init__(self, *args,
                 progress_callback: Callable[[int, int], None] | None = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tape: List[Frame] = []
        self._progress_callback = progress_callback

    def evaluate(self, x: np.ndarray) -> float:
        f = super().evaluate(x)
        self.tape.append(Frame(
            positions=np.asarray(x, dtype=float).reshape(1, -1).copy(),
            fitnesses=np.array([f], dtype=float),
            fe_count=self.fe_count,
            best_x=self.best_x.copy() if self.best_x is not None else None,
            best_f=float(self.best_f),
        ))
        self._emit_progress()
        return f

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        frames_before = len(self.tape)
        f = super().evaluate_batch(X)
        # The parent delegates a one-row batch to our scalar evaluate(),
        # which has already recorded that evaluation and emitted progress.
        if f.size > 0 and len(self.tape) == frames_before:
            X_arr = np.asarray(X, dtype=float).reshape(-1, self.d)[: len(f)]
            self.tape.append(Frame(
                positions=X_arr.copy(),
                fitnesses=np.asarray(f, dtype=float).copy(),
                fe_count=self.fe_count,
                best_x=self.best_x.copy() if self.best_x is not None else None,
                best_f=float(self.best_f),
            ))
            self._emit_progress()
        return f

    def reset(self) -> None:
        super().reset()
        self.tape = []

    def _emit_progress(self) -> None:
        if self._progress_callback is not None:
            self._progress_callback(int(self.fe_count), int(self.max_fe))
