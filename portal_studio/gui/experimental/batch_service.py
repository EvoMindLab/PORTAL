"""Qt-side bridge to the experiment_runner engine.

One :class:`BatchService` lives on the Experiment tab. ``submit()``
spawns a fresh :class:`QThread` that owns a :class:`BatchScheduler` (and
its :class:`ProcessPoolExecutor`); the worker emits progress / finish /
fail signals back into the main thread, which the panels consume.

Cancellation is cooperative: ``cancel(batch_id)`` flips the scheduler's
``threading.Event``; in-flight runs finish, the pool tears down, and the
worker emits ``batch_canceled``.

Multiple batches can be in flight — every ``submit`` gets its own
thread + scheduler, indexed by ``batch_id``. The main thread never
touches a ``ProcessPoolExecutor`` directly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from portal_studio.experiment_runner.config import BatchConfig
from portal_studio.experiment_runner.scheduler import (
    BatchScheduler, BatchProgress,
)


@dataclass
class BatchSubmission:
    """Bookkeeping the service hands back so callers can cancel by id."""
    batch_id: str
    config: BatchConfig
    output_dir: str
    label: str          # human-readable, shown in the Jobs table


class _BatchWorker(QObject):
    """Runs one batch end-to-end inside its QThread.

    The scheduler is owned by *this* object, not the QThread; that way
    cancel() can be called from any thread and the worker just observes
    the flag on its next progress emit.
    """

    progress = Signal(object)          # BatchProgress
    finished = Signal(object)          # BatchProgress (terminal)
    failed = Signal(str, str)          # batch_id, message
    unit_finished = Signal(str, dict)  # batch_id, {runset_uid, run_index, ...}

    def __init__(self, scheduler: BatchScheduler) -> None:
        super().__init__()
        self._scheduler = scheduler

    @Slot()
    def run(self) -> None:
        try:
            final = self._scheduler.run(
                progress=self._on_progress,
                unit_finished=self._on_unit_finished,
            )
        except Exception as exc:                                  # noqa: BLE001
            self.failed.emit(self._scheduler.batch_id, str(exc))
            return
        self.finished.emit(final)

    def _on_unit_finished(self, payload: dict) -> None:
        self.unit_finished.emit(self._scheduler.batch_id, payload)

    def _on_progress(self, snapshot: BatchProgress) -> None:
        # Copy progress snapshots before passing them between threads.
        self.progress.emit(copy.copy(snapshot))

    def cancel(self) -> None:
        self._scheduler.cancel()


class BatchService(QObject):
    """Singleton-per-tab service that submits and tracks batches."""

    # Re-emitted from worker signals so consumers don't have to subscribe
    # per worker.
    batch_progress = Signal(object)        # BatchProgress
    batch_finished = Signal(object)        # BatchProgress (terminal)
    batch_failed = Signal(str, str)        # batch_id, message
    batch_submitted = Signal(object)       # BatchSubmission
    unit_finished = Signal(str, dict)      # batch_id, run-completion payload

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # batch_id -> (QThread, _BatchWorker, BatchSubmission)
        self._active: dict[str, tuple[QThread, _BatchWorker, BatchSubmission]] = {}

    def submit(self, cfg: BatchConfig, composite: dict[str, Any],
               output_dir: str, *, label: str) -> BatchSubmission:
        scheduler = BatchScheduler(cfg, composite, output_dir)
        worker = _BatchWorker(scheduler)
        thread = QThread()
        thread.setProperty('batch_id', scheduler.batch_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Explicit QObject slots give every Python callback GUI affinity.
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.unit_finished.connect(self._on_unit_finished)
        # quit() is thread-safe. A direct connection also lets shutdown wait
        # without depending on the GUI event loop to deliver a quit request.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._forget_finished_thread)

        submission = BatchSubmission(
            batch_id=scheduler.batch_id,
            config=cfg,
            output_dir=output_dir,
            label=label,
        )
        self._active[scheduler.batch_id] = (thread, worker, submission)
        self.batch_submitted.emit(submission)
        thread.start()
        return submission

    def cancel(self, batch_id: str) -> None:
        slot = self._active.get(batch_id)
        if slot is None:
            return
        _thread, worker, _sub = slot
        worker.cancel()

    def shutdown(self) -> None:
        """Cancel and wait for every in-flight batch (on app close)."""
        for batch_id in list(self._active):
            self.cancel(batch_id)
        for batch_id in list(self._active):
            slot = self._active.get(batch_id)
            if slot is None:
                continue
            thread, _w, _s = slot
            thread.quit()
            # In-flight runs finish cooperatively. Returning on a timeout lets
            # window destruction release a still-running QThread and can abort
            # the process. Keep its owner alive until native teardown completes.
            thread.wait()

    @Slot(object)
    def _on_progress(self, snapshot: BatchProgress) -> None:
        self.batch_progress.emit(snapshot)

    @Slot(object)
    def _on_finished(self, snapshot: BatchProgress) -> None:
        self.batch_finished.emit(snapshot)

    @Slot(str, str)
    def _on_failed(self, batch_id: str, message: str) -> None:
        self.batch_failed.emit(batch_id, message)

    @Slot(str, dict)
    def _on_unit_finished(self, batch_id: str, payload: dict) -> None:
        self.unit_finished.emit(batch_id, payload)

    @Slot()
    def _forget_finished_thread(self) -> None:
        thread = self.sender()
        if not isinstance(thread, QThread):
            return
        batch_id = thread.property('batch_id')
        slot = self._active.get(batch_id)
        if slot is None or slot[0] is not thread:
            return
        # finished precedes native thread-local teardown. Retain references
        # until wait() confirms it is safe to release the Python wrappers.
        thread.wait()
        self._active.pop(batch_id)
        thread.deleteLater()
