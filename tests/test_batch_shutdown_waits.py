"""Closing Studio must not release a worker that outlasts the old 2s wait."""
import os
import threading
import time
import uuid

import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from portal_studio.experiment_runner.config import BatchConfig
from portal_studio.experiment_runner.scheduler import BatchProgress
from portal_studio.gui.experimental import batch_service as module


@pytest.mark.parametrize('fail', [False, True])
def test_shutdown_waits_for_slow_inflight_worker_before_return(tmp_path, monkeypatch, fail):
    app = QApplication.instance() or QApplication([])
    started, release, settled = threading.Event(), threading.Event(), threading.Event()
    marker = tmp_path / 'worker-settled.txt'
    class Scheduler:
        def __init__(self, *args):
            self.batch_id = uuid.uuid4().hex
        def cancel(self):
            pass  # In-flight work can finish normally after cancellation.
        def run(self, **kwargs):
            started.set()
            assert release.wait(10)
            marker.write_text('scientific work finished before owner destruction')
            settled.set()
            if fail:
                raise RuntimeError('late worker failure')
            return BatchProgress(self.batch_id, 'canceled', 1, 1)
    monkeypatch.setattr(module, 'BatchScheduler', Scheduler)
    service = module.BatchService()
    service.submit(BatchConfig(tasks=[]), {}, str(tmp_path), label='slow close')
    thread = next(iter(service._active.values()))[0]
    assert started.wait(5)
    timer = threading.Timer(2.6, release.set)
    timer.start()
    try:
        service.shutdown()
        running_at_return = thread.isRunning()
        settled_at_return = settled.is_set() and marker.exists()
    finally:
        release.set()
        timer.join()
        thread.wait()
        deadline = time.monotonic() + 5
        while service._active and time.monotonic() < deadline:
            QTest.qWait(2)
            time.sleep(.001)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not running_at_return, 'shutdown returned while a native QThread was still running'
    assert settled_at_return, 'shutdown returned before the in-flight result settled'
    assert not service._active
