"""Process and numerical-thread controls shared by the command-line runners."""
from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from typing import Iterator


NUMERIC_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "TORCH_INTEROP_NUM_THREADS",
)
WINDOWS_MAX_PROCESS_WORKERS = 61
_WORKER_THREAD_LIMITER = None


def cap_process_workers(
    requested: int,
    cpu_count: int,
    *,
    task_count: int | None = None,
    platform: str | None = None,
) -> int:
    """Respect CPU, workload, and ProcessPoolExecutor platform limits."""
    workers = max(1, min(int(requested), max(1, int(cpu_count))))
    if (sys.platform if platform is None else platform) == "win32":
        workers = min(workers, WINDOWS_MAX_PROCESS_WORKERS)
    if task_count is not None:
        workers = min(workers, max(1, int(task_count)))
    return workers


@contextmanager
def numeric_worker_environment() -> Iterator[None]:
    """Make spawned workers single-threaded, restoring the caller afterwards."""
    previous = {name: os.environ.get(name) for name in NUMERIC_THREAD_VARIABLES}
    try:
        for name in NUMERIC_THREAD_VARIABLES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def single_threaded_numerics() -> Iterator[None]:
    """Also limit libraries already loaded in a serial or forking parent."""
    from threadpoolctl import threadpool_limits

    torch = sys.modules.get("torch")
    previous_torch_threads = torch.get_num_threads() if torch is not None else None
    with numeric_worker_environment(), threadpool_limits(limits=1):
        try:
            if torch is not None:
                torch.set_num_threads(1)
            yield
        finally:
            if torch is not None:
                torch.set_num_threads(previous_torch_threads)


def initialize_numeric_worker() -> None:
    """Limit inherited native thread pools for the lifetime of one worker."""
    from threadpoolctl import threadpool_limits

    global _WORKER_THREAD_LIMITER
    for name in NUMERIC_THREAD_VARIABLES:
        os.environ[name] = "1"
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
