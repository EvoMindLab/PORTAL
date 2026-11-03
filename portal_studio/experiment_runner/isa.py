"""Deprecated import alias; use ``experiment_runner.performance`` instead.

Only compatibility names retain ISA terminology. This file performs no
Instance Space Analysis and is not used by the current Studio workflow.
"""
from .performance import (
    PERFORMANCE_CHECKPOINTS_FE,
    PERFORMANCE_LOG_EPS,
    checkpoint_best_so_far,
    median_log_error,
)

ISA_CHECKPOINTS_FE = PERFORMANCE_CHECKPOINTS_FE
ISA_LOG_EPS = PERFORMANCE_LOG_EPS
