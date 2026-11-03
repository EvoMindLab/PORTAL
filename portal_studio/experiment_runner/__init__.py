"""Batch-experiment engine for the Experimentation tab.

This package is the pure-Python engine layer (no Qt) that drives the
Experimentation tab. It does not own the solvers — those live in
``portal_studio.algorithms`` and are reused through the existing
``SOLVERS`` registry and ``run_solver_on_instance`` runner. What this
package adds on top is the **batch** layer:

- :mod:`config`     batch + task dataclasses + JSONL load/save
- :mod:`store`      per-batch sqlite result store (summary + trend DBs)
- :mod:`runner`     per-run executor that adapts ``run_solver_on_instance``
                    into the worker-process entry point
- :mod:`scheduler`  ProcessPoolExecutor wrapper that fans a batch into runs

The Qt layer (``portal_studio.gui.experimental``) drives this engine through a
``QThread`` that owns one scheduler per submitted batch. Cancellation is
cooperative; Qt signals carry progress and terminal state back to the GUI.
"""
from __future__ import annotations
