"""Outer Experiment tab: three columns, all visible at once.

The layout presents Task Settings, stacked Pending/Finished Tasks,
and Statistical Analysis. The panels share a single
:class:`TaskRegistry` (source of truth) and a single
:class:`BatchService` (engine bridge). Each panel subscribes to the
slice of the registry it cares about and ignores the rest.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from portal_studio.gui.common.composite_labels import generator_composite_label
from portal_studio.gui.experimental.batch_service import BatchService
from portal_studio.gui.experimental.completed_tasks import CompletedTasksPanel
from portal_studio.gui.experimental.pending_tasks import PendingTasksPanel
from portal_studio.gui.analysis.statistical_analysis import SaPanel
from portal_studio.gui.experimental.task_model import TaskRegistry
from portal_studio.gui.experimental.task_settings import TaskSettingsPanel


class ExperimentationTab(QWidget):
    """Top-level Experiment tab with task, execution, and analysis panels."""

    def __init__(self, generator_inventory,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._inventory = generator_inventory
        self.registry = TaskRegistry(self)
        self.batch_service = BatchService(self)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(8)

        self.task_settings_panel = TaskSettingsPanel(
            inventory=generator_inventory, registry=self.registry)
        self.task_settings_panel.setMinimumWidth(360)
        self.pending_panel = PendingTasksPanel(
            registry=self.registry, batch_service=self.batch_service)
        self.completed_panel = CompletedTasksPanel(
            registry=self.registry, batch_service=self.batch_service)
        self.middle_stack = QSplitter(Qt.Orientation.Vertical)
        self.middle_stack.setHandleWidth(8)
        self.middle_stack.addWidget(self.pending_panel)
        self.middle_stack.addWidget(self.completed_panel)
        self.middle_stack.setStretchFactor(0, 1)
        self.middle_stack.setStretchFactor(1, 1)
        self.middle_stack.setSizes([500, 500])
        self.middle_stack.setMinimumWidth(360)
        self.sa_panel = SaPanel(registry=self.registry)
        self.sa_panel.setMinimumWidth(360)

        self.splitter.addWidget(self.task_settings_panel)
        self.splitter.addWidget(self.middle_stack)
        self.splitter.addWidget(self.sa_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes([400, 800, 400])

        outer.addWidget(self.splitter)

        # Pending panel needs to read the *live* generated instance at
        # Run time — we wire that as a small callback so we don't have to
        # pass the inventory through every layer.
        self.pending_panel.install_generator_composite_provider(
            self._find_generator_composite)

    def shutdown(self) -> None:
        """Drain any in-flight batches before the main window closes."""
        self.batch_service.shutdown()

    def refresh_problem_panel(self) -> None:
        """Hook for the main window's tab-activated callback."""
        self.task_settings_panel.refresh_problem_combo()

    def _generator_composites(self) -> list[dict]:
        composites = [
            comp for comp in (getattr(self._inventory, "composites", []) or [])
            if isinstance(comp, dict)
        ]
        latest = getattr(self._inventory, "composite", None)
        if isinstance(latest, dict) and all(id(latest) != id(comp) for comp in composites):
            composites.append(latest)

        return composites

    def _find_generator_composite(self, label: str | None = None) -> dict | None:
        """Resolve one generated composite by display label, or latest if unset."""
        composites = self._generator_composites()
        latest = getattr(self._inventory, "composite", None)
        if label:
            for comp in composites:
                if generator_composite_label(comp) == label:
                    return comp
        return latest if isinstance(latest, dict) else (composites[-1] if composites else None)

    @property
    def tasks_subtab(self):
        """Compatibility facade for callers that use the task-subtab API."""
        return _LegacyTasksFacade(self)


class _LegacyTasksFacade:
    """Adapter exposing task-panel aliases."""

    def __init__(self, tab: ExperimentationTab) -> None:
        self._tab = tab

    @property
    def problem_panel(self):
        return self._tab.task_settings_panel

    @property
    def settings_panel(self):
        return self._tab.task_settings_panel
