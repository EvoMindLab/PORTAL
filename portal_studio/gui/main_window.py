"""Top-level main window: a QTabWidget hosting the three studio tabs."""
from __future__ import annotations

from PySide6 import QtCore
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QTabWidget,
    QToolBar,
    QWidget,
)

from portal_studio.gui.common.icons import make_app_icon
from portal_studio import __version__
from portal_studio.gui.common.theme import ThemeManager
from portal_studio.session import SessionInventory


class PortalStudioMainWindow(QMainWindow):
    """Three-tab PORTAL Studio window backed by one shared SessionInventory."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PORTAL Studio {__version__}")
        self.inventory = SessionInventory()

        app = QApplication.instance()
        icon = make_app_icon()
        self.setWindowIcon(icon)
        if app is not None and app.windowIcon().isNull():
            app.setWindowIcon(icon)
        self.theme = ThemeManager(app)

        # Open at a size that fits the available screen (never larger), and
        # allow the window to shrink well below the content's natural size so
        # nothing is clipped off-screen on small displays.
        self.setMinimumSize(960, 600)
        screen = app.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(min(1600, avail.width() - 60), min(960, avail.height() - 60))
        else:
            self.resize(1280, 800)

        # Lazy imports to keep this module light if a tab grows heavy deps.
        from portal_studio.gui.educational.builder import BuilderTab
        from portal_studio.gui.educational.tab import EducationTab
        from portal_studio.gui.generator.tab import GeneratorTab

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.builder_tab = BuilderTab(self.inventory)
        self.education_tab = EducationTab(self.inventory)

        # The Education area has two internal views; navigation is handled by
        # the view buttons while the tab strip stays hidden.
        self.education_subtabs = QTabWidget()
        self.education_subtabs.setDocumentMode(True)
        self.education_subtabs.addTab(self.builder_tab, "2-D Authoring")
        self.education_subtabs.addTab(self.education_tab, "Benchmarking")
        self.education_subtabs.tabBar().hide()
        self.education_subtabs.setCurrentWidget(self.builder_tab)

        self._build_app_bar()

        self.generator_tab = GeneratorTab(self.inventory)
        self.tabs.addTab(self.generator_tab, "Generator")

        # Experiment consumes the Generator's live composite inventory.
        from portal_studio.gui.experimental.tab import ExperimentationTab
        self.experimentation_tab = ExperimentationTab(self.generator_tab.inventory)
        self.tabs.addTab(self.experimentation_tab, "Experiment")
        self.tabs.addTab(self.education_subtabs, "Education")

        self.setCentralWidget(self.tabs)

        # Cross-view navigation buttons drive the internal view switch.
        self.builder_tab.try_algorithm_btn.clicked.connect(
            lambda: self.education_subtabs.setCurrentWidget(self.education_tab))
        self.education_tab.back_btn.clicked.connect(
            lambda: self.education_subtabs.setCurrentWidget(self.builder_tab))

        # Refresh Benchmarking whenever it becomes visible.
        self.tabs.currentChanged.connect(self._maybe_refresh_education)
        self.education_subtabs.currentChanged.connect(self._maybe_refresh_education)
        self.tabs.currentChanged.connect(self._sync_benchmark_back_button)
        self.education_subtabs.currentChanged.connect(self._sync_benchmark_back_button)
        # Refresh the Experiment problem picker when the tab is shown.
        self.tabs.currentChanged.connect(self._maybe_refresh_experimentation)

        # Apply the initial (light) theme now that listeners exist.
        self.theme.set_mode("light")
        self.builder_tab.apply_palette(self.theme.palette)
        self._sync_benchmark_back_button()
        self._tour_overlay = None
        self._tour_started_this_session = False
        self._tour_auto_start_enabled = app.platformName().lower() != "offscreen"

    def closeEvent(self, event) -> None:                 # noqa: N802
        """Drain background work before destroying its GUI owners."""
        self.generator_tab.ela_panel.shutdown()
        self.education_tab.shutdown()
        if hasattr(self, "experimentation_tab"):
            self.experimentation_tab.shutdown()
        super().closeEvent(event)

    def showEvent(self, event) -> None:                  # noqa: N802
        super().showEvent(event)
        if self._tour_auto_start_enabled and not self._tour_started_this_session:
            self._tour_started_this_session = True
            QtCore.QTimer.singleShot(500, self.start_guided_tour)

    def guided_tour_steps(self) -> list:
        from portal_studio.gui.tour.guided_tour import TourStep

        def show_educational_authoring() -> None:
            self.tabs.setCurrentWidget(self.education_subtabs)
            self.education_subtabs.setCurrentWidget(self.builder_tab)

        def show_generator() -> None:
            self.tabs.setCurrentWidget(self.generator_tab)

        def show_experimental() -> None:
            self.tabs.setCurrentWidget(self.experimentation_tab)

        return [
            TourStep(
                self.generator_tab.add_block_btn,
                "Create high-dimensional blocks",
                "Generator starts with blocks. Add a block first, then attach "
                "components inside it to build a high-dimensional instance.",
                show_generator,
            ),
            TourStep(
                self.generator_tab._param_pane,
                "Author component parameters",
                "Block and component parameters appear here. Select any block or "
                "component in Instance Layers to edit values, apply changes, reset "
                "parameters, and control shape, rotation, neutralization, and "
                "transforms. If nothing is listed yet, create one first.",
                show_generator,
            ),
            TourStep(
                self.generator_tab.instance_actions_widget,
                "Prepare the generator instance",
                "Apply component edits, select Prepare Instance for ELA, then "
                "use Save Instance JSON to preserve the prepared problem.",
                show_generator,
            ),
            TourStep(
                self.generator_tab.ela_panel,
                "Explore landscape features",
                "The ELA panel reads the prepared Generator instance and helps you "
                "inspect feature groups for the landscape you created.",
                show_generator,
            ),
            TourStep(
                self.experimentation_tab.task_settings_panel,
                "Configure an experiment",
                "Choose a problem instance, algorithm, run count, FE budget, and "
                "seed settings, then choose the realisation type before adding "
                "a task.",
                show_experimental,
            ),
            TourStep(
                self.experimentation_tab.pending_panel,
                "Queue and run tasks",
                "Tick the pending tasks, choose the pool and workers, then "
                "select Run Selected Tasks.",
                show_experimental,
            ),
            TourStep(
                self.experimentation_tab.completed_panel,
                "Review completed runs",
                "Finished tasks appear here with summary results and export controls.",
                show_experimental,
            ),
            TourStep(
                self.experimentation_tab.sa_panel,
                "Compare algorithms",
                "Use statistical analysis after runs complete to compare algorithms "
                "and inspect trend or box plots.",
                show_experimental,
            ),
            TourStep(
                self.builder_tab.add_block_btn,
                "Create an instance",
                "In 2-D Authoring, add an instance first so PORTAL Studio "
                "has a place to hold the components you build.",
                show_educational_authoring,
            ),
            TourStep(
                self.builder_tab.parameter_panel,
                "Prepare a component",
                "This middle panel is where component parameters appear. Add or "
                "select an instance first, then the component controls become active.",
                show_educational_authoring,
            ),
            TourStep(
                self.builder_tab.visualization_tabs,
                "Inspect the landscape",
                "The plots here show the current landscape, components, and built "
                "instances. Use this panel to check what you made.",
                show_educational_authoring,
            ),
            TourStep(
                self.builder_tab.try_algorithm_btn,
                "Benchmark what you built",
                "This button opens Benchmarking for the 2-D Authoring instance. It "
                "becomes active after at least one instance contains at least one "
                "component.",
                show_educational_authoring,
            ),
        ]

    def start_guided_tour(self) -> None:
        from portal_studio.gui.tour.guided_tour import GuidedTourOverlay

        if self._tour_overlay is not None:
            self._tour_overlay.close()
        overlay = GuidedTourOverlay(self, self.guided_tour_steps())
        overlay.finished.connect(lambda: setattr(self, "_tour_overlay", None))
        self._tour_overlay = overlay
        overlay.start()

    def _maybe_refresh_education(self, *_args) -> None:
        if (self.tabs.currentWidget() is self.education_subtabs
                and self.education_subtabs.currentWidget() is self.education_tab):
            self.education_tab.refresh_sources()

    def _maybe_refresh_experimentation(self, *_args) -> None:
        if (hasattr(self, "experimentation_tab")
                and self.tabs.currentWidget() is self.experimentation_tab):
            self.experimentation_tab.refresh_problem_panel()

    def _sync_benchmark_back_button(self, *_args) -> None:
        if not hasattr(self, "benchmark_back_action"):
            return
        visible = (
            self.tabs.currentWidget() is self.education_subtabs
            and self.education_subtabs.currentWidget() is self.education_tab
        )
        self.benchmark_back_action.setVisible(visible)

    def _build_app_bar(self) -> None:
        from PySide6.QtWidgets import QStyle

        toolbar = QToolBar("PORTAL Studio")
        toolbar.setObjectName("appToolbar")
        toolbar.setMovable(False)
        title = QLabel("PORTAL Studio")
        title.setObjectName("panelTitle")
        title.setStyleSheet("padding-right: 18px;")
        toolbar.addWidget(title)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.setStyleSheet("background: transparent;")  # no visible rectangle
        toolbar.addWidget(spacer)

        style = self.style()
        self.help_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion), "Help", self
        )
        self.help_action.setToolTip("Start the PORTAL Studio guided tour")
        self.help_action.setStatusTip("Start the PORTAL Studio guided tour")
        self.help_action.triggered.connect(self._on_help_clicked)
        self.benchmark_back_action = toolbar.addWidget(self.education_tab.back_btn)
        self.benchmark_back_action.setVisible(False)
        toolbar.addAction(self.help_action)

        self.addToolBar(toolbar)

    def _on_load_clicked(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getOpenFileName(
            self, "Load PORTAL instance JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            import json as _json

            from portal_core.compose_blocks import unified_payload_to_composite
            from portal_core.instance_format import decode_instance_file
            from portal_core.load_instance import load_instance_payload
            from portal_studio.validation import validate_instance_for_gui

            with open(path, "r", encoding="utf-8") as _fh:
                payload = decode_instance_file(_json.load(_fh))
            # Apply the same complete schema validation used by script-based
            # evaluation before adapting the instance to the editor runtime.
            validate_instance_for_gui(load_instance_payload(payload))
            # Rebuild the transient in-memory composition used by the 2-D
            # editor. The file itself remains a unified PORTAL instance.
            composite = unified_payload_to_composite(payload)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            QMessageBox.critical(
                self, "Load failed", f"Could not read this as a PORTAL instance:\n{exc}"
            )
            return

        # A composite's top-level d is the total (2 per block); PORTAL Studio is
        # a 2-D editor, so every block must itself be 2-D.
        blocks = composite.get("blocks", [])
        block_dims = sorted({int(b.get("d", 0)) for b in blocks})
        if not blocks or block_dims != [2]:
            QMessageBox.critical(
                self,
                "Unsupported dimension",
                f"This composite has blocks of dimension {block_dims or 'unknown'}. "
                "This editor view only supports 2-D (d = 2) blocks.",
            )
            return

        import os

        from portal_studio.session import CompositeRef

        self.inventory.add_composite(
            CompositeRef(label=os.path.basename(path), composite=composite, saved_path=path)
        )
        self.builder_tab._refresh_project_tree()
        QMessageBox.information(
            self, "Loaded", f"Loaded PORTAL instance from {os.path.basename(path)}."
        )

    def _on_help_clicked(self) -> None:
        self.start_guided_tour()
