"""PORTAL Studio entry point.

Run with: ``python -m portal_studio.gui.app``.
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from portal_studio import __version__
from portal_studio.gui.common.icons import make_app_icon
from portal_studio.gui.main_window import PortalStudioMainWindow


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(make_app_icon())
    win = PortalStudioMainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    run()
