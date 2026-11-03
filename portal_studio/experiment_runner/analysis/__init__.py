"""Pure-Python stats + plotting layer for the SA sub-tab.

No Qt imports. Modules:

- :mod:`stats`     Friedman + Holm-corrected Wilcoxon SR post-hoc, Wilcoxon SR.
- :mod:`plotting`  matplotlib trend curve + boxplot. Returns a
                   :class:`matplotlib.figure.Figure` so the GUI can
                   embed it in a ``FigureCanvasQTAgg``; an ``export_*``
                   helper persists the figure to PNG / PDF / SVG.
"""
from __future__ import annotations
