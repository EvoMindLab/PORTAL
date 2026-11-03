"""Matplotlib trend + boxplot renderers, plus PNG/PDF/SVG exporters.

PORTAL trend plots use function evaluations (FE) as the x-axis.

Two pure functions return a :class:`matplotlib.figure.Figure` that the
GUI embeds in a :class:`FigureCanvasQTAgg`. The ``export_*`` helpers
save the same figure to any subset of ``{png, pdf, svg}``.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.figure import Figure

from portal_studio.experiment_runner.analysis.stats import SelectionItem
from portal_studio.experiment_runner.normalisation import read_normalised
from portal_studio.experiment_runner.convergence import read_aligned_moments, standard_error


# Legend label modes.
LEGEND_LABEL_MODES = (
    "algorithm", "label", "algorithmLabel", "problemAlgorithm",
    "runsetId", "algorithmParam",
)

# Legend positions for trend plots.
LEGEND_POSITIONS = (
    "outside_right", "best", "upper_left", "upper_right",
    "lower_left", "lower_right",
)

EXPORT_FORMATS = ("png", "pdf", "svg")


@dataclass
class TrendSeries:
    """One line on a trend chart."""
    label: str                   # what the user sees in the legend
    fe: np.ndarray               # x-axis FE checkpoints
    mean: np.ndarray             # y-axis mean
    stderr: np.ndarray           # half-band around the mean


@dataclass
class BoxplotSeries:
    """One box on a boxplot."""
    label: str
    values: np.ndarray           # per-run final_error values


# ───────────────────────── data loaders ───────────────────────────────

def load_trend_series_from_selection(selection: Sequence[SelectionItem],
                                     *, metric: str = "best_so_far",
                                     legend_label_mode: str = "algorithm",
                                     algorithm_param_key: str = "",
                                     ) -> list[TrendSeries]:
    """Build trend series from a SelectionItem list.

    Reads ``trend.sqlite`` for each selection; missing or empty traces
    are skipped silently — the GUI surfaces a "no data" hint downstream.
    """
    out: list[TrendSeries] = []
    for item in selection:
        if metric in {'normalised_error', 'percentage_error'}:
            fe, mean, stderr = read_normalised(item.bundle_path, item.runset_uid, metric, trend=True)
            out.append(TrendSeries(_resolve_label(item, legend_label_mode, algorithm_param_key),
                                   fe, mean, stderr))
            continue
        if metric in {'final_error', 'best_so_far'}:
            summary = (Path(item.bundle_path) / 'summary.sqlite').resolve()
            if summary.is_file():
                cx = sqlite3.connect(summary.as_uri() + '?mode=ro', uri=True)
                try:
                    aligned = read_aligned_moments(cx, item.runset_uid)
                finally:
                    cx.close()
                if aligned is not None:
                    fe, mean, m2, count = aligned
                    if fe.size:
                        out.append(TrendSeries(
                            _resolve_label(item, legend_label_mode, algorithm_param_key),
                            fe, mean, standard_error(m2, count)))
                    continue
        trend_path = Path(item.bundle_path) / "trend.sqlite"
        if not trend_path.is_file():
            continue
        cx = sqlite3.connect(trend_path, timeout=5.0)
        try:
            cur = cx.execute(
                """SELECT fe, mean, stderr, count FROM trends
                   WHERE runset_uid = ? AND metric = ?
                   ORDER BY fe""",
                (item.runset_uid, metric),
            )
            fe, mean, s, count = [], [], [], []
            for r in cur:
                fe.append(r[0]); mean.append(r[1])
                s.append(r[2]); count.append(r[3])
        finally:
            cx.close()
        if not fe:
            continue
        s_arr = np.asarray(s, dtype=float)
        n_arr = np.asarray(count, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            stderr = np.where(
                n_arr > 1,
                np.sqrt(s_arr / (n_arr * (n_arr - 1))),
                0.0,
            )
        out.append(TrendSeries(
            label=_resolve_label(item, legend_label_mode, algorithm_param_key),
            fe=np.asarray(fe, dtype=int),
            mean=np.asarray(mean, dtype=float),
            stderr=stderr,
        ))
    return out


def load_boxplot_series_from_selection(selection: Sequence[SelectionItem],
                                       *, metric: str = "final_error",
                                       legend_label_mode: str = "algorithm",
                                       algorithm_param_key: str = "",
                                       ) -> list[BoxplotSeries]:
    """Build boxplot series — one box per selection item."""
    out: list[BoxplotSeries] = []
    for item in selection:
        if metric in {'normalised_error', 'percentage_error'}:
            values = read_normalised(item.bundle_path, item.runset_uid, metric)
            out.append(BoxplotSeries(_resolve_label(item, legend_label_mode, algorithm_param_key), values))
            continue
        cx = sqlite3.connect(
            Path(item.bundle_path) / "summary.sqlite", timeout=5.0)
        try:
            cur = cx.execute(
                f"""SELECT {_safe_scalar_column(metric)} FROM runs
                    WHERE runset_uid = ? ORDER BY run_index""",
                (item.runset_uid,),
            )
            values = np.asarray([r[0] for r in cur if r[0] is not None],
                                dtype=float)
        finally:
            cx.close()
        if values.size == 0:
            continue
        out.append(BoxplotSeries(
            label=_resolve_label(item, legend_label_mode, algorithm_param_key),
            values=values,
        ))
    return out


def _safe_scalar_column(metric: str) -> str:
    """Whitelist a scalar metric name → a column in the runs table."""
    allowed = {"final_error": "final_error", "fe_used": "fe_used"}
    return allowed.get(metric, "final_error")


# ───────────────────────── rendering ──────────────────────────────────

def render_trend(series: Sequence[TrendSeries], *,
                 metric: str = "best_so_far",
                 title: str | None = None,
                 legend_position: str = "outside_right",
                 x_min: float | None = None,
                 x_max: float | None = None,
                 y_scale: str = "auto",
                 acceptance_threshold: float | None = None,
                 ) -> Figure:
    """Render a trend chart and return its :class:`Figure`."""
    fig = Figure(figsize=(8, 5), facecolor="white")
    ax = fig.add_subplot(111)
    palette = _build_palette(len(series))

    all_positive = True
    for i, s in enumerate(series):
        mask = np.ones_like(s.fe, dtype=bool)
        if x_min is not None:
            mask &= (s.fe >= x_min)
        if x_max is not None:
            mask &= (s.fe <= x_max)
        fe = s.fe[mask]; mean = s.mean[mask]; stderr = s.stderr[mask]
        if fe.size == 0:
            continue
        if mean.size and np.nanmin(mean) <= 0:
            all_positive = False
        color = palette[i]
        ax.plot(fe, mean, label=s.label, color=color, linewidth=1.5)
        ax.fill_between(fe, mean - stderr, mean + stderr,
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)
    if y_scale == "log" or (y_scale == "auto" and all_positive and series):
        try:
            ax.set_yscale("log")
        except ValueError:
            ax.set_yscale("linear")
    elif y_scale == "linear":
        ax.set_yscale("linear")
    ax.grid(True, alpha=0.25)

    # Checkpoint lines + x-ticks: pick a "nice" interval that gives ~5-6
    # divisions regardless of the FE budget (5k → 1k, 50k → 10k, 300k → 50k).
    x_end = float(x_max) if x_max is not None else (
        max(float(s.fe[-1]) for s in series if s.fe.size) if series else 0.0)
    if x_end > 0:
        interval = _nice_fe_interval(x_end)
        from matplotlib.ticker import FuncFormatter, MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(interval))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: "0" if v == 0
                          else (f"{int(v) // 1000}k" if v >= 1000 else str(int(v)))))
        x_start = float(x_min) if x_min is not None else 0.0
        cp = interval
        while cp <= x_end:
            if cp > x_start:
                ax.axvline(cp, color="#b0b7c3", linestyle=":", linewidth=0.8, zorder=0)
            cp += interval

    if acceptance_threshold is not None:
        ax.axhline(acceptance_threshold, color="#2eb872", linestyle="--",
                   linewidth=1.0, label=f"acceptance ({acceptance_threshold:g})")

    _apply_legend_position(ax, fig, legend_position)
    fig.tight_layout()
    return fig


def _nice_fe_interval(x_max: float) -> float:
    """Return a round FE interval giving ~5-6 divisions for any budget."""
    raw = x_max / 6.0
    # Round to the nearest power-of-10 multiple: 1, 2, or 5 × 10^n
    import math
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10 ** exp
    for factor in (1, 2, 5, 10):
        candidate = factor * base
        if x_max / candidate <= 6:
            return float(candidate)
    return float(base * 10)


def render_boxplot(series: Sequence[BoxplotSeries], *,
                   metric: str = "final_error",
                   title: str | None = None,
                   ) -> Figure:
    fig = Figure(figsize=(8, 5), facecolor="white")
    ax = fig.add_subplot(111)
    if not series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#9aa0a6")
        ax.set_axis_off()
        return fig
    labels = [s.label for s in series]
    data = [s.values for s in series]
    ax.boxplot(data, tick_labels=labels, notch=True)
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if len(labels) > 5:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ───────────────────────── export helpers ─────────────────────────────

def export_figure(fig: Figure, output_dir: str | Path, *,
                  basename: str,
                  formats: Sequence[str] | None = None,
                  ) -> dict[str, Path]:
    """Save ``fig`` to ``output_dir/basename.<fmt>`` for each requested format.

    Returns ``{format: written_path}``. Invalid formats raise
    ``ValueError`` — silent skips would mask user typos.
    """
    fmts = list(formats) if formats else list(EXPORT_FORMATS)
    unknown = [f for f in fmts if f not in EXPORT_FORMATS]
    if unknown:
        raise ValueError(
            f"Unknown export format(s): {unknown}. Allowed: {EXPORT_FORMATS}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in fmts:
        path = out_dir / f"{_safe_name(basename)}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight")
        written[fmt] = path
    return written


# ───────────────────────── label resolver ─────────────────────────────

def _resolve_label(item: SelectionItem, mode: str,
                   algorithm_param_key: str = "") -> str:
    """Render the legend label for one selection item.

    - ``algorithm``        algorithm name only
    - ``label``            task label only
    - ``algorithmLabel``   ``algorithm: task_label``
    - ``problemAlgorithm`` ``PORTAL: algorithm``
    - ``runsetId``         the runset_uid
    - ``algorithmParam``   ``algorithm (param_key=value)``; falls back to
                           the algorithm name when the key is missing.
    """
    if mode == "algorithm":
        return item.algorithm
    if mode == "label":
        return item.task_label
    if mode == "algorithmLabel":
        return f"{item.algorithm}: {item.task_label}"
    if mode == "problemAlgorithm":
        return f"PORTAL: {item.algorithm}"
    if mode == "runsetId":
        return item.runset_uid
    if mode == "algorithmParam" and algorithm_param_key:
        # SelectionItem does not carry algorithm parameter values; use the
        # parameter key to label the series when a value is unavailable.
        return f"{item.algorithm} ({algorithm_param_key})"
    return item.algorithm


def _apply_legend_position(ax, fig, position: str) -> None:
    if position == "outside_right":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False)
        fig.subplots_adjust(right=0.78)
    elif position == "best":
        ax.legend(loc="best", frameon=False)
    else:
        loc = {
            "upper_left": "upper left", "upper_right": "upper right",
            "lower_left": "lower left", "lower_right": "lower right",
        }.get(position, "best")
        ax.legend(loc=loc, frameon=False)


def _build_palette(n: int) -> list:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    if n <= 10:
        return [cmap(i) for i in range(n)]
    # For larger sets, recycle the palette.
    return [cmap(i % 10) for i in range(n)]


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
