"""Shared Matplotlib sizing helpers for dense GUI plot panels."""

COMPACT_TITLE_SIZE = 10
COMPACT_LABEL_SIZE = 8
COMPACT_TICK_SIZE = 8
COMPACT_LEGEND_SIZE = 8


def style_compact_axes(ax) -> None:
    """Keep axis text compact so plots can use more of the available canvas."""
    ax.title.set_fontsize(COMPACT_TITLE_SIZE)
    ax.xaxis.label.set_size(COMPACT_LABEL_SIZE)
    ax.yaxis.label.set_size(COMPACT_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=COMPACT_TICK_SIZE)
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_size(COMPACT_LABEL_SIZE)
        ax.zaxis.set_tick_params(labelsize=COMPACT_TICK_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(COMPACT_LEGEND_SIZE)


def style_compact_colorbar(colorbar) -> None:
    colorbar.ax.tick_params(labelsize=COMPACT_TICK_SIZE)
