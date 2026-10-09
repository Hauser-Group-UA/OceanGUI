import numpy as np

LINE_RED = "#cc1f1f"
LINE_BLUE = "#1f3fcc"
GREY = "#9a9a9a"
BAND_1 = "#cc1f1f"
BAND_2 = "#f0a0a0"

PAPER_FIGSIZE = (6.0, 4.5)
PAPER_DPI = 600
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11

XLABEL = "Wavelength (nm)"
YLABEL = "Intensity (counts)"

PROBE_GID = "probe:"

COMPARE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")  # blue, orange, aqua, violet
COMPARE_LINESTYLES = ("-", "--", ":", "-.")
COMPARE_LINEWIDTH = 1.4


def compare_style(index: int):
    """(colour, linestyle) for the ``index``-th style slot: colours first, then dashes."""
    n = len(COMPARE_COLORS)
    return (COMPARE_COLORS[index % n],
            COMPARE_LINESTYLES[(index // n) % len(COMPARE_LINESTYLES)])


def style_axes(ax, wavelengths=None, *, y_from_zero: bool = False,
               ylabel: str = YLABEL, xlabel: str = XLABEL) -> None:
    """Apply the shared paper-quality styling to an axes.

    - axis labels with units, no title;
    - inward major+minor ticks on all four sides;
    - x-limits tightened to the data so ticks reach the plot edges.
    """
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)

    if wavelengths is not None and np.size(wavelengths) > 1:
        ax.set_xlim(float(np.nanmin(wavelengths)), float(np.nanmax(wavelengths)))
    ax.margins(y=0.02)
    if y_from_zero:
        ax.set_ylim(bottom=0.0)

    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=5, width=1.0, labelsize=TICK_FONTSIZE)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def draw_placeholder(ax, message: str = "Awaiting acquisition\n(example axes)", *,
                     x=None, xlabel: str = XLABEL, ylabel: str = YLABEL) -> None:
    """Draw example dummy axes shown before any real data exists."""
    t = np.linspace(0, 10, 200 if x is None else np.size(x))
    x = t if x is None else np.asarray(x, dtype=float)
    y = np.sin(t) * np.exp(-0.1 * t)
    ax.clear()
    ax.plot(x, y, color=GREY, linestyle="--", linewidth=1.0)
    ax.text(0.5, 0.5, message,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#666666",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    style_axes(ax, x, xlabel=xlabel, ylabel=ylabel)


def draw_current(ax, wavelengths, intensities, *, ylabel: str = YLABEL,
                 xlabel: str = XLABEL, y_from_zero: bool = True,
                 gid: str = None) -> None:
    """Draw the most recent single integration."""
    ax.clear()
    ax.plot(wavelengths, intensities, color=LINE_BLUE, linewidth=1.0, gid=gid)
    style_axes(ax, wavelengths, y_from_zero=y_from_zero, ylabel=ylabel, xlabel=xlabel)


def draw_average(
    ax,
    wavelengths,
    average,
    std=None,
    *,
    bars_1sigma: bool = False,
    bars_2sigma: bool = False,
    band_1sigma: bool = False,
    band_2sigma: bool = False,
    color: str = LINE_RED,
    ylabel: str = YLABEL,
    xlabel: str = XLABEL,
    y_from_zero: bool = False,
    gid: str = None,
) -> None:
    """Draw the average spectrum with optional uncertainty bars/bands."""
    ax.clear()
    if std is None:
        std = np.zeros_like(average)

    if band_2sigma:
        ax.fill_between(wavelengths, average - 2 * std, average + 2 * std,
                        color=BAND_2, alpha=0.5, label=r"2$\sigma$ band")
    if band_1sigma:
        ax.fill_between(wavelengths, average - std, average + std,
                        color=BAND_1, alpha=0.25, label=r"1$\sigma$ band")

    if bars_1sigma or bars_2sigma:
        idx = _bar_indices(np.size(wavelengths))
        k = 2 if bars_2sigma else 1
        label = r"2$\sigma$ bars" if bars_2sigma else r"1$\sigma$ bars"
        ax.errorbar(wavelengths[idx], average[idx], yerr=k * std[idx],
                    fmt="none", ecolor="#444444", elinewidth=0.8,
                    capsize=2, alpha=0.7, label=label)

    ax.plot(wavelengths, average, color=color, linewidth=1.2, label="average", gid=gid)
    style_axes(ax, wavelengths, y_from_zero=y_from_zero, ylabel=ylabel, xlabel=xlabel)
    if bars_1sigma or bars_2sigma or band_1sigma or band_2sigma:
        ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9)


def draw_overlay(ax, wavelengths, all_intensities, average, *, ylabel: str = YLABEL,
                 xlabel: str = XLABEL, y_from_zero: bool = True) -> None:
    """Average in red with each individual integration in grey behind it."""
    ax.clear()
    for row in all_intensities:
        ax.plot(wavelengths, row, color=GREY, linewidth=0.6, alpha=0.5)
    ax.plot(wavelengths, average, color=LINE_RED, linewidth=1.4)
    style_axes(ax, wavelengths, y_from_zero=y_from_zero, ylabel=ylabel, xlabel=xlabel)


def draw_compare(ax, series, *, bars_1sigma: bool = False, bars_2sigma: bool = False,
                 band_1sigma: bool = False, band_2sigma: bool = False,
                 legend: bool = True, ylabel: str = YLABEL, xlabel: str = XLABEL,
                 y_from_zero: bool = False) -> None:
    """Overlay several averages; the first series is drawn on top.

    Each series is a dict with keys x, y, std, color, linestyle, label, gid.
    """
    ax.clear()
    n = len(series)
    lines = []
    for i, s in enumerate(series):
        x, y, std = s["x"], s["y"], s["std"]
        z = 2 + 3 * (n - i)
        if band_2sigma:
            ax.fill_between(x, y - 2 * std, y + 2 * std, color=s["color"],
                            alpha=0.10, linewidth=0, zorder=z)
        if band_1sigma:
            ax.fill_between(x, y - std, y + std, color=s["color"],
                            alpha=0.22, linewidth=0, zorder=z)
        if bars_1sigma or bars_2sigma:
            idx = _bar_indices(np.size(x))
            k = 2 if bars_2sigma else 1
            ax.errorbar(x[idx], y[idx], yerr=k * std[idx], fmt="none",
                        ecolor=s["color"], elinewidth=0.8, capsize=2, alpha=0.7,
                        zorder=z + 1)
        line, = ax.plot(x, y, color=s["color"], linestyle=s["linestyle"],
                        linewidth=COMPARE_LINEWIDTH, label=s["label"],
                        gid=s.get("gid"), zorder=z + 2)
        lines.append(line)
    all_x = np.concatenate([np.asarray(s["x"], dtype=float) for s in series])
    style_axes(ax, all_x, y_from_zero=y_from_zero, ylabel=ylabel, xlabel=xlabel)
    if legend and lines:
        ax.legend(handles=lines, loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.9)


def _bar_indices(n: int) -> np.ndarray:
    """About 60 evenly spaced points, so error bars stay legible."""
    return np.arange(0, n, max(1, n // 60))
