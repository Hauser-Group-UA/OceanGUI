import re
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from . import plotting
from .processing import DEFAULT_OUTLIER_SIGMA, OUTLIER_CHOICES
from .units import XUNIT_LABELS, XUnit, format_x, format_y


class DisplaySettings(QtCore.QObject):
    """Display choices shared by every mode: x-axis unit and outlier threshold."""

    xunit_changed = QtCore.pyqtSignal(object)
    outlier_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.xunit = XUnit.WAVELENGTH
        self.outlier_sigma = DEFAULT_OUTLIER_SIGMA

    def set_xunit(self, unit: XUnit) -> None:
        if unit is not self.xunit:
            self.xunit = unit
            self.xunit_changed.emit(unit)

    def set_outlier_sigma(self, value) -> None:
        value = float(value or 0.0)
        if value != self.outlier_sigma:
            self.outlier_sigma = value
            self.outlier_changed.emit(value)


_TIME_UNITS = (("ms", 1.0), ("s", 1000.0), ("min", 60000.0), ("h", 3600000.0))


def _readable_unit(ms: float):
    """Largest unit showing ``ms`` as a number >= 1 with at most 3 decimals."""
    for name, mult in reversed(_TIME_UNITS):
        value = ms / mult
        if value >= 1 and abs(round(value, 3) - value) < 1e-9:
            return name, mult
    return _TIME_UNITS[0]


def format_duration(ms: float) -> str:
    """A readable duration, e.g. '250 ms', '65.535 s', '30 min', '1.5 h'."""
    name, mult = _readable_unit(ms)
    return f"{ms / mult:g} {name}"


class TimeField(QtWidgets.QWidget):
    """A value spin-box plus a ms/s/min/h unit selector. Time is stored in ms."""

    changed = QtCore.pyqtSignal()
    _UNITS = _TIME_UNITS

    def __init__(self, default_ms: float = 1000.0, default_unit: str = "s",
                 parent=None) -> None:
        super().__init__(parent)
        self._mult = dict(self._UNITS)
        self._ms = float(default_ms)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(0.0, 1.0e9)
        self.unit = QtWidgets.QComboBox()
        for name, _ in self._UNITS:
            self.unit.addItem(name)
        self.unit.setCurrentText(default_unit)
        lay.addWidget(self.spin, 1)
        lay.addWidget(self.unit, 0)
        self._display_from_ms()
        self.spin.valueChanged.connect(self._on_spin)
        self.unit.currentIndexChanged.connect(self._on_unit_changed)

    def _on_spin(self, *_) -> None:
        self._ms = self.spin.value() * self._mult[self.unit.currentText()]
        self.changed.emit()

    def _on_unit_changed(self) -> None:
        self._display_from_ms()
        self.changed.emit()

    def _display_from_ms(self) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(self._ms / self._mult[self.unit.currentText()])
        self.spin.blockSignals(False)

    def milliseconds(self) -> float:
        return self._ms

    def set_milliseconds(self, ms: float, readable_unit: bool = False) -> None:
        """Set the time; ``readable_unit`` also switches to a unit that shows it exactly."""
        self._ms = float(ms)
        if readable_unit:
            self.unit.blockSignals(True)
            self.unit.setCurrentText(_readable_unit(ms)[0])
            self.unit.blockSignals(False)
        self._display_from_ms()


_BREAK_AFTER = re.compile(r"([\\/_.\-])")


def breakable(text: str) -> str:
    """Let a word-wrapped label break long paths and file names after separators.

    Qt wraps only at spaces and a few symbols - never after a backslash - so a
    Windows path like C:\\Users\\...\\saved_data was one unbreakable "word" and
    its label demanded the full path width. A zero-width space after each
    separator gives the wrapper somewhere to break.
    """
    return _BREAK_AFTER.sub("\\1\u200b", text)


class ElidedLabel(QtWidgets.QLabel):
    """One-line label that shortens long text with '…' instead of widening the layout."""

    def __init__(self, text: str = "", elide=QtCore.Qt.ElideRight, parent=None) -> None:
        super().__init__(parent)
        self._full = ""
        self._elide = elide
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text or ""
        self._refresh()
        self.updateGeometry()

    def fullText(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:
        margins = self.contentsMargins()
        width = (self.fontMetrics().horizontalAdvance(self._full) + 4
                 + margins.left() + margins.right() + 2 * self.margin())
        return QtCore.QSize(width, super().sizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self._refresh()

    def _refresh(self) -> None:
        width = max(0, self.contentsRect().width() - 2 * self.margin())
        super().setText(self.fontMetrics().elidedText(self._full, self._elide, width))


class PanelScrollArea(QtWidgets.QScrollArea):
    """A vertically scrolling settings page that is always wide enough for its content.

    A plain QScrollArea reports a small minimum width, so when its content
    needed more room (long paths, large fonts from display scaling) the
    content was laid out wider than the panel and - with the horizontal
    scroll bar off - its right-hand side was silently cut off. Reporting the
    content's minimum width (plus room for the vertical scroll bar) lets the
    surrounding layout make the panel wide enough instead.
    """

    def __init__(self, inner: QtWidgets.QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setWidget(inner)

    def _content_width(self) -> int:
        inner = self.widget()
        if inner is None:
            return 0
        bar = max(self.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent, None, self),
                  self.verticalScrollBar().sizeHint().width())
        return inner.minimumSizeHint().width() + bar + 2 * self.frameWidth()

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        return QtCore.QSize(max(hint.width(), self._content_width()), hint.height())

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        return QtCore.QSize(max(hint.width(), self._content_width()), hint.height())

    def eventFilter(self, obj, event) -> bool:
        if obj is self.widget() and event.type() == QtCore.QEvent.LayoutRequest:
            self.updateGeometry()  # the content's minimum width may have changed
        return super().eventFilter(obj, event)


def scroll_page(groups) -> PanelScrollArea:
    """Stack group boxes in a vertically scrolling page."""
    inner = QtWidgets.QWidget()
    vb = QtWidgets.QVBoxLayout(inner)
    vb.setContentsMargins(6, 6, 6, 6)
    for group in groups:
        vb.addWidget(group)
    vb.addStretch(1)
    return PanelScrollArea(inner)


class SidePanel(QtWidgets.QWidget):
    """Left-hand column: 350 px wide, or wider whenever its content needs it.

    Replaces a hard-coded ``setFixedWidth(350)`` that cropped the content when
    it needed more room (larger fonts from display scaling, long paths).
    """

    DEFAULT_WIDTH = 350

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(max(self.DEFAULT_WIDTH, self.minimumSizeHint().width()),
                            super().sizeHint().height())


class OutlierCombo(QtWidgets.QComboBox):
    """Outlier-threshold drop-down; every instance stays in sync."""

    def __init__(self, settings: DisplaySettings, parent=None) -> None:
        super().__init__(parent)
        for value in OUTLIER_CHOICES:
            self.addItem(f"{value:g}σ" if value else "Off", float(value))
        self.setToolTip(
            "Before averaging, drop any scan's value that lies more than this many "
            "σ from the median of all scans at the same wavelength (σ estimated "
            "robustly from the median absolute deviation, adjusted for the number "
            "of scans). On pure noise 3σ removes about 0.27% of values.")
        self._show(settings.outlier_sigma)
        settings.outlier_changed.connect(self._show)
        self.currentIndexChanged.connect(
            lambda i: settings.set_outlier_sigma(self.itemData(i)))

    def _show(self, value: float) -> None:
        self.blockSignals(True)
        self.setCurrentIndex(max(0, self.findData(float(value))))
        self.blockSignals(False)


def describe_outliers(stats, n_scans: int) -> str:
    """Short note for plot titles, e.g. '12 outlier values removed (3σ)'."""
    if not stats.threshold_sigma or n_scans < 3:
        return ""
    n = stats.n_removed
    return f"{n} outlier value{'s' if n != 1 else ''} removed ({stats.threshold_sigma:g}σ)"


def outlier_group(settings: DisplaySettings) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox("Outlier filtering")
    form = QtWidgets.QFormLayout(box)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
    form.addRow("Threshold:", OutlierCombo(settings))
    hint = QtWidgets.QLabel("Judged separately at each wavelength, against "
                            "the median of all scans. Applies live.")
    hint.setWordWrap(True)
    hint.setStyleSheet("color: #666666; font-size: 11px;")
    form.addRow(hint)
    return box


class UncertaintyBox(QtWidgets.QGroupBox):
    """1std / 2std error bars (one at a time) and bands for an average plot."""

    changed = QtCore.pyqtSignal()

    def __init__(self, title: str = "Uncertainty (average plot)", parent=None) -> None:
        super().__init__(title, parent)
        grid = QtWidgets.QGridLayout(self)
        self.cb_bars1 = QtWidgets.QCheckBox("1σ bars")
        self.cb_bars2 = QtWidgets.QCheckBox("2σ bars")
        self.cb_band1 = QtWidgets.QCheckBox("1σ band")
        self.cb_band2 = QtWidgets.QCheckBox("2σ band")
        self.cb_bars1.toggled.connect(lambda on: on and self.cb_bars2.setChecked(False))
        self.cb_bars2.toggled.connect(lambda on: on and self.cb_bars1.setChecked(False))
        for w in (self.cb_bars1, self.cb_bars2, self.cb_band1, self.cb_band2):
            w.toggled.connect(lambda *_: self.changed.emit())
        grid.addWidget(self.cb_bars1, 0, 0)
        grid.addWidget(self.cb_bars2, 0, 1)
        grid.addWidget(self.cb_band1, 1, 0)
        grid.addWidget(self.cb_band2, 1, 1)
        grid.setRowStretch(2, 1)

    def flags(self) -> dict:
        return dict(bars_1sigma=self.cb_bars1.isChecked(),
                    bars_2sigma=self.cb_bars2.isChecked(),
                    band_1sigma=self.cb_band1.isChecked(),
                    band_2sigma=self.cb_band2.isChecked())


MENU_MARK = " ▾"

class _Canvas(FigureCanvas):
    """A plot canvas that doesn't render while hidden and catches up when shown.

    Pages for the other modes keep receiving updates (an acquisition can run
    while you're in View mode); rendering them unseen wastes time, and before
    their first layout they are too small for matplotlib's tight layout.
    """

    def __init__(self, figure) -> None:
        super().__init__(figure)
        self._missed_draw = False

    def draw(self) -> None:
        if not self.isVisible():
            self._missed_draw = True
            return
        super().draw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._missed_draw:
            self._missed_draw = False
            self.draw_idle()


class SpectrumPlot(QtWidgets.QWidget):
    """A titled spectrum plot you can read values from and switch units on.

    - click the x-axis label: a menu switches wavelength / energy (and Raman
      shift when the data allows) - reported through ``unit_selected``;
    - hover: the value under the cursor is shown below the plot;
    - left-click: marks the nearest data point with its (x, y); right-click
      clears the mark. The mark stays at that x through redraws, so it
      follows live data;
    - click the legend (when ``legend_editable``): ``legend_clicked``;
    - double-click inside the axes: ``double_clicked``.

    Values are read from lines whose gid starts with ``plotting.PROBE_GID``.
    Owners draw onto ``ax``, call :meth:`set_units`, then :meth:`finish_draw`.
    """

    unit_selected = QtCore.pyqtSignal(object)
    legend_clicked = QtCore.pyqtSignal()
    double_clicked = QtCore.pyqtSignal()

    HINT = "Click to read a value  ·  click the x-axis label to change units"

    def __init__(self, title: Optional[str] = "", header_widget=None, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.title = ElidedLabel(title or "")
        if title is not None:
            header = QtWidgets.QWidget()
            header.setFixedHeight(34)
            row = QtWidgets.QHBoxLayout(header)
            row.setContentsMargins(0, 0, 0, 0)
            font = QtGui.QFont()
            font.setBold(True)
            font.setPointSize(12)
            self.title.setFont(font)
            self.title.setAlignment(QtCore.Qt.AlignCenter)
            row.addWidget(self.title, 1)
            if header_widget is not None:
                row.addWidget(header_widget, 0)
            lay.addWidget(header, 0)

        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = _Canvas(self.figure)
        self.canvas.setMinimumSize(220, 180)
        lay.addWidget(self.canvas, 1)

        self.readout = ElidedLabel(self.HINT)
        self.readout.setAlignment(QtCore.Qt.AlignCenter)
        self.readout.setStyleSheet("color: #555555; font-size: 11px;")
        lay.addWidget(self.readout, 0)

        self.legend_editable = False
        self._units = (XUnit.WAVELENGTH, XUnit.ENERGY)
        self._unit = XUnit.WAVELENGTH
        self._pin = None
        self._pin_artists = []
        self._hovering = False
        self._hand = False
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)

    def set_title(self, text: str) -> None:
        self.title.setText(text)
        self.title.setToolTip(text)

    def set_units(self, available, current: XUnit) -> None:
        """Units offered by the x-axis-label menu, and the one being shown."""
        if current is not self._unit:
            self._pin = None
        self._units = tuple(available)
        self._unit = current

    def finish_draw(self) -> None:
        """Call after drawing onto ``ax``: restores the marked point and repaints."""
        label = self.ax.xaxis.label
        text = label.get_text()
        if text and not text.endswith(MENU_MARK):
            label.set_text(text + MENU_MARK)
        self._remove_pin_artists()
        self._place_pin()
        self.canvas.draw_idle()
        if not self._hovering:
            self._show_idle_text()

    def clear_pin(self) -> None:
        if self._pin is None:
            return
        self._pin = None
        self._pin_text = ""
        self._remove_pin_artists()
        self.canvas.draw_idle()
        self._show_idle_text()

    def _targets(self):
        return [line for line in self.ax.get_lines()
                if str(line.get_gid() or "").startswith(plotting.PROBE_GID)]

    @staticmethod
    def _xy(line):
        return (np.asarray(line.get_xdata(), dtype=float),
                np.asarray(line.get_ydata(), dtype=float))

    @staticmethod
    def _nearest_index(x: np.ndarray, x0: float) -> Optional[int]:
        if x.size == 0:
            return None
        with np.errstate(invalid="ignore"):
            dist = np.abs(x - x0)
        if not np.isfinite(dist).any():
            return None
        return int(np.nanargmin(dist))

    def _nearest(self, event):
        """(line, index) of the data point nearest the cursor, or None."""
        if event.xdata is None or event.ydata is None:
            return None
        best = None
        for line in self._targets():
            x, y = self._xy(line)
            i = self._nearest_index(x, event.xdata)
            if i is None or not np.isfinite(y[i]):
                continue
            _, py = self.ax.transData.transform((x[i], y[i]))
            dist = abs(py - event.y)
            if best is None or dist < best[0]:
                best = (dist, line, i)
        return None if best is None else best[1:]

    def _describe(self, line, x: float, y: float, with_name: bool) -> str:
        text = f"({format_x(x, self._unit)}, {format_y(y, self.ax.get_ylabel())})"
        name = line.get_label()
        if with_name and name and not name.startswith("_"):
            text = f"{name}: {text}"
        return text

    def _show_idle_text(self) -> None:
        if self._pin is not None and self._pin_text:
            self.readout.setText(f"Marked {self._pin_text}  ·  right-click to clear")
        else:
            self.readout.setText(self.HINT)

    def _remove_pin_artists(self) -> None:
        for artist in self._pin_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
        self._pin_artists = []

    def _place_pin(self) -> None:
        self._pin_text = ""
        if self._pin is None:
            return
        gid, x0 = self._pin
        targets = self._targets()
        line = next((t for t in targets if t.get_gid() == gid), None)
        if line is None:
            return
        x, y = self._xy(line)
        i = self._nearest_index(x, x0)
        if i is None or not np.isfinite(y[i]):
            return
        xi, yi = float(x[i]), float(y[i])
        self._pin_text = self._describe(line, xi, yi, with_name=len(targets) > 1)

        self.ax.get_xlim(), self.ax.get_ylim()
        fx, fy = self.ax.transAxes.inverted().transform(
            self.ax.transData.transform((xi, yi)))
        dx, ha = (-10, "right") if fx > 0.5 else (10, "left")
        dy, va = (-14, "top") if fy > 0.8 else (12, "bottom")
        label = f"({format_x(xi, self._unit)}, {format_y(yi, self.ax.get_ylabel())})"
        name = line.get_label()
        if len(targets) > 1 and name and not name.startswith("_"):
            label = f"{name}\n{label}"
        marker, = self.ax.plot([xi], [yi], linestyle="none", marker="o", markersize=7,
                               markerfacecolor="none", markeredgecolor="#111111",
                               markeredgewidth=1.5, zorder=1000)
        note = self.ax.annotate(
            label, xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=10, zorder=1001, annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.95))
        self._keep_inside(note)
        self._pin_artists = [marker, note]

    def _keep_inside(self, note) -> None:
        """Wrap and nudge the value box so it stays inside a narrow plot."""
        try:
            renderer = self.canvas.get_renderer()
            fig = self.figure.bbox
            pad = 8.0
            if note.get_window_extent(renderer).width > fig.width - 2 * pad:
                note.set_text(note.get_text().replace(", ", ",\n"))
            box = note.get_window_extent(renderer)
            shift_x = min(0.0, fig.x1 - pad - box.x1) or max(0.0, fig.x0 + pad - box.x0)
            shift_y = min(0.0, fig.y1 - pad - box.y1) or max(0.0, fig.y0 + pad - box.y0)
        except Exception:
            return
        if shift_x or shift_y:
            to_points = 72.0 / self.figure.dpi
            ox, oy = note.xyann
            note.xyann = (ox + shift_x * to_points, oy + shift_y * to_points)

    def _over_xlabel(self, event) -> bool:
        try:
            return bool(self.ax.xaxis.label.contains(event)[0])
        except Exception:
            return False

    def _over_legend(self, event) -> bool:
        legend = self.ax.get_legend()
        if legend is None or not legend.get_visible():
            return False
        try:
            return bool(legend.contains(event)[0])
        except Exception:
            return False

    def _set_hand(self, on: bool) -> None:
        if on != self._hand:
            self._hand = on
            if on:
                self.canvas.setCursor(QtCore.Qt.PointingHandCursor)
            else:
                self.canvas.unsetCursor()

    def _on_motion(self, event) -> None:
        if self._over_xlabel(event):
            self._set_hand(True)
            names = " / ".join(XUNIT_LABELS[u].split(" (")[0].lower() for u in self._units)
            self.readout.setText(f"Click to switch the x-axis: {names}")
            return
        if self.legend_editable and self._over_legend(event):
            self._set_hand(True)
            self.readout.setText("Click the legend to edit its labels")
            return
        self._set_hand(False)
        hit = self._nearest(event) if event.inaxes is self.ax else None
        self._hovering = hit is not None
        if hit is None:
            self._show_idle_text()
            return
        line, i = hit
        x, y = self._xy(line)
        self.readout.setText(self._describe(line, x[i], y[i],
                                            with_name=len(self._targets()) > 1))

    def _on_leave(self, _event) -> None:
        self._hovering = False
        self._set_hand(False)
        self._show_idle_text()

    def _on_press(self, event) -> None:
        if event.dblclick:
            if event.inaxes is self.ax:
                self.double_clicked.emit()
            return
        if self._over_xlabel(event):
            pos = QtGui.QCursor.pos()
            QtCore.QTimer.singleShot(0, lambda: self._show_unit_menu(pos))
            return
        if self.legend_editable and self._over_legend(event):
            QtCore.QTimer.singleShot(0, self.legend_clicked.emit)
            return
        if event.inaxes is not self.ax:
            return
        if event.button == 3:
            self.clear_pin()
            return
        if event.button != 1:
            return
        hit = self._nearest(event)
        if hit is None:
            return
        line, i = hit
        self._pin = (line.get_gid(), float(self._xy(line)[0][i]))
        self._remove_pin_artists()
        self._place_pin()
        self.canvas.draw_idle()

    def _show_unit_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        group = QtWidgets.QActionGroup(menu)
        for unit in self._units:
            action = menu.addAction(XUNIT_LABELS[unit])
            action.setCheckable(True)
            action.setChecked(unit is self._unit)
            action.setData(unit.value)
            group.addAction(action)
        chosen = menu.exec_(pos)
        if chosen is not None:
            unit = XUnit(chosen.data())
            if unit is not self._unit:
                self.unit_selected.emit(unit)


class FullScreenPlot(QtWidgets.QDialog):
    """A full-screen view of one plot. Esc or the button exits full screen."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = SpectrumPlot(title=None)
        lay.addWidget(self.plot, 1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn = QtWidgets.QPushButton("Exit full screen (Esc)")
        btn.clicked.connect(self.close)
        row.addWidget(btn)
        lay.addLayout(row)

    def render(self, draw_fn, units, unit: XUnit) -> None:
        draw_fn(self.plot.ax)
        self.plot.set_units(units, unit)
        self.plot.finish_draw()
