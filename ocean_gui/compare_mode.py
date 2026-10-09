from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from . import plotting, storage
from .processing import MeasurementMode, ScanStats, scan_statistics
from .units import EXAMPLE_WAVELENGTHS, XUnit, available_units, spectral_axis
from .widgets import (DisplaySettings, ElidedLabel, PanelScrollArea, SidePanel,
                      SpectrumPlot, UncertaintyBox, outlier_group)

_QT_PEN = {"-": QtCore.Qt.SolidLine, "--": QtCore.Qt.DashLine,
           ":": QtCore.Qt.DotLine, "-.": QtCore.Qt.DashDotLine}
_MIXED_YLABEL = "Value - mixed quantities"
_SAVE_FILTERS = {"PNG image (*.png)": ".png", "PDF document (*.pdf)": ".pdf",
                 "SVG image (*.svg)": ".svg"}


@dataclass
class CompareItem:
    """One run in the comparison."""

    run: storage.RunData
    label: str
    style: int
    included: bool = True
    _cache: Dict[float, ScanStats] = field(default_factory=dict)

    def stats(self, threshold: float) -> ScanStats:
        if threshold not in self._cache:
            self._cache = {threshold: scan_statistics(self.run.scans, threshold)}
        return self._cache[threshold]


def _swatch(style: int, ratio: float = 1.0) -> QtGui.QPixmap:
    """A short sample of a run's line (colour + dash) for lists."""
    color, dash = plotting.compare_style(style)
    pixmap = QtGui.QPixmap(int(32 * ratio), int(14 * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidthF(2.2)
    pen.setStyle(_QT_PEN.get(dash, QtCore.Qt.SolidLine))
    pen.setCapStyle(QtCore.Qt.FlatCap)
    painter.setPen(pen)
    painter.drawLine(QtCore.QPointF(1, 7), QtCore.QPointF(31, 7))
    painter.end()
    return pixmap


class LegendLabelsDialog(QtWidgets.QDialog):
    """Edit the legend text of each run (scrolls when there are many)."""

    def __init__(self, items: List[CompareItem], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Legend labels")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Text shown in the legend for each run:"))

        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner)
        bold = QtGui.QFont()
        bold.setBold(True)
        for col, text in ((1, "File"), (2, "Legend label")):
            head = QtWidgets.QLabel(text)
            head.setFont(bold)
            grid.addWidget(head, 0, col)
        ratio = self.devicePixelRatioF()
        self._edits = []
        for row, item in enumerate(items, start=1):
            swatch = QtWidgets.QLabel()
            swatch.setPixmap(_swatch(item.style, ratio))
            name = ElidedLabel(item.run.display_name)
            name.setToolTip(str(item.run.path))
            edit = QtWidgets.QLineEdit(item.label)
            edit.setPlaceholderText(item.run.run_name)
            grid.addWidget(swatch, row, 0)
            grid.addWidget(name, row, 1)
            grid.addWidget(edit, row, 2)
            self._edits.append((item, edit))
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(len(items) + 1, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Reset)
        buttons.button(QtWidgets.QDialogButtonBox.Reset).setText("Use run names")
        buttons.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self._reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(660, min(560, 140 + 36 * len(items)))

    def _reset(self) -> None:
        for item, edit in self._edits:
            edit.setText(item.run.run_name)

    def labels(self) -> List[str]:
        return [edit.text().strip() or item.run.run_name for item, edit in self._edits]


class ComparePage(QtWidgets.QWidget):
    """Overlay the run averages of saved files, redrawn on demand."""

    status_message = QtCore.pyqtSignal(str)

    def __init__(self, display: DisplaySettings, save_dir: Callable[[], Path],
                 parent=None) -> None:
        super().__init__(parent)
        self.display = display
        self._save_dir = save_dir
        self._items: List[CompareItem] = []
        self._last_dir: Optional[Path] = None

        layout = QtWidgets.QHBoxLayout(self)
        panel = SidePanel()
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(4, 4, 4, 4)
        pl.setSpacing(6)
        pl.addWidget(self._group_files(), 1)
        pl.addWidget(outlier_group(display), 0)
        self.uncertainty = UncertaintyBox("Uncertainty")
        self.uncertainty.changed.connect(self._mark_stale)
        pl.addWidget(self.uncertainty, 0)
        pl.addWidget(self._group_legend(), 0)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        pl.addWidget(sep)
        pl.addWidget(self._build_actions(), 0)
        layout.addWidget(panel, 0)
        layout.addWidget(self._build_plot(), 1)

        display.xunit_changed.connect(self._mark_stale)
        display.outlier_changed.connect(self._mark_stale)
        self._rebuild_rows()
        self._redraw()

    def _group_files(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Runs (top of the list is drawn on top)")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add files…")
        add_btn.setToolTip("Add saved runs (*_data.csv); each is shown as its run average")
        add_btn.clicked.connect(self._add_files)
        clear_btn = QtWidgets.QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        row.addWidget(add_btn)
        row.addWidget(clear_btn)
        v.addLayout(row)

        inner = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(inner)
        self.rows_layout.setContentsMargins(2, 2, 2, 2)
        self.rows_layout.setSpacing(2)
        self.list_area = PanelScrollArea(inner)
        self.list_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.list_area.setMinimumHeight(110)
        v.addWidget(self.list_area, 1)
        return box

    def _group_legend(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Legend")
        h = QtWidgets.QHBoxLayout(box)
        self.legend_cb = QtWidgets.QCheckBox("Show legend")
        self.legend_cb.setChecked(True)
        self.legend_cb.toggled.connect(self._mark_stale)
        labels_btn = QtWidgets.QPushButton("Edit labels…")
        labels_btn.setToolTip("Rename the legend entries (or click the legend on the plot)")
        labels_btn.clicked.connect(self._edit_labels)
        h.addWidget(self.legend_cb)
        h.addStretch(1)
        h.addWidget(labels_btn)
        return box

    def _build_actions(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)
        self.redraw_btn = QtWidgets.QPushButton("Redraw plot")
        self.redraw_btn.setToolTip("Draw the plot with the current runs and settings")
        self.redraw_btn.clicked.connect(self._redraw)
        save_btn = QtWidgets.QPushButton("Save figure…")
        save_btn.setToolTip("Save the comparison at paper quality (600 DPI)")
        save_btn.clicked.connect(self._save_figure)
        export_btn = QtWidgets.QPushButton("Export data…")
        export_btn.setToolTip("CSV with the x-axis and one column per ticked run "
                              "(averages with the current outlier filter)")
        export_btn.clicked.connect(self._export_data)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(save_btn)
        row.addWidget(export_btn)
        lay.addWidget(self.redraw_btn)
        lay.addLayout(row)
        return bar

    def _build_plot(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        self.stale_banner = QtWidgets.QLabel(
            "Changes not shown yet - press Redraw plot to update.")
        self.stale_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.stale_banner.setStyleSheet(
            "background: #fff4ce; border: 1px solid #e3c766; border-radius: 3px;"
            " padding: 3px; color: #4a3c00;")
        self.stale_banner.hide()
        self.plot = SpectrumPlot("Compare run averages")
        self.plot.legend_editable = True
        self.plot.legend_clicked.connect(self._edit_labels)
        self.plot.unit_selected.connect(self.display.set_xunit)
        v.addWidget(self.stale_banner, 0)
        v.addWidget(self.plot, 1)
        return panel

    def _rebuild_rows(self) -> None:
        while self.rows_layout.count():
            widget = self.rows_layout.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        if not self._items:
            empty = QtWidgets.QLabel("No runs yet - use Add files… to pick\nsaved *_data.csv files.")
            empty.setStyleSheet("color: #666666;")
            self.rows_layout.addWidget(empty)
        ratio = self.devicePixelRatioF()
        last = len(self._items) - 1
        for i, item in enumerate(self._items):
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(2, 1, 2, 1)
            h.setSpacing(3)
            swatch = QtWidgets.QLabel()
            swatch.setPixmap(_swatch(item.style, ratio))
            name = ElidedLabel(item.run.display_name)
            name.setToolTip(str(item.run.path))
            name.setEnabled(item.included)
            include = QtWidgets.QCheckBox()
            include.setChecked(item.included)
            include.setToolTip("Show this run in the plot (unticked runs stay in the list)")
            include.toggled.connect(
                lambda on, it=item, label=name: self._set_included(it, label, on))
            h.addWidget(swatch, 0)
            h.addWidget(name, 1)
            h.addWidget(include, 0)
            for text, tip, slot, enabled in (
                    ("▲", "Move up (drawn above the runs below)",
                     lambda checked=False, k=i: self._move(k, -1), i > 0),
                    ("▼", "Move down", lambda checked=False, k=i: self._move(k, 1), i < last),
                    ("✕", "Remove from the comparison",
                     lambda checked=False, k=i: self._remove(k), True)):
                btn = QtWidgets.QToolButton()
                btn.setText(text)
                btn.setToolTip(tip)
                btn.setEnabled(enabled)
                btn.clicked.connect(slot)
                h.addWidget(btn, 0)
            self.rows_layout.addWidget(row)
        self.rows_layout.addStretch(1)

    def _included(self) -> List[CompareItem]:
        return [item for item in self._items if item.included]

    def _set_included(self, item: CompareItem, label: QtWidgets.QLabel, on: bool) -> None:
        item.included = on
        label.setEnabled(on)
        self._mark_stale()

    def _free_style(self) -> int:
        used = {item.style for item in self._items}
        return next(s for s in range(len(used) + 1) if s not in used)

    def _default_label(self, run: storage.RunData) -> str:
        taken = {item.label for item in self._items}
        return run.display_name if run.run_name in taken else run.run_name

    def _add_files(self) -> None:
        start = str(self._last_dir or self._save_dir())
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add runs to compare", start,
            "Run data (*_data.csv);;CSV files (*.csv);;All files (*)")
        if not paths:
            return
        self._last_dir = Path(paths[-1]).parent
        listed = {item.run.path.resolve() for item in self._items}
        added, skipped = 0, []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for path in paths:
                resolved = Path(path).resolve()
                if resolved in listed:
                    continue
                try:
                    run = storage.load_run(path)
                except Exception as exc:
                    skipped.append(f"{Path(path).name}: {exc}")
                    continue
                self._items.append(CompareItem(run, self._default_label(run), self._free_style()))
                listed.add(resolved)
                added += 1
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if skipped:
            QtWidgets.QMessageBox.warning(self, "Some files were skipped",
                                          "These could not be read:\n\n" + "\n".join(skipped))
        if added:
            self._rebuild_rows()
            self._mark_stale()
            self.status_message.emit(
                f"Added {added} run{'s' if added != 1 else ''} - press Redraw plot to show.")

    def _move(self, index: int, step: int) -> None:
        j = index + step
        if 0 <= index < len(self._items) and 0 <= j < len(self._items):
            self._items[index], self._items[j] = self._items[j], self._items[index]
            self._rebuild_rows()
            self._mark_stale()

    def _remove(self, index: int) -> None:
        if 0 <= index < len(self._items):
            del self._items[index]
            self._rebuild_rows()
            self._mark_stale()

    def _clear_all(self) -> None:
        if not self._items:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Clear all runs?", "Remove every run from the comparison?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self._items.clear()
            self._rebuild_rows()
            self._mark_stale()

    def _edit_labels(self) -> None:
        if not self._items:
            self.status_message.emit("Add runs first - there are no legend entries yet.")
            return
        dialog = LegendLabelsDialog(self._items, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        labels = dialog.labels()
        if labels != [item.label for item in self._items]:
            for item, text in zip(self._items, labels):
                item.label = text
            self._mark_stale()

    def _mark_stale(self, *_) -> None:
        if not self._items:
            self._redraw()
            return
        self.stale_banner.show()
        self.redraw_btn.setText("Redraw plot  (changes pending)")
        self.redraw_btn.setStyleSheet(
            "QPushButton { background: #2a78d6; color: white; font-weight: bold;"
            " border-radius: 4px; padding: 6px; }"
            "QPushButton:pressed { background: #1c5cab; }")

    def _set_fresh(self) -> None:
        self.stale_banner.hide()
        self.redraw_btn.setText("Redraw plot")
        self.redraw_btn.setStyleSheet("")

    def _units(self) -> List[XUnit]:
        items = self._included()
        if not items:
            return available_units(None, None)
        sets = [available_units(item.run.mode, item.run.excitation_nm) for item in items]
        return [u for u in sets[0] if all(u in s for s in sets[1:])]

    def _unit(self, units: List[XUnit]) -> XUnit:
        return self.display.xunit if self.display.xunit in units else XUnit.WAVELENGTH

    def _render(self, ax, unit: XUnit) -> None:
        threshold = self.display.outlier_sigma
        items = self._included()
        series, ylabels, xlabel = [], set(), ""
        for item in items:
            stats = item.stats(threshold)
            axis = spectral_axis(item.run.wavelengths, unit, item.run.mode,
                                 item.run.excitation_nm, item.run.ylabel)
            color, dash = plotting.compare_style(item.style)
            series.append(dict(x=axis.x, y=axis.apply(stats.average),
                               std=axis.apply(stats.std), color=color, linestyle=dash,
                               label=item.label, gid=f"{plotting.PROBE_GID}{id(item)}"))
            ylabels.add(axis.ylabel)
            xlabel = axis.xlabel
        plotting.draw_compare(
            ax, series, **self.uncertainty.flags(), legend=self.legend_cb.isChecked(),
            ylabel=ylabels.pop() if len(ylabels) == 1 else _MIXED_YLABEL, xlabel=xlabel,
            y_from_zero=all(i.run.mode is MeasurementMode.SCOPE for i in items))

    def _redraw(self) -> None:
        units = self._units()
        unit = self._unit(units)
        self.plot.hold_input()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            shown, total = len(self._included()), len(self._items)
            if shown:
                self._render(self.plot.ax, unit)
                count = f"{shown} run{'s' if shown != 1 else ''}"
                if shown < total:
                    count = f"{shown} of {total} runs"
                self.plot.set_title(f"Average of each run  ({count})")
            else:
                axis = spectral_axis(EXAMPLE_WAVELENGTHS, unit)
                message = ("Tick a run in the list to show it" if total
                           else "Add saved runs to compare their averages")
                plotting.draw_placeholder(self.plot.ax, message, x=axis.x,
                                          xlabel=axis.xlabel, ylabel=axis.ylabel)
                self.plot.set_title("Compare run averages")
            self.plot.set_units(units, unit)
            self.plot.finish_draw()
            self._set_fresh()
        except Exception as exc:
            self.status_message.emit(f"Redraw failed: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _save_figure(self) -> None:
        if not self._included():
            self.status_message.emit("Tick at least one run - there is nothing to save yet.")
            return
        start = Path(self._last_dir or self._save_dir()) / "comparison.png"
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save comparison figure", str(start), ";;".join(_SAVE_FILTERS))
        if not path:
            return
        if Path(path).suffix.lower() not in _SAVE_FILTERS.values():
            path += _SAVE_FILTERS.get(chosen, ".png")
        units = self._units()
        fig = Figure(figsize=plotting.PAPER_FIGSIZE, tight_layout=True)
        try:
            self._render(fig.add_subplot(111), self._unit(units))
            fig.savefig(path, dpi=plotting.PAPER_DPI)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._redraw()
        self.status_message.emit(f"Saved {path}")

    def _export_data(self) -> None:
        items = self._included()
        if not items:
            self.status_message.emit("Tick at least one run - there is nothing to export yet.")
            return
        start = Path(self._last_dir or self._save_dir()) / "comparison.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export data", str(start), "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        unit = self._unit(self._units())
        threshold = self.display.outlier_sigma
        ref = items[0]
        ref_axis = spectral_axis(ref.run.wavelengths, unit, ref.run.mode,
                                 ref.run.excitation_nm, ref.run.ylabel)
        order = np.argsort(ref_axis.x)
        x = ref_axis.x[order]
        headers, columns, interpolated = [ref_axis.xlabel], [x], 0
        for item in items:
            axis = spectral_axis(item.run.wavelengths, unit, item.run.mode,
                                 item.run.excitation_nm, item.run.ylabel)
            y = axis.apply(item.stats(threshold).average)
            if np.array_equal(item.run.wavelengths, ref.run.wavelengths):
                column = y[order]
            else:
                src = np.argsort(axis.x)
                column = np.interp(x, axis.x[src], y[src], left=np.nan, right=np.nan)
                interpolated += 1
            headers.append(item.label)
            columns.append(column)
        try:
            storage.save_columns_csv(path, headers, columns)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        note = (f"; {interpolated} interpolated onto {ref.label}'s wavelength grid"
                if interpolated else "")
        self.status_message.emit(f"Exported {len(items)} run{'s' if len(items) != 1 else ''} "
                                 f"to {path}{note}")
