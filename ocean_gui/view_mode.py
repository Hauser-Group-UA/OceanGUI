from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from . import plotting, storage
from .processing import MODE_LABELS, MeasurementMode, scan_statistics
from .units import available_units, spectral_axis
from .widgets import (DisplaySettings, ElidedLabel, FullScreenPlot, SidePanel,
                      SpectrumPlot, UncertaintyBox, breakable, describe_outliers,
                      outlier_group, scroll_page)


class ViewPage(QtWidgets.QWidget):
    """Open a saved run and browse it scan by scan."""

    status_message = QtCore.pyqtSignal(str)

    def __init__(self, display: DisplaySettings, save_dir: Callable[[], Path],
                 parent=None) -> None:
        super().__init__(parent)
        self.display = display
        self._save_dir = save_dir
        self._run: Optional[storage.RunData] = None
        self._stats = None
        self._index = 0
        self._follow_latest = True
        self._latest_saved: Optional[Path] = None
        self._needs_latest = True
        self._fs_plot: Optional[FullScreenPlot] = None

        layout = QtWidgets.QHBoxLayout(self)
        panel = SidePanel()
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(4, 4, 4, 4)
        self.uncertainty = UncertaintyBox()
        self.uncertainty.changed.connect(self._draw_average)
        pl.addWidget(scroll_page([self._group_file(), self._group_scan(),
                                  outlier_group(display), self.uncertainty]), 1)
        layout.addWidget(panel, 0)
        layout.addWidget(self._build_plots(), 1)

        display.xunit_changed.connect(lambda *_: self._redraw())
        display.outlier_changed.connect(self._on_outlier_changed)
        self._show_empty("Open a saved run (…_data.csv)")

    def _group_file(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Run file")
        v = QtWidgets.QVBoxLayout(box)
        self.file_name = ElidedLabel("No file loaded")
        font = self.file_name.font()
        font.setBold(True)
        self.file_name.setFont(font)
        self.file_info = QtWidgets.QLabel("")
        self.file_info.setWordWrap(True)
        self.file_info.setStyleSheet("font-size: 11px; color: #555555;")
        row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open…")
        open_btn.setToolTip("Open a specific saved run (its *_data.csv file)")
        open_btn.clicked.connect(self._choose_file)
        latest_btn = QtWidgets.QPushButton("Latest run")
        latest_btn.setToolTip("Show the most recent run, and keep following new runs")
        latest_btn.clicked.connect(lambda: self.load_latest(quiet=False))
        row.addWidget(open_btn)
        row.addWidget(latest_btn)
        v.addWidget(self.file_name)
        v.addWidget(self.file_info)
        v.addLayout(row)
        return box

    def _group_scan(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Scan")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        style = self.style()
        self.first_btn = self._nav_button(style.standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward),
                                          "First scan", lambda: self._set_index(0))
        self.prev_btn = self._nav_button(style.standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward),
                                         "Previous scan", lambda: self._set_index(self._index - 1),
                                         repeat=True)
        self.scan_spin = QtWidgets.QSpinBox()
        self.scan_spin.setRange(1, 1)
        self.scan_spin.setKeyboardTracking(False)
        self.scan_spin.setToolTip("Type a scan number and press Enter to jump to it")
        self.scan_spin.valueChanged.connect(lambda v: self._set_index(v - 1))
        self.of_label = QtWidgets.QLabel("of 0")
        self.next_btn = self._nav_button(style.standardIcon(QtWidgets.QStyle.SP_MediaSeekForward),
                                         "Next scan", lambda: self._set_index(self._index + 1),
                                         repeat=True)
        self.last_btn = self._nav_button(style.standardIcon(QtWidgets.QStyle.SP_MediaSkipForward),
                                         "Last scan", lambda: self._set_index(self._n_scans() - 1))
        row.addWidget(self.first_btn)
        row.addWidget(self.prev_btn)
        row.addWidget(self.scan_spin, 1)
        row.addWidget(self.of_label)
        row.addWidget(self.next_btn)
        row.addWidget(self.last_btn)
        self.scan_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scan_slider.setRange(1, 1)
        self.scan_slider.valueChanged.connect(lambda v: self._set_index(v - 1))
        v.addLayout(row)
        v.addWidget(self.scan_slider)
        self._set_nav_enabled()
        return box

    @staticmethod
    def _nav_button(icon, tip: str, slot, repeat: bool = False) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setIcon(icon)
        btn.setToolTip(tip)
        btn.setAutoRepeat(repeat)
        btn.clicked.connect(slot)
        return btn

    def _build_plots(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(panel)
        self.plot_scan = SpectrumPlot("Scan")
        self.fullscreen_btn = QtWidgets.QToolButton()
        self.fullscreen_btn.setText("⤢")
        self.fullscreen_btn.setToolTip("Maximise the average plot to full screen "
                                       "(double-click the plot; Esc exits)")
        self.fullscreen_btn.clicked.connect(self._open_fullscreen)
        self.plot_avg = SpectrumPlot("Average", header_widget=self.fullscreen_btn)
        self.plot_avg.double_clicked.connect(self._open_fullscreen)
        for plot in (self.plot_scan, self.plot_avg):
            plot.unit_selected.connect(self.display.set_xunit)
        h.addWidget(self.plot_scan, 1)
        h.addWidget(self.plot_avg, 1)
        return panel

    def activate(self) -> None:
        """Called when View mode is shown: pick up the newest run if following."""
        if self._needs_latest and self._follow_latest:
            self._needs_latest = False
            self.load_latest(quiet=True)

    def notify_run_saved(self, path: Path) -> None:
        """A run was (re)saved in Capture mode."""
        self._latest_saved = Path(path)
        if not self._follow_latest:
            return
        if self.isVisible():
            self.load_file(path, follow=True)
        else:
            self._needs_latest = True

    def load_latest(self, quiet: bool = False) -> None:
        folder = self._save_dir()
        path = storage.find_latest_run(folder, extra=[self._latest_saved])
        if path is None:
            self._follow_latest = True
            if not quiet:
                QtWidgets.QMessageBox.information(
                    self, "No runs found",
                    f"No saved runs (*_data.csv) were found in:\n{folder}")
            if self._run is None:
                self._show_empty("No saved runs yet - acquire one in Capture mode\n"
                                 "or Open… a *_data.csv file")
            return
        self.load_file(path, follow=True)

    def _choose_file(self) -> None:
        start = str(self._run.path.parent if self._run is not None else self._save_dir())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open saved run", start,
            "Run data (*_data.csv);;CSV files (*.csv);;All files (*)")
        if path:
            self.load_file(path, follow=False)

    def load_file(self, path, follow: bool = False) -> bool:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            run = storage.load_run(path)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Could not open file",
                                          f"{path}\n\nThis does not look like a saved run: {exc}")
            return False
        QtWidgets.QApplication.restoreOverrideCursor()

        self._run = run
        self._stats = None
        self._follow_latest = follow
        self.file_name.setText(run.display_name)
        self.file_name.setToolTip(str(run.path))
        self.file_info.setText(breakable(self._describe(run)))
        self.file_info.setToolTip(str(run.path))
        n = run.n_scans
        for w in (self.scan_spin, self.scan_slider):
            w.blockSignals(True)
            w.setRange(1, n)
            w.blockSignals(False)
        self.of_label.setText(f"of {n}")
        self._index = -1
        self._set_index(n - 1, draw=False)
        self._redraw()
        self.status_message.emit(f"Viewing {run.path.name} ({n} scan{'s' if n != 1 else ''})")
        return True

    @staticmethod
    def _describe(run: storage.RunData) -> str:
        parts = [f"{run.n_scans} scan{'s' if run.n_scans != 1 else ''}"]
        if run.integration_ms is not None:
            parts.append(f"{run.integration_ms:g} ms each")
        if run.mode is not None:
            parts.append(MODE_LABELS[run.mode])
        if run.acquired is not None:
            parts.append(run.acquired.strftime("%Y-%m-%d %H:%M:%S"))
        return " · ".join(parts) + f"\n{run.path.parent}"

    def _n_scans(self) -> int:
        return self._run.n_scans if self._run is not None else 0

    def _set_index(self, index: int, draw: bool = True) -> None:
        if self._run is None:
            return
        index = int(np.clip(index, 0, self._n_scans() - 1))
        for w in (self.scan_spin, self.scan_slider):
            w.blockSignals(True)
            w.setValue(index + 1)
            w.blockSignals(False)
        changed = index != self._index
        self._index = index
        self._set_nav_enabled()
        if draw and changed:
            self._draw_scan()

    def _set_nav_enabled(self) -> None:
        n = self._n_scans()
        self.first_btn.setEnabled(self._index > 0)
        self.prev_btn.setEnabled(self._index > 0)
        self.next_btn.setEnabled(0 <= self._index < n - 1)
        self.last_btn.setEnabled(0 <= self._index < n - 1)
        for w in (self.scan_spin, self.scan_slider):
            w.setEnabled(n > 1)

    def _axis(self):
        run = self._run
        return spectral_axis(run.wavelengths, self.display.xunit, run.mode,
                             run.excitation_nm, run.ylabel)

    def _units(self):
        return available_units(self._run.mode, self._run.excitation_nm)

    def _y_from_zero(self) -> bool:
        return self._run.mode is MeasurementMode.SCOPE

    def _show_empty(self, message: str) -> None:
        axis = spectral_axis(np.array([400.0, 800.0]), self.display.xunit)
        for plot in (self.plot_scan, self.plot_avg):
            plotting.draw_placeholder(plot.ax, message, xlabel=axis.xlabel, ylabel=axis.ylabel)
            plot.set_units(available_units(None, None), axis.unit)
            plot.finish_draw()

    def _redraw(self) -> None:
        if self._run is None:
            self._show_empty("Open a saved run (…_data.csv)")
            return
        self._draw_scan()
        self._draw_average()

    def _draw_scan(self) -> None:
        if self._run is None:
            return
        try:
            axis = self._axis()
            plotting.draw_current(self.plot_scan.ax, axis.x,
                                  axis.apply(self._run.scans[self._index]),
                                  ylabel=axis.ylabel, xlabel=axis.xlabel,
                                  y_from_zero=self._y_from_zero(),
                                  gid=plotting.PROBE_GID + "scan")
            self.plot_scan.set_title(f"Scan {self._index + 1} of {self._n_scans()}")
            self.plot_scan.set_units(self._units(), axis.unit)
            self.plot_scan.finish_draw()
        except Exception as exc:
            self.status_message.emit(f"Plot update skipped: {exc}")

    def _stats_now(self):
        if self._stats is None:
            self._stats = scan_statistics(self._run.scans, self.display.outlier_sigma)
        return self._stats

    def _draw_average_into(self, ax, axis) -> None:
        stats = self._stats_now()
        plotting.draw_average(ax, axis.x, axis.apply(stats.average), axis.apply(stats.std),
                              **self.uncertainty.flags(), ylabel=axis.ylabel,
                              xlabel=axis.xlabel, y_from_zero=self._y_from_zero(),
                              gid=plotting.PROBE_GID + "average")

    def _draw_average(self) -> None:
        if self._run is None:
            return
        try:
            axis = self._axis()
            self._draw_average_into(self.plot_avg.ax, axis)
            n = self._n_scans()
            title = f"Average of {n} scan{'s' if n != 1 else ''}"
            note = describe_outliers(self._stats_now(), n)
            self.plot_avg.set_title(f"{title}  ·  {note}" if note else title)
            self.plot_avg.set_units(self._units(), axis.unit)
            self.plot_avg.finish_draw()
            if self._fs_plot is not None:
                self._fs_plot.render(lambda ax: self._draw_average_into(ax, axis),
                                     self._units(), axis.unit)
        except Exception as exc:
            self.status_message.emit(f"Plot update skipped: {exc}")

    def _on_outlier_changed(self, _sigma) -> None:
        self._stats = None
        self._draw_average()

    def _open_fullscreen(self) -> None:
        if self._run is None:
            return
        if self._fs_plot is None:
            self._fs_plot = FullScreenPlot("Average", self)
            self._fs_plot.finished.connect(self._on_fullscreen_closed)
            self._fs_plot.plot.unit_selected.connect(self.display.set_xunit)
        axis = self._axis()
        self._fs_plot.render(lambda ax: self._draw_average_into(ax, axis),
                             self._units(), axis.unit)
        self._fs_plot.showFullScreen()
        self._fs_plot.raise_()

    def _on_fullscreen_closed(self, *_) -> None:
        self._fs_plot = None
