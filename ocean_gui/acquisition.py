import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from PyQt5 import QtCore

from .processing import Processor
from .spectrometer import SpectrometerInterface


class RunMode(Enum):
    NUMBER = "number"
    TOTAL_TIME = "total_time"


@dataclass
class AcquisitionSettings:
    single_time_ms: float = 100.0
    down_time_ms: float = 0.0
    mode: RunMode = RunMode.NUMBER
    n_integrations: int = 10
    total_time_ms: float = 1000.0
    correct_dark_counts: bool = False
    correct_nonlinearity: bool = False

    def integrations_count(self) -> int:
        if self.mode is RunMode.NUMBER:
            return max(1, int(self.n_integrations))
        return max(1, int(self.total_time_ms // max(self.single_time_ms, 1e-6)))

    def validate(self) -> Optional[str]:
        if self.single_time_ms <= 0:
            return "Single integration time must be greater than 0 ms."
        if self.down_time_ms < 0:
            return "Down time cannot be negative."
        if self.mode is RunMode.NUMBER and self.n_integrations < 1:
            return "Number of integrations must be at least 1."
        if self.mode is RunMode.TOTAL_TIME and self.total_time_ms < self.single_time_ms:
            return "Total integration time must be >= single integration time."
        return None


class AcquisitionWorker(QtCore.QThread):
    """Collect processed scans. Averaging (with outlier filtering) is done by
    the GUI, so the filter can be changed while the run is in progress."""

    progress = QtCore.pyqtSignal(int, int, object, object)
    finished_ok = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spec: SpectrometerInterface, settings: AcquisitionSettings,
                 processor: Optional[Processor] = None):
        super().__init__()
        self._spec = spec
        self._settings = settings
        self._processor = processor or Processor()
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            settings = self._settings
            processor = self._processor
            total = settings.integrations_count()
            self._spec.set_integration_time_ms(settings.single_time_ms)
            self._spec.stabilize(correct_dark_counts=settings.correct_dark_counts,
                                 correct_nonlinearity=settings.correct_nonlinearity)

            wavelengths = self._spec.wavelengths()
            collected = np.empty((total, wavelengths.size), dtype=float)
            count = 0

            for i in range(total):
                if self._abort:
                    break
                raw = self._spec.intensities(
                    correct_dark_counts=settings.correct_dark_counts,
                    correct_nonlinearity=settings.correct_nonlinearity,
                )
                intensities = processor.apply(
                    raw, wavelengths, settings.single_time_ms / 1000.0)
                collected[count] = intensities
                count += 1
                self.progress.emit(count, total, wavelengths, intensities)

                if settings.down_time_ms > 0 and i < total - 1:
                    self._sleep_ms(settings.down_time_ms)

            if count == 0:
                self.failed.emit("Acquisition aborted before any data was collected.")
                return

            self.finished_ok.emit(wavelengths, collected[:count])
        except Exception as exc:
            self.failed.emit(str(exc))

    def _sleep_ms(self, ms: float) -> None:
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            if self._abort:
                return
            time.sleep(min(0.02, max(0.0, end - time.monotonic())))


class CaptureWorker(QtCore.QThread):
    """Capture a single averaged spectrum (used for dark/reference storage)."""

    captured = QtCore.pyqtSignal(object, object, float)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spec: SpectrometerInterface, single_time_ms: float,
                 n_average: int = 10, *, correct_dark_counts: bool = False,
                 correct_nonlinearity: bool = False):
        super().__init__()
        self._spec = spec
        self._single_time_ms = single_time_ms
        self._n_average = max(1, int(n_average))
        self._correct_dark_counts = correct_dark_counts
        self._correct_nonlinearity = correct_nonlinearity

    def run(self) -> None:
        try:
            actual_ms = self._spec.set_integration_time_ms(self._single_time_ms)
            self._spec.stabilize(correct_dark_counts=self._correct_dark_counts,
                                 correct_nonlinearity=self._correct_nonlinearity)
            wavelengths = self._spec.wavelengths()
            acc = np.zeros(wavelengths.size, dtype=float)
            for _ in range(self._n_average):
                acc += self._spec.intensities(
                    correct_dark_counts=self._correct_dark_counts,
                    correct_nonlinearity=self._correct_nonlinearity,
                )
            spectrum = acc / self._n_average
            self.captured.emit(wavelengths, spectrum, actual_ms)
        except Exception as exc:
            self.failed.emit(str(exc))
