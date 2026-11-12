import time
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from seabreeze.spectrometers import Spectrometer, list_devices

    _SEABREEZE_AVAILABLE = True
except Exception:
    Spectrometer = None
    list_devices = None
    _SEABREEZE_AVAILABLE = False


SIMULATED_SERIAL = "SIM-0001"


@dataclass
class DeviceInfo:
    """A spectrometer the user can select and connect to."""

    label: str
    serial: str
    simulated: bool


class SpectrometerError(RuntimeError):
    """Raised when the spectrometer cannot be opened or read."""


class CorrectionUnsupported(SpectrometerError):
    """The device reads fine but cannot apply a requested correction."""


class SimulatedSpectrometer:
    model = "Simulated"
    serial_number = "SIM-0001"
    supports_electric_dark = True
    supports_nonlinearity = False

    def __init__(self, n_pixels: int = 2048) -> None:
        self._n = n_pixels
        self._wavelengths = np.linspace(340.0, 1024.0, n_pixels)
        self._integration_us = 100_000
        self._peaks = [(486.0, 4000, 6.0), (589.0, 9000, 4.0),
                       (656.0, 6000, 8.0), (820.0, 2500, 12.0)]

    def integration_time_micros(self, micros: int) -> None:
        self._integration_us = int(micros)

    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def intensities(self, correct_dark_counts: bool = False,
                    correct_nonlinearity: bool = False) -> np.ndarray:
        if correct_nonlinearity:
            raise SpectrometerError(
                "Nonlinearity correction is not supported by this device.")
        time.sleep(self._integration_us / 1_000_000.0)
        scale = self._integration_us / 100_000.0
        baseline = 200.0 * scale
        base = baseline + np.zeros(self._n)
        for center, amp, width in self._peaks:
            base += amp * scale * np.exp(-0.5 * ((self._wavelengths - center) / width) ** 2)
        noise = np.random.normal(0.0, np.sqrt(np.abs(base)) + 15.0 * scale)
        out = np.clip(base + noise, 0.0, None)
        if correct_dark_counts:
            out = out - baseline
        return out

    def close(self) -> None:
        pass


class SpectrometerInterface:
    def __init__(self, device=None, simulated: bool = False) -> None:
        self._device = device
        self.simulated = simulated

    @classmethod
    def open_first(cls, allow_simulation: bool = True) -> "SpectrometerInterface":
        if _SEABREEZE_AVAILABLE:
            try:
                devices = list_devices()
                if devices:
                    spec = Spectrometer.from_first_available()
                    return cls(spec, simulated=False)
            except Exception as exc:
                if not allow_simulation:
                    raise SpectrometerError(f"Could not open spectrometer: {exc}") from exc

        if not allow_simulation:
            raise SpectrometerError(
                "No spectrometer found and simulation disabled. "
                "Connect an Ocean device and install seabreeze."
            )
        return cls(SimulatedSpectrometer(), simulated=True)

    @staticmethod
    def list_available(include_simulated: bool = True) -> List[DeviceInfo]:
        """Enumerate connected spectrometers (plus a simulation entry)."""
        infos: List[DeviceInfo] = []
        if _SEABREEZE_AVAILABLE:
            try:
                for d in list_devices() or []:
                    infos.append(DeviceInfo(
                        label=f"{d.model} ({d.serial_number})",
                        serial=str(d.serial_number),
                        simulated=False,
                    ))
            except Exception:
                pass
        if include_simulated:
            infos.append(DeviceInfo(
                label=f"Simulated device ({SIMULATED_SERIAL})",
                serial=SIMULATED_SERIAL,
                simulated=True,
            ))
        return infos

    @classmethod
    def open_serial(cls, serial: str) -> "SpectrometerInterface":
        """Open a specific device by serial number.

        ``SIMULATED_SERIAL`` opens the built-in simulated device.
        """
        if serial == SIMULATED_SERIAL:
            return cls(SimulatedSpectrometer(), simulated=True)
        if not _SEABREEZE_AVAILABLE:
            raise SpectrometerError("seabreeze backend is not installed.")
        try:
            for d in list_devices() or []:
                if str(d.serial_number) == str(serial):
                    return cls(Spectrometer(d), simulated=False)
        except Exception as exc:
            raise SpectrometerError(f"Could not open device {serial}: {exc}") from exc
        raise SpectrometerError(f"Device with serial '{serial}' not found.")

    def is_alive(self) -> bool:
        """Whether a device handle is currently held.

        We deliberately do NOT call ``list_devices()`` here: re-enumerating the
        USB bus while a device is open crashes some seabreeze backends (a native
        crash that Python cannot catch). A genuine disconnect instead surfaces
        as an error on the next acquisition, which the GUI reports.
        """
        return self._device is not None

    @property
    def model(self) -> str:
        return getattr(self._device, "model", "Unknown")

    @property
    def serial_number(self) -> str:
        return getattr(self._device, "serial_number", "Unknown")

    def integration_limits_micros(self):
        """Return the device's (min, max) integration time in microseconds.

        Returns ``(None, None)`` if the device does not report limits.
        """
        try:
            lims = getattr(self._device, "integration_time_micros_limits", None)
            if lims:
                return int(lims[0]), int(lims[1])
        except Exception:
            pass
        return None, None

    def set_integration_time_ms(self, milliseconds: float) -> float:
        """Set the integration time, clamped to the device's limits.

        Returns the integration time actually applied (ms), which may differ
        from the request if the device clamped it.
        """
        micros = int(round(milliseconds * 1000))
        lo, hi = self.integration_limits_micros()
        if lo is not None and hi is not None:
            micros = max(lo, min(micros, hi))
        self._device.integration_time_micros(micros)
        return micros / 1000.0

    def stabilize(self, correct_dark_counts: bool = False,
                  correct_nonlinearity: bool = False) -> None:
        """Discard one spectrum after an integration-time change.

        Real Ocean devices return one spectrum still acquired at the *previous*
        integration time after the setting is changed, so the first read must
        be thrown away. The simulator has no such buffering, so this is a no-op
        there (and avoids doubling simulated capture time).
        """
        if self.simulated:
            return
        try:
            self.intensities(correct_dark_counts=correct_dark_counts,
                             correct_nonlinearity=correct_nonlinearity)
        except Exception:
            pass

    def wavelengths(self) -> np.ndarray:
        return np.asarray(self._device.wavelengths(), dtype=float)

    def intensities(self, correct_dark_counts: bool = False,
                    correct_nonlinearity: bool = False) -> np.ndarray:
        """Read a spectrum, optionally with device-level corrections.

        ``correct_dark_counts`` and ``correct_nonlinearity`` require hardware
        support (electric-dark pixels / EEPROM coefficients). Devices lacking
        them raise; :meth:`probe_corrections` tells that apart from a failed read.
        """
        try:
            data = self._device.intensities(
                correct_dark_counts=correct_dark_counts,
                correct_nonlinearity=correct_nonlinearity,
            )
        except SpectrometerError:
            raise
        except Exception as exc:
            raise SpectrometerError(f"Could not read the spectrometer: {exc}") from exc
        return np.asarray(data, dtype=float)

    def probe_corrections(self, correct_dark_counts: bool,
                          correct_nonlinearity: bool) -> None:
        """Take one reading with the given corrections to verify support.

        Raises :class:`CorrectionUnsupported` if a plain read works but the
        corrected one doesn't, or :class:`SpectrometerError` if nothing reads.
        """
        try:
            self.intensities(correct_dark_counts=correct_dark_counts,
                             correct_nonlinearity=correct_nonlinearity)
        except SpectrometerError as exc:
            self.intensities()
            raise CorrectionUnsupported(
                f"This device does not support the requested correction "
                f"(dark={correct_dark_counts}, nonlinearity={correct_nonlinearity}): {exc}"
            ) from exc

    def supports_electric_dark(self) -> bool:
        """Whether electric-dark correction is available (no acquisition).

        Uses the device's electric-dark pixel list (``_dp`` in seabreeze);
        returns False if unknown so the feature isn't auto-enabled blindly.
        """
        if self.simulated:
            return bool(getattr(self._device, "supports_electric_dark", False))
        return bool(getattr(self._device, "_dp", None))

    def supports_nonlinearity(self) -> bool:
        """Whether nonlinearity coefficients are available (no acquisition)."""
        if self.simulated:
            return bool(getattr(self._device, "supports_nonlinearity", False))
        return bool(getattr(self._device, "_nc", None))

    def close(self) -> None:
        try:
            self._device.close()
        except Exception:
            pass
        self._device = None


def backend_status() -> str:
    if not _SEABREEZE_AVAILABLE:
        return "seabreeze not installed - simulation only"
    try:
        devices = list_devices() or []
    except Exception as exc:
        return f"seabreeze installed - device query failed ({exc})"
    if not devices:
        return "seabreeze installed - no device connected (simulation)"
    return f"seabreeze installed - {len(devices)} device(s) found"
