import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-9

OUTLIER_CHOICES = (0.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_OUTLIER_SIGMA = 3.0
_MAD_TO_SIGMA = 1.4826
_MEAN_AD_TO_SIGMA = 1.2533

_CAL_SIGMAS = (1.5, 2.0, 2.5, 3.0)
_CAL_TABLE = {
    3: (1.603, 3.912, 11.876, 45.803),
    4: (1.114, 1.683, 2.772, 5.135),
    5: (1.288, 1.926, 3.178, 5.869),
    6: (1.137, 1.475, 2.021, 2.972),
    7: (1.187, 1.534, 2.105, 3.107),
    8: (1.116, 1.349, 1.696, 2.241),
    9: (1.139, 1.374, 1.731, 2.292),
    10: (1.097, 1.274, 1.527, 1.901),
    11: (1.110, 1.287, 1.542, 1.924),
    12: (1.084, 1.227, 1.423, 1.702),
    13: (1.091, 1.233, 1.430, 1.715),
    14: (1.072, 1.191, 1.351, 1.573),
    15: (1.079, 1.197, 1.358, 1.581),
    16: (1.064, 1.166, 1.300, 1.483),
    17: (1.068, 1.169, 1.305, 1.487),
    18: (1.058, 1.147, 1.263, 1.419),
    19: (1.061, 1.150, 1.266, 1.422),
    20: (1.052, 1.132, 1.234, 1.370),
    21: (1.055, 1.133, 1.235, 1.370),
    22: (1.047, 1.118, 1.210, 1.329),
    23: (1.049, 1.120, 1.211, 1.330),
    24: (1.044, 1.108, 1.191, 1.299),
    25: (1.045, 1.110, 1.192, 1.299),
    26: (1.040, 1.099, 1.174, 1.270),
    27: (1.042, 1.101, 1.176, 1.274),
    28: (1.037, 1.092, 1.161, 1.249),
    29: (1.039, 1.093, 1.162, 1.250),
}
_CAL_FIT = ((1.0852, -0.1240), (2.4806, 4.1871), (4.1327, 12.6717), (6.0898, 27.3070))


def threshold_factor(n_scans: int, threshold_sigma: float) -> float:
    """How much to widen a k-std outlier cut for ``n_scans`` scans (see above).

    Thresholds between the calibrated ones are interpolated; outside
    1.5-3-std the nearest calibrated column is used.
    """
    if n_scans in _CAL_TABLE:
        row = _CAL_TABLE[n_scans]
    else:
        row = tuple(1.0 + a / n_scans + b / n_scans ** 2 for a, b in _CAL_FIT)
    return float(np.interp(threshold_sigma, _CAL_SIGMAS, row))


class MeasurementMode(Enum):
    SCOPE = "scope"
    DARK_SUBTRACT = "dark_subtract"
    ABSORBANCE = "absorbance"
    TRANSMITTANCE = "transmittance"
    REFLECTANCE = "reflectance"
    IRRADIANCE = "irradiance"
    RAMAN = "raman"


MODE_LABELS = {
    MeasurementMode.SCOPE: "Scope (raw counts)",
    MeasurementMode.DARK_SUBTRACT: "Scope minus dark",
    MeasurementMode.ABSORBANCE: "Absorbance",
    MeasurementMode.TRANSMITTANCE: "Transmittance (%)",
    MeasurementMode.REFLECTANCE: "Reflectance (%)",
    MeasurementMode.IRRADIANCE: "Irradiance (absolute)",
    MeasurementMode.RAMAN: "Raman shift",
}

MODE_YLABELS = {
    MeasurementMode.SCOPE: "Intensity (counts)",
    MeasurementMode.DARK_SUBTRACT: "Intensity (counts, dark-subtracted)",
    MeasurementMode.ABSORBANCE: "Absorbance (AU)",
    MeasurementMode.TRANSMITTANCE: "Transmittance (%)",
    MeasurementMode.REFLECTANCE: "Reflectance (%)",
    MeasurementMode.IRRADIANCE: "Irradiance (µW/cm²/nm)",
    MeasurementMode.RAMAN: "Intensity (counts, dark-subtracted)",
}

XLABEL_WAVELENGTH = "Wavelength (nm)"
XLABEL_RAMAN = "Raman shift (cm⁻¹)"


def requires_dark(mode: MeasurementMode) -> bool:
    return mode is not MeasurementMode.SCOPE


def requires_reference(mode: MeasurementMode) -> bool:
    return mode in (MeasurementMode.ABSORBANCE,
                    MeasurementMode.TRANSMITTANCE,
                    MeasurementMode.REFLECTANCE)


def requires_calibration(mode: MeasurementMode) -> bool:
    return mode is MeasurementMode.IRRADIANCE


def requires_excitation(mode: MeasurementMode) -> bool:
    return mode is MeasurementMode.RAMAN


def raman_shift(wavelengths: np.ndarray, excitation_nm: float) -> np.ndarray:
    """Raman shift in cm⁻¹ for the given wavelengths and excitation laser."""
    return (1.0e7 / float(excitation_nm)) - (1.0e7 / np.asarray(wavelengths, dtype=float))


def boxcar_smooth(arr: np.ndarray, width: int) -> np.ndarray:
    """Boxcar (moving-average) smoothing with a half-window of ``width``.

    ``width=0`` returns the array unchanged. Edges are handled by edge-padding
    so the output keeps the same length without end artifacts.
    """
    if width <= 0:
        return np.asarray(arr, dtype=float)
    window = 2 * int(width) + 1
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(np.asarray(arr, dtype=float), width, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


@dataclass
class Processor:
    """Holds the current mode plus stored dark/reference, calibration, etc."""

    mode: MeasurementMode = MeasurementMode.SCOPE
    dark: Optional[np.ndarray] = None
    reference: Optional[np.ndarray] = None
    boxcar_width: int = 0
    calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None
    collection_area_cm2: float = 1.0
    excitation_nm: Optional[float] = None

    def ylabel(self) -> str:
        return MODE_YLABELS[self.mode]

    def xlabel(self) -> str:
        return XLABEL_RAMAN if self.mode is MeasurementMode.RAMAN else XLABEL_WAVELENGTH

    def xvalues(self, wavelengths: np.ndarray) -> np.ndarray:
        """X-axis values for plotting (wavelength, or Raman shift for Raman)."""
        if self.mode is MeasurementMode.RAMAN and self.excitation_nm:
            return raman_shift(wavelengths, self.excitation_nm)
        return np.asarray(wavelengths, dtype=float)

    def missing_requirement(self) -> Optional[str]:
        """Return an error message if the mode needs data that isn't set."""
        label = MODE_LABELS[self.mode]
        if requires_dark(self.mode) and self.dark is None:
            return f"{label} needs a stored dark/background spectrum. Capture one first."
        if requires_reference(self.mode) and self.reference is None:
            return f"{label} needs a stored reference spectrum. Capture one first."
        if requires_calibration(self.mode) and self.calibration is None:
            return f"{label} needs a radiometric calibration file. Load one first."
        if requires_excitation(self.mode) and not self.excitation_nm:
            return f"{label} needs an excitation wavelength (nm)."
        return None

    def apply(self, raw: np.ndarray, wavelengths: Optional[np.ndarray] = None,
              integration_time_s: Optional[float] = None) -> np.ndarray:
        """Transform a raw spectrum's intensities for the current mode.

        ``wavelengths`` and ``integration_time_s`` are required for irradiance
        (and ignored by the simpler modes).
        """
        s = boxcar_smooth(raw, self.boxcar_width)
        if self.mode is MeasurementMode.SCOPE:
            return s

        d = boxcar_smooth(self.dark, self.boxcar_width) if self.dark is not None \
            else np.zeros_like(s)
        if self.mode in (MeasurementMode.DARK_SUBTRACT, MeasurementMode.RAMAN):
            return s - d

        if self.mode is MeasurementMode.IRRADIANCE:
            return self._irradiance(s, d, wavelengths, integration_time_s)

        r = boxcar_smooth(self.reference, self.boxcar_width) \
            if self.reference is not None else np.ones_like(s)
        num = s - d
        denom = r - d
        if self.mode is MeasurementMode.ABSORBANCE:
            ratio = np.clip(num, _EPS, None) / np.clip(denom, _EPS, None)
            return -np.log10(ratio)
        denom_safe = np.where(np.abs(denom) < _EPS, _EPS, denom)
        return 100.0 * num / denom_safe

    def _irradiance(self, s, d, wavelengths, integration_time_s) -> np.ndarray:
        if wavelengths is None or integration_time_s is None or self.calibration is None:
            return s - d
        cal_wl, cal_coeff = self.calibration
        cal = np.interp(wavelengths, cal_wl, cal_coeff)
        dlambda = np.abs(np.gradient(np.asarray(wavelengths, dtype=float)))
        dlambda = np.where(dlambda < _EPS, _EPS, dlambda)
        t = max(float(integration_time_s), _EPS)
        area = max(float(self.collection_area_cm2), _EPS)
        return (s - d) * cal / (t * dlambda * area)


@dataclass
class ScanStats:
    """Per-pixel statistics across a stack of scans, after outlier removal."""

    average: np.ndarray
    std: np.ndarray
    n_used: np.ndarray
    n_removed: int
    threshold_sigma: float


def scan_statistics(scans, threshold_sigma: float = 0.0) -> ScanStats:
    """Mean and sample std of each pixel across scans, ignoring outliers.

    Outliers are judged one pixel at a time: a scan's value is dropped when it
    lies more than ``threshold_sigma`` std from the median of all scans at that
    pixel. std is estimated robustly (1.4826 x median absolute deviation), so an
    extreme value cannot hide itself by inflating std the way it would with a
    plain mean/std test, and the cut is widened for small scan counts (see
    :func:`threshold_factor`) so clean data isn't over-trimmed. Filtering
    needs at least 3 scans; a threshold of 0 keeps everything. Non-finite
    values are always ignored.
    """
    x = np.asarray(scans, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    finite = np.isfinite(x)
    keep = finite.copy()
    all_finite = bool(finite.all())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels, 0/0
        if threshold_sigma and x.shape[0] >= 3:
            data = x if all_finite else np.where(finite, x, np.nan)
            median = np.median if all_finite else np.nanmedian
            dev = np.abs(data - median(data, axis=0))
            sigma = _MAD_TO_SIGMA * median(dev, axis=0)
            flat = ~(sigma > 0)
            if flat.any():
                sigma[flat] = _MEAN_AD_TO_SIGMA * np.nanmean(dev[:, flat], axis=0)
            if all_finite:
                factor = threshold_factor(x.shape[0], threshold_sigma)
            else:
                n_valid = finite.sum(axis=0)
                factor = np.full(n_valid.shape, np.inf)  # < 3 values: never filter
                for n in np.unique(n_valid[n_valid >= 3]):
                    factor[n_valid == n] = threshold_factor(int(n), threshold_sigma)
            keep &= ~(dev > threshold_sigma * factor * sigma)

        n_used = keep.sum(axis=0)
        avg = np.where(keep, x, 0.0).sum(axis=0) / n_used
        resid = np.where(keep, x - avg, 0.0)
        var = (resid ** 2).sum(axis=0) / np.maximum(n_used - 1, 1)
    std = np.where(n_used > 1, np.sqrt(var), 0.0)
    avg = np.where(n_used > 0, avg, np.nan)
    return ScanStats(average=avg, std=std, n_used=n_used,
                     n_removed=int(finite.sum() - keep.sum()),
                     threshold_sigma=float(threshold_sigma or 0.0))
