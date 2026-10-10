"""X-axis units (wavelength / energy / Raman shift) and intensity conversion.

By default an energy axis just re-plots the measured values. With the
Jacobian option on, intensities (densities *per unit x*) are multiplied by
|dλ/dE| = λ²/hc so peak shapes and areas are correct per unit energy:

- Irradiance (µW/cm²/nm) is converted exactly to µW/cm²/eV.
- Counts are per *pixel*; each pixel spans a different energy width, so they
  are divided by that width and renormalised so the total number of counts is
  unchanged (values mid-range stay about the same).
- Ratios (absorbance, transmittance, reflectance) are dimensionless and are
  never rescaled.
- Raman shift follows the usual Raman convention: counts are not rescaled.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np

from .processing import MODE_YLABELS, MeasurementMode, raman_shift

HC_EV_NM = 1239.841984

class XUnit(Enum):
    WAVELENGTH = "wavelength"
    ENERGY = "energy"
    RAMAN = "raman"


XUNIT_LABELS = {
    XUnit.WAVELENGTH: "Wavelength (nm)",
    XUnit.ENERGY: "Energy (eV)",
    XUnit.RAMAN: "Raman shift (cm⁻¹)",
}

_SYMBOLS = {XUnit.WAVELENGTH: "nm", XUnit.ENERGY: "eV", XUnit.RAMAN: "cm⁻¹"}
_DECIMALS = {XUnit.WAVELENGTH: 2, XUnit.ENERGY: 4, XUnit.RAMAN: 1}

_RATIO_MODES = (MeasurementMode.ABSORBANCE, MeasurementMode.TRANSMITTANCE,
                MeasurementMode.REFLECTANCE)

EXAMPLE_WAVELENGTHS = np.linspace(400.0, 800.0, 200)


@dataclass
class SpectralAxis:
    """A spectrum's x values and intensity scaling for one x-axis unit."""

    unit: XUnit
    x: np.ndarray
    scale: Union[np.ndarray, float]
    xlabel: str
    ylabel: str

    def apply(self, values):
        """Rescale intensities (1-D, or one row per scan) for this axis."""
        return np.asarray(values, dtype=float) * self.scale


def available_units(mode: Optional[MeasurementMode],
                    excitation_nm: Optional[float]) -> List[XUnit]:
    """The x-axis units that make sense for data taken in ``mode``."""
    units = [XUnit.WAVELENGTH, XUnit.ENERGY]
    if mode is MeasurementMode.RAMAN and excitation_nm:
        units.append(XUnit.RAMAN)
    return units


def spectral_axis(wavelengths, unit: XUnit, mode: Optional[MeasurementMode] = None,
                  excitation_nm: Optional[float] = None,
                  ylabel: Optional[str] = None, jacobian: bool = False) -> SpectralAxis:
    """Express a spectrum on ``unit``; falls back to wavelength if unavailable.

    ``ylabel`` defaults to the mode's label; unknown modes are treated as counts.
    ``jacobian`` rescales intensities to per unit energy on an energy axis.
    """
    wl = np.asarray(wavelengths, dtype=float)
    ylabel = ylabel or MODE_YLABELS.get(mode, "Intensity (counts)")
    if unit is XUnit.ENERGY:
        wl = np.clip(wl, 1e-6, None)
        if not jacobian:
            return SpectralAxis(unit, HC_EV_NM / wl, 1.0, XUNIT_LABELS[unit], ylabel)
        return SpectralAxis(unit, HC_EV_NM / wl, _energy_jacobian(wl, mode),
                            XUNIT_LABELS[unit], _energy_ylabel(mode, ylabel))
    if unit is XUnit.RAMAN and unit in available_units(mode, excitation_nm):
        return SpectralAxis(unit, raman_shift(wl, excitation_nm), 1.0,
                            XUNIT_LABELS[unit], ylabel)
    return SpectralAxis(XUnit.WAVELENGTH, wl, 1.0,
                        XUNIT_LABELS[XUnit.WAVELENGTH], ylabel)


def _energy_jacobian(wl: np.ndarray, mode: Optional[MeasurementMode]):
    if mode in _RATIO_MODES or wl.size < 2:
        return 1.0
    if mode is MeasurementMode.IRRADIANCE:
        return wl ** 2 / HC_EV_NM
    pixel_ev = HC_EV_NM * np.abs(np.gradient(wl)) / wl ** 2
    pixel_ev = np.maximum(pixel_ev, 1e-12 * float(np.max(pixel_ev)) + 1e-300)
    return float(np.mean(pixel_ev)) / pixel_ev


def _energy_ylabel(mode: Optional[MeasurementMode], ylabel: str) -> str:
    if mode is MeasurementMode.IRRADIANCE:
        return ylabel.replace("/nm", "/eV")
    return ylabel


def format_x(value: float, unit: XUnit) -> str:
    """e.g. '589.12 nm', '2.1049 eV', '1032.5 cm⁻¹'."""
    return f"{value:.{_DECIMALS[unit]}f} {_SYMBOLS[unit]}"


def format_y(value: float, ylabel: str) -> str:
    """The value with the unit taken from an axis label like 'Intensity (counts)'."""
    match = re.search(r"\(([^()]*)\)\s*$", ylabel or "")
    unit = match.group(1).split(",")[0].strip() if match else ""
    sep = "" if unit == "%" else " "
    return f"{value:.5g}{sep}{unit}" if unit else f"{value:.5g}"


@dataclass
class Spectrum:
    """The per-wavelength data behind a plotted line, for readouts in every unit."""

    wavelengths: np.ndarray
    values: np.ndarray
    mode: Optional[MeasurementMode] = None
    excitation_nm: Optional[float] = None
    ylabel: Optional[str] = None

    def describe(self, index: int, first: XUnit, jacobian: bool) -> List[str]:
        """'(x, ..., y)' groups for one point, the ``first`` unit leading.

        Units with the same intensity share a group, e.g. '(589.12 nm, 2.1046 eV,
        1834 counts)'; with the Jacobian on, energy gets its own rescaled group.
        """
        units = available_units(self.mode, self.excitation_nm)
        if first not in units:
            first = XUnit.WAVELENGTH
        groups = []
        for unit in [first] + [u for u in units if u is not first]:
            axis = spectral_axis(self.wavelengths, unit, self.mode, self.excitation_nm,
                                 self.ylabel, jacobian)
            scale = axis.scale if np.ndim(axis.scale) == 0 else axis.scale[index]
            y = format_y(float(self.values[index]) * scale, axis.ylabel)
            x = format_x(float(axis.x[index]), unit)
            for xs, shared in groups:
                if shared == y:
                    xs.append(x)
                    break
            else:
                groups.append(([x], y))
        return [f"({', '.join(xs)}, {y})" for xs, y in groups]
