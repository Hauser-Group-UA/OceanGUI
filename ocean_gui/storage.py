import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .processing import MODE_LABELS, MODE_YLABELS, MeasurementMode


def _base_dir() -> Path:
    """Folder that ``saved_data`` lives next to.

    For a normal source checkout this is the repository root. For a frozen
    standalone build (PyInstaller) it is the folder containing the executable,
    so output is written beside the app (writable on a flash drive) rather
    than into the read-only bundle.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _base_dir()
DEFAULT_SAVE_DIR = REPO_ROOT / "saved_data"


def sanitize_name(name: str) -> str:
    """Sanitise a single path component (no separators)."""
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._") or "run"


def split_run_name(name: str):
    """Split a run name into (folder components, base name).

    A ``/`` or ``\\`` in the name is treated as a folder separator, so
    "batch1/sampleA" saves into ``<save_dir>/batch1/sampleA_<timestamp>/``.
    ``.`` and ``..`` components are dropped for safety.
    """
    parts = [p for p in re.split(r"[\\/]+", name.strip())
             if p.strip() not in ("", ".", "..")]
    parts = [sanitize_name(p) for p in parts]
    if not parts:
        parts = ["run"]
    return parts[:-1], parts[-1]


def run_basename(name: str) -> str:
    """The sanitised final component used as the file-name prefix."""
    return split_run_name(name)[1]


def run_directory(name: str, save_dir: Path = DEFAULT_SAVE_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folders, base = split_run_name(name)
    path = Path(save_dir).joinpath(*folders) / f"{base}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(
    path: Path,
    single_time_ms: float,
    wavelengths: np.ndarray,
    all_intensities: np.ndarray,
    average: np.ndarray,
    std: np.ndarray,
    metadata: dict = None,
) -> Path:
    """Write the spectra to CSV.

    ``metadata`` is an optional dict of extra ``# key,value`` header lines
    (e.g. measurement mode, corrections, smoothing, dark/reference state).
    """
    n_integrations = all_intensities.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# single_integration_time_ms,{single_time_ms}\n")
        fh.write(f"# n_integrations,{n_integrations}\n")
        for key, value in (metadata or {}).items():
            fh.write(f"# {key},{value}\n")
        writer = csv.writer(fh)
        header = ["wavelength_nm"]
        header += [f"integration_{i + 1}" for i in range(n_integrations)]
        header += ["average", "std"]
        writer.writerow(header)
        for j in range(wavelengths.size):
            row = [f"{wavelengths[j]:.4f}"]
            row += [f"{all_intensities[i, j]:.4f}" for i in range(n_integrations)]
            row += [f"{average[j]:.4f}", f"{std[j]:.4f}"]
            writer.writerow(row)
    return path


def threshold_tag(sigma: float) -> str:
    """File-name tag for an outlier threshold: '3sigma', '2.5sigma' or 'unfiltered'."""
    return f"{sigma:g}sigma" if sigma else "unfiltered"


def average_header(ylabel: str) -> str:
    """'Intensity (counts)' -> 'Average intensity (counts)'."""
    return f"Average {ylabel[:1].lower()}{ylabel[1:]}"


def save_columns_csv(path, headers, columns) -> Path:
    """Write columns under one header row, ready to open in Excel.

    No comment lines; the UTF-8 byte-order mark makes Excel show units such as
    µ and ² correctly. Missing (NaN) values are left blank.
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in zip(*columns):
            writer.writerow([_cell(v, ".8g" if k == 0 else ".6g") for k, v in enumerate(row)])
    return Path(path)


def _cell(value, fmt: str) -> str:
    return format(value, fmt) if np.isfinite(value) else ""


def save_average_csv(path: Path, wavelengths: np.ndarray, average: np.ndarray,
                     ylabel: str = "Intensity (counts)") -> Path:
    """Write a plain two-column file (wavelength, average) that opens in Excel."""
    return save_columns_csv(path, ["Wavelength (nm)", average_header(ylabel)],
                            [wavelengths, average])


_RUN_FOLDER = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{8}_\d{6})$")


@dataclass
class RunData:
    """A saved run read back from its ``*_data.csv`` file."""

    path: Path
    wavelengths: np.ndarray
    scans: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)
    header_ylabel: Optional[str] = None

    @property
    def n_scans(self) -> int:
        return int(self.scans.shape[0])

    @property
    def run_name(self) -> str:
        """The name the run was saved under (folder name minus its timestamp)."""
        match = _RUN_FOLDER.match(self.path.parent.name)
        if match:
            return match.group("name")
        stem = self.path.stem
        for suffix in ("_data", "_average"):
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    @property
    def display_name(self) -> str:
        """Run folder name (name + timestamp) when it has one, else the file name."""
        if _RUN_FOLDER.match(self.path.parent.name):
            return self.path.parent.name
        return self.path.name

    @property
    def acquired(self) -> Optional[datetime]:
        match = _RUN_FOLDER.match(self.path.parent.name)
        if match:
            try:
                return datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        return None

    @property
    def mode(self) -> Optional[MeasurementMode]:
        mode_id = self.metadata.get("measurement_mode_id")
        for mode in MeasurementMode:
            if mode.value == mode_id:
                return mode
        label = self.metadata.get("measurement_mode")
        for mode, text in MODE_LABELS.items():
            if text == label:
                return mode
        for mode, text in MODE_YLABELS.items():
            if text == self.ylabel:
                return mode
        return None

    @property
    def ylabel(self) -> str:
        if self.metadata.get("quantity"):
            return self.metadata["quantity"]
        if self.header_ylabel:
            return self.header_ylabel
        return "Intensity (counts)"

    @property
    def excitation_nm(self) -> Optional[float]:
        return _to_float(self.metadata.get("excitation_nm"))

    @property
    def integration_ms(self) -> Optional[float]:
        return _to_float(self.metadata.get("single_integration_time_ms"))


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_text(path: Path) -> str:
    """Files written on Windows by older versions may not be UTF-8."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def load_run(path) -> RunData:
    """Read a run's ``*_data.csv`` (or a two-column average/export CSV)."""
    path = Path(path)
    lines = _read_text(path).splitlines()
    metadata: Dict[str, str] = {}
    header, start = None, 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            key, _, value = line[1:].partition(",")
            metadata[key.strip()] = value.strip()
        elif line.strip():
            header, start = next(csv.reader([line])), i + 1
            break
    if not header or not header[0].strip().lower().startswith("wavelength"):
        raise ValueError("not a saved spectrum: the first column must be the wavelength.")
    try:
        data = np.loadtxt(lines[start:], delimiter=",", ndmin=2)
    except ValueError as exc:
        raise ValueError(f"could not read the numeric data ({exc}).") from exc
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("the file holds no spectrum data.")

    names = [h.strip() for h in header]
    columns = [i for i, name in enumerate(names) if name.startswith("integration_")]
    if not columns:
        columns = [names.index("average")] if "average" in names else [1]
    header_ylabel = None
    if len(names) == 2 and names[1].lower().startswith("average "):
        text = names[1][len("average "):]
        header_ylabel = text[:1].upper() + text[1:]
    return RunData(path=path, wavelengths=data[:, 0].copy(),
                   scans=np.ascontiguousarray(data[:, columns].T),
                   metadata=metadata, header_ylabel=header_ylabel)


def find_latest_run(save_dir: Path, extra: Iterable[Optional[Path]] = ()) -> Optional[Path]:
    """The most recently written ``*_data.csv`` under ``save_dir`` (or in ``extra``)."""
    candidates = [Path(p) for p in extra if p is not None]
    try:
        candidates += list(Path(save_dir).rglob("*_data.csv"))
    except OSError:
        pass
    best, best_time = None, None
    for candidate in candidates:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if best_time is None or mtime > best_time:
            best, best_time = candidate, mtime
    return best
