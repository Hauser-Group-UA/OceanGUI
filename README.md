# Ocean Spectrometer GUI

A cross-platform (Windows + Linux) desktop GUI for acquiring spectra from
**Ocean Optics / Ocean Insight** spectrometers using the
[python-seabreeze](https://github.com/ap--/python-seabreeze) backend.

If no spectrometer (or backend) is present, the app automatically runs in
**simulation mode** so every feature can be tried offline.

---

## Features

- **Three modes** (the **Mode** menu, Ctrl+1/2/3):
  - **Capture** - acquire spectra (everything below);
  - **View** - open a saved run and step through its scans (or jump to a
    scan number) next to the run average. It follows the most recent run
    until you open a specific file;
  - **Compare** - overlay the averages of several saved runs, in a reorderable
    list (top = drawn on top; untick a run to hide it without removing it) with
    colour-blind-safe colours that switch to dashed/dotted lines after four, an
    optional clickable legend with editable labels, a **Redraw plot** button so
    large comparisons stay responsive, and **Export data…** to a CSV with one
    column per ticked run.
- **Wavelength / energy x-axis**: click any plot's x-axis label to switch
  between wavelength (nm) and energy (eV) (and Raman shift for Raman data).
  An optional **Jacobian** setting (Off by default, next to the outlier
  threshold) converts intensities to per eV with λ²/hc on an energy axis
  (counts keep their total; irradiance becomes µW/cm²/eV; ratios are not
  rescaled).
- **Read values off the plots**: hover to see the value under the cursor;
  click to mark a point; right-click to clear. Readouts always give both
  wavelength and energy (with the Jacobian-scaled intensity when it is on).
- **Automatic outlier filtering** (e.g. cosmic-ray spikes): each wavelength is
  checked against the median of all scans with a robust σ, adjusted for the
  number of scans; choose Off / 1.5σ / 2σ / 2.5σ / 3σ (default) live, from the
  Acquire or Display tab.
- **Live device panel**: a perpetual connection indicator (green = connected,
  red = disconnected), a drop-down of available devices, and **Connect**
  (also used to change device), **Reconnect**, and **Refresh** buttons. The
  status auto-refreshes every 2 s.
- **Measurement modes** (like OceanView): **Scope** (raw counts), **Scope minus
  dark**, **Absorbance**, **Transmittance (%)**, **Reflectance (%)**,
  **Absolute irradiance** (µW/cm²/nm, from a radiometric calibration file) and
  **Raman shift** (cm⁻¹, relative to an excitation wavelength). The plot axes
  are labelled to match the selected mode (the x-axis becomes Raman shift in
  Raman mode).
- **Background & reference capture**: store an averaged **dark/background** for
  subtraction and a **reference** for the ratio modes (with a warning if they
  were taken at a different integration time).
- **Device corrections** (gated by hardware support, with clear popups when a
  device lacks them): **electric-dark** and **nonlinearity** correction, plus
  software **boxcar smoothing**.
- **Side-by-side plots**: the current integration (left) and the running
  average integration (right).
- **Paper-quality figures**: descriptive axis labels with units and *no* plot
  title, inward tick marks on all four sides, data drawn edge-to-edge so ticks
  reach the borders, and **600 DPI** output with a publication-friendly
  font-to-figure ratio.
- **Placeholder axes**: example dummy axes are shown until real data arrives.
- **Two run modes** (mutually exclusive):
  - *Number of integrations* — run an exact count, or
  - *Total integration time* — run for a total exposure (count = total ÷ single).
- **Single integration time** and **down time between integrations** settings,
  in ms, s, min or h.
- **Uncertainty toggles**: 1σ / 2σ **bars** and 1σ / 2σ **bands** on the average.
- **Required run name** before a run can start.
- **Automatic outputs** on completion (into `saved_data/<name>_<timestamp>/`):
  - `*_data.csv` — single integration time, wavelengths, every integration
    and the outlier-filtered average/std.
  - `*_average_3sigma.csv` and `*_average_unfiltered.csv` — two columns
    (wavelength, average) with the 3σ outlier filter and with no filter, that
    open directly in Excel.
  - `*_total.png` — picture of the total/average integration.
  - `*_average.png` — average integration, no bars/bands.
  - `*_average_overlay.png` — average in **red** over each individual
    integration in **grey**.
- **Save outputs (current settings)** button — re-writes the run's files with
  the current outlier filter and x-axis, plus the average plot with the
  currently-enabled uncertainty overlays.
- **Export data…** (Display tab) — the run average with the current outlier
  filter as a two-column CSV, in the plot's x-axis units.
- **Interrupt** button — stops an in-progress run after an *"Are you sure?"*
  confirmation; integrations collected so far are still saved.
- **In-app Help menu** (also `F1`).
- **Desktop shortcut** with a generated icon (white background, black border,
  red spectrum line), created by the installer.

---

## Installation

### Linux / macOS

```bash
git clone <this-repo> OceanGUI
cd OceanGUI
./install.sh
```

### Windows

Double-click `install.bat`, **or** from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

The installer will:

1. create a `.venv` virtual environment in the repo,
2. install all dependencies from `requirements.txt`,
3. generate the application icon,
4. (Linux) configure seabreeze udev rules via `seabreeze_os_setup`,
5. create a **desktop shortcut**.

### Air-gapped / flash-drive install (no internet on the target machine)

There are two ways to deploy to a computer with no internet, depending on
whether that computer already has Python:

| Target machine | Use | Needs Python on target? |
| --- | --- | --- |
| **No Python installed** | **A. Standalone build** (recommended) | **No** |
| Already has Python | B. Offline wheel bundle | Yes |

#### A. Standalone build — no Python needed on the target

Bundles Python + every dependency (including the seabreeze hardware backend)
into a double-clickable app. PyInstaller can't cross-compile, so build on an
**online machine of the same OS** as the target.

1. **On an online machine (same OS as target)**:

   ```bash
   ./build_standalone.sh        # Linux/macOS
   # or:  powershell -ExecutionPolicy Bypass -File build_standalone.ps1   (Windows)
   ```

   This produces `dist/OceanSpectrometerGUI/` (Python and all deps embedded).

2. **Copy that `dist/OceanSpectrometerGUI` folder** onto the flash drive and
   over to the air-gapped machine.

3. **On the target**, run the `OceanSpectrometerGUI` executable inside it —
   no installation, no Python, no internet. Acquisition output is saved to a
   `saved_data/` folder created next to the executable.

   Use `--onefile` for a single executable file instead of a folder.

#### B. Offline wheel bundle — target already has Python

1. **On an online machine** (ideally the same OS/arch as the target):

   ```bash
   ./prepare_offline.sh          # Linux/macOS
   # or:  powershell -ExecutionPolicy Bypass -File prepare_offline.ps1   (Windows)
   ```

   This downloads every wheel into `vendor/wheels/`. For a different platform:

   ```bash
   python assets/prepare_offline.py --target windows64 --python-version 3.11
   ```

2. **Copy the whole `OceanGUI` folder** (including `vendor/wheels/`) onto the
   flash drive and over to the offline machine.

3. **On the offline machine**, run the normal installer:

   ```bash
   ./install.sh        # or install.bat on Windows
   ```

   It auto-detects `vendor/wheels/` and installs with `pip --no-index`, so **no
   network access is used**.

See [vendor/README.md](vendor/README.md) for more detail.

---

## Running

- **Standalone build**: run the `OceanSpectrometerGUI` executable (no Python).
- **Desktop shortcut**: *Ocean Spectrometer GUI*
- **Linux / macOS**: `./run.sh`
- **Windows**: `run.bat`
- **Directly**: `python -m ocean_gui.main` (with the venv active)

---

## Usage

1. Pick a device from the **Available devices** drop-down and press
   **Connect** (or just **Start** — it connects to the selection automatically).
   The indicator turns green when connected. Use **Reconnect** after a
   replug, or **Refresh** to re-scan the bus.
2. Set the **single integration time**, **down time**, and pick a **run mode**.
   Optionally pick a **measurement mode** — for Absorbance/Transmittance/
   Reflectance, first **Capture dark** (light blocked) and **Capture reference**.
3. Enter a **run name** (required — Start stays disabled until you do).
4. Press **Start**. Plots update live; toggle 1σ/2σ bars and bands anytime.
   Press **Interrupt** to stop early (with confirmation) — partial data is kept.
5. On completion the CSVs and figures are saved automatically. Use
   **Save outputs (current settings)** after changing the outlier filter,
   x-axis or bars/bands to re-write them.
6. Switch to **View** (Mode menu) to inspect individual scans, or **Compare**
   to overlay several runs.

All output is stored inside the repository under `saved_data/`.

---

## Hardware notes (seabreeze)

- On **Linux**, USB access requires udev rules. The installer runs
  `seabreeze_os_setup`; if you skipped it, run that command once manually and
  replug the device.
- On **Windows**, you may need the device driver from Ocean Insight / a
  WinUSB/libusb driver (e.g. via Zadig) for some models.
- See the
  [seabreeze quickstart](https://github.com/ap--/python-seabreeze/blob/main/docs/source/quickstart.rst).

---

## Project layout

```
OceanGUI/
├── ocean_gui/
│   ├── main.py          # entry point
│   ├── gui.py           # PyQt5 main window, Mode menu + Capture mode
│   ├── view_mode.py     # View mode (browse a saved run scan by scan)
│   ├── compare_mode.py  # Compare mode (overlay run averages)
│   ├── widgets.py       # shared widgets (interactive plot, side panel, ...)
│   ├── acquisition.py   # settings model + acquisition/capture threads
│   ├── spectrometer.py  # seabreeze wrapper + simulation fallback
│   ├── processing.py    # measurement modes, dark/reference, smoothing, outlier filter
│   ├── units.py         # wavelength / energy / Raman-shift axes
│   ├── plotting.py      # shared matplotlib rendering
│   └── storage.py       # CSV / figure saving and loading
├── assets/
│   ├── make_icon.py        # generates the app icon
│   ├── make_shortcut.py    # creates the desktop shortcut (Linux/Windows)
│   ├── prepare_offline.py  # downloads wheels for an offline bundle
│   └── build_standalone.py # builds the no-Python standalone app (PyInstaller)
├── standalone_entry.py  # entry point for the standalone build
├── vendor/wheels/       # offline dependency bundle (built on demand)
├── saved_data/          # acquisition output (next to the app)
├── install.sh / run.sh / prepare_offline.sh / build_standalone.sh         # Linux / macOS
├── install.ps1 / install.bat / run.bat / prepare_offline.ps1 / build_standalone.ps1   # Windows
├── requirements.txt        # runtime dependencies
└── build-requirements.txt  # extra tooling to build the standalone app
```
