# CM1 Storm Motion Estimator

Scientific desktop application for analyzing CM1 simulations and estimating storm motion from tracked updraft and vertical-vorticity centers.

> **Status:** This repository is under active development. Synthetic tests and successful application startup do not establish scientific validation. Compatibility with real CM1 r22.0 files must be checked with the read-only diagnostic below and validated against known CM1 grid conventions.

## Requirements

- Ubuntu under WSL2 (or Linux Mint/Ubuntu desktop)
- Python 3.10 or newer
- A WSLg/desktop-enabled environment for the graphical application; the test suite and NetCDF diagnostic can run headlessly

The project dependencies are declared in `pyproject.toml`: NumPy, SciPy, pandas, xarray, netCDF4, PySide6, PyQtGraph, Matplotlib, and pytest. The optional `dev` extra adds pytest-cov and Ruff. No separate NetCDF system library installation is normally needed when installing the declared Python packages in a supported environment.

## WSL2 Ubuntu setup

Open the Ubuntu terminal in WSL2 and install Python tooling:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Clone the repository and enter it:

```bash
git clone https://github.com/SusOg-wx/CM1-Storm-Motion-Estimator.git
cd CM1-Storm-Motion-Estimator
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the application and development/test dependencies from the repository metadata:

```bash
python -m pip install -e '.[dev]'
```

Check the interpreter version if needed:

```bash
python --version
```

It must be Python 3.10 or later. If Ubuntu's available `python3` is older, install/use a Python 3.10+ interpreter and substitute its command when creating the virtual environment.

## Run the complete test suite

With the virtual environment active and dependencies installed, from the repository root run:

```bash
python -m pytest -ra
```

The `-ra` option makes pytest report a final summary including passed, failed, and skipped tests (and reasons for skips). Keep the complete terminal output when reporting results. Tests have not been executed as part of this documentation update; do not infer a result from the command or from synthetic test coverage.

## Inspect real CM1 NetCDF output (read only)

The diagnostic utility is `tools/inspect_cm1_netcdf.py`. It opens the specified NetCDF file read-only through xarray and prints a JSON report. Provide one or more file paths as command-line arguments; quote paths containing spaces:

```bash
python tools/inspect_cm1_netcdf.py /path/to/cm1out_000001.nc
```

Multiple files can be inspected together:

```bash
python tools/inspect_cm1_netcdf.py /path/to/cm1out_000001.nc /path/to/cm1out_000002.nc
```

For files on the Windows filesystem, access them under `/mnt/<drive-letter>/`, for example:

```bash
python tools/inspect_cm1_netcdf.py '/mnt/c/Users/YourName/CM1/output/cm1out_000001.nc'
```

The JSON report includes variable names, dimensions, shapes, dtypes and attributes; recognized coordinate names, dimensions, units, values and horizontal spacing summaries; and `u`, `v`, and `w` dimensions, units, associated coordinate candidates, and time/vertical dimensions and values. The reported “mass-grid candidate”/“face or staggered candidate” is a name-based diagnostic, not a definitive proof of CM1 staggering. Compare the output to the actual CM1 r22.0 grid conventions before relying on a calculation. The utility does not write to the input file.

For readable terminal output saved separately from the input, you may redirect stdout:

```bash
python tools/inspect_cm1_netcdf.py /path/to/cm1out_000001.nc > cm1-netcdf-report.json
```

## Start the application

With the environment active and a graphical WSL desktop available:

```bash
python -m cm1_storm_motion_estimator
```

The GUI and analysis functionality are still under development. The vorticity calculation and synthetic tests are not a substitute for validating against real CM1 output.

## Scientific limitations and validation

The vorticity routine requires explicit xarray wind fields, native coordinate arrays with supported units, and a supplied common mass grid. Before using it with CM1 r22.0, inspect actual output and verify the u/v coordinate associations, staggering, unit conventions, missing-value encoding, vertical selection, and target grid. No real CM1 sample is currently included in this repository; real-data compatibility and scientific validation therefore remain unverified.

Do not use automatically generated motion recommendations without reviewing the inputs, tracking quality, and CM1 moving-reference-frame conventions. This application does not modify simulation output, CM1 source files, or namelists.

## License

MIT. See `LICENSE`.
