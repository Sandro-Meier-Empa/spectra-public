# Repository Guidelines

## Project Overview

This repository contains `spectra` (SPECTRA: Synthetic Plume Embedding in
Cubes for Trace-gas Remote-sensing Applications), a Python package that embeds
simulated trace-gas plumes from large-eddy simulations into real imaging
spectrometer radiance (AVIRIS-3, AVIRIS-4, EMIT) and writes
the attenuated radiance back in the input's native file format with the
embedding metadata attached. It retrieves nothing and prepares nothing for
downstream applications: users feed the output into their own pipelines. It
sits between the MicroHH processing chain (plume columns) and whatever
consumes L1 radiance. The package uses a `src/` layout.

## Repository Layout

- `src/spectra/contracts/`: xarray contracts (plume column, sensor scene,
  absorption table, embedded output schema), units and `ContractError`.
- `src/spectra/types.py`, `protocols.py`: shared dataclasses (`Window`,
  `Placement`, `NativeLayout`, `SceneGeometry`, `ViewingGeometry`, ...) and
  the `SensorAdapter` and `ForwardModel` protocols.
- `src/spectra/sensors/`: one adapter per sensor (`aviris3`, `aviris4`,
  `emit`) plus shared native-file helpers; `SENSORS` maps the run-file names
  to them.
- `src/spectra/geometry/`, `src/spectra/regrid/`: anchors (geographic, pixel,
  random), placement, swath outline, exact polygon-overlap regridding.
- `src/spectra/spectral/`: Lambert-Beer kernel, absorption tables, the
  libRadtran/ARTS forward model (ported from `aviris4_ch4`) and a toy model.
- `src/spectra/run.py`, `pipeline.py`, `result.py`, `api.py`, `cli.py`: run
  files, the multi-plume pipeline, its result, `spectra.embed` and the
  `spectra` command.
- `src/spectra/export/native.py`: copy-and-patch export into the native file.
- `src/spectra/plume/`: plume-column I/O and partial columns.
- `src/spectra/testing/`: synthetic plumes and scenes, schema-exact synthetic
  native files, storage-level comparison and adapter conformance.
- `scripts/`: `golden/` (provenance of the frozen fixtures) and benchmarks.
- `tests/`: pytest suite; `tests/golden/` holds frozen reference outputs.
- `notebooks/`, `docs/`, `examples/`: per-sensor notebooks, documentation,
  example run files.

## Environment And Dependencies

- Python `>=3.11` is required.
- Install with `python -m pip install -e ".[dev]"`.
- On CSCS Alps prefix Python commands with `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1` or imports can hang.
- Radiative-transfer resources (libRadtran optical depths, ARTS profiles,
  solar reference) and real scene data are supplied at runtime through
  explicit paths and, in tests, the `SPECTRA_TEST_RESOURCES`,
  `SPECTRA_TEST_<SENSOR>_FILES`, `SPECTRA_TEST_DATA` and
  `SPECTRA_TEST_MICROHH_CASE` environment variables; do not assume they are
  present.

## Coding Conventions

- Prefer existing package patterns over new abstractions. Sensor-specific
  knowledge belongs in `sensors/`; everything else must work for any sensor
  through the `SensorScene` contract and the adapter's `NativeLayout`.
- No directory-layout or file-name knowledge anywhere, adapters and forward
  model included: SPECTRA never searches directories, globs for files or
  parses file names. The user lists every file of a scene by role and every
  forward-model resource file by role in the run file; adapters declare
  their roles (`ROLES`) and only open what they are given. Examples and docs
  use `/path/to/...` placeholders, never site-specific paths.
- Outputs are byte copies of the native radiance file: only plume pixels of
  the attenuated bands may change, inputs are never modified, and new
  metadata goes into the root group (`source` dimension, `source_*`
  variables, `spectra_*` attributes).
- `aviris4_ch4`, `ddeq`, `microhh_processing_chain` and `prism` must never
  import `spectra`, and `spectra` does not import them either.
- Keep scientific data handling explicit with `xarray`, `numpy`, `netCDF4`,
  `shapely` and `Path` APIs. Contract violations raise `ContractError`.
- Type annotations are expected for new public functions and methods.
- Use concise Google-style docstrings matching the Ruff configuration.
- Import order: standard library, third party, local package.
- Keep comments focused on non-obvious geometry, radiative-transfer, unit or
  storage behaviour.
- The test suite is the regression oracle: the golden fixtures freeze the
  ported numerics and the v0.1.0 pipeline output. Add tests for new
  behaviour; native-format changes need a schema-exact writer in
  `spectra.testing.native`.
- Avoid committing generated caches, local environment files, large outputs or
  bulky hyperspectral products. The pre-commit hook rejects files above 6000 KB.

## Workflow Notes

- Dimension names are part of the contracts: sensor geometry uses
  `(across, along, band)`, plume columns `(time, y, x)`; outputs keep the
  native dimension names.
- Columns are stored in `kg m-2`; conversions go through `ucat`. Radiance
  keeps the sensor's native units.
- Do not run remote jobs, submit scheduler scripts, transfer data or access
  external scientific resources unless the user explicitly asks for that.
