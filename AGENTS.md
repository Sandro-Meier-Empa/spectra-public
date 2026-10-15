# Repository Guidelines

## Project Overview

This repository contains `spectra` (SPECTRA: Synthetic Plume Embedding in
Cubes for Trace-gas Remote-sensing Applications), a Python package that embeds
simulated trace-gas plumes from large-eddy simulations into real observations
of imaging spectrometers and generates synthetic segmentation datasets. It is
the sensor-independent middle layer between the MicroHH processing chain
(plume columns), sensor-specific retrieval packages such as `aviris4_ch4`, and
the `prism` machine-learning repository. The package uses a `src/` layout.

## Repository Layout

- `src/spectra/contracts/`: xarray data contracts (plume column, sensor scene,
  column product, absorption table) with validators.
- `src/spectra/types.py`, `protocols.py`, `registry.py`: shared dataclasses,
  structural protocols for sensor adapters, retrievers, unit-absorption
  providers and regridders, and the entry-point plugin registry.
- `src/spectra/geometry/`: coordinate transforms, pixel corner geometry,
  plume placement in a swath, window selection.
- `src/spectra/regrid/`: conservative regridders (polygon overlap, bucket,
  CDO).
- `src/spectra/spectral/`: absorption look-up tables, Lambert-Beer radiance
  embedding, unit-absorption providers.
- `src/spectra/retrievers/`, `src/spectra/sensors/`: first-party plugins.
- `src/spectra/plume/`, `catalog/`, `recipe/`, `export/`: plume-column I/O and
  augmentation, scene catalogue and QA, recipe-driven seeded generation,
  export to the prism contract.
- `src/spectra/testing/`: synthetic sensors, providers and retrievers plus the
  adapter conformance suite.
- `scripts/`: runnable helpers (golden-fixture builder, data preparation).
- `tests/`: pytest suite; `tests/golden/` holds the recorded aviris4-ch4
  reference outputs the port is checked against.
- `notebooks/`, `docs/`: workflow notebooks and design documentation.

## Environment And Dependencies

- Python `>=3.11` is required.
- Install with `python -m pip install -e ".[dev]"`; sensor adapters need their
  extras (`[aviris4]`, `[emit]`).
- On CSCS Alps prefix Python commands with `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1` or imports can hang.
- Radiative-transfer resources (libRadtran optical depths, ARTS profiles, solar
  reference) and real scene data are supplied at runtime through paths and
  the `SPECTRA_TEST_RESOURCES` / `SPECTRA_TEST_DATA` /
  `SPECTRA_TEST_AVIRIS4_SCENES` / `SPECTRA_TEST_MICROHH_CASE` environment
  variables in tests; do not assume they are present.

## Coding Conventions

- Prefer existing package patterns over new abstractions. Sensor-specific
  knowledge belongs in `sensors/`, `retrievers/` and `spectral/providers/`;
  everything else must work for any sensor through the contracts.
- No directory-layout or file-name knowledge anywhere, adapters included:
  SPECTRA never searches directories, globs for scene files or parses file
  names. The user lists every file of every scene explicitly (a scene table
  read by `spectra.sensors.table`, `record_from_files`, or `--radiance`/
  `--metadata`/`--file ROLE=PATH` on the command line); adapters declare
  their file roles (`ROLES`) and only open what they are given. Examples and
  docs use `/path/to/...` placeholders, never site-specific paths.
- `aviris4_ch4`, `ddeq`, `microhh_processing_chain` and `prism` must never
  import `spectra`; keep the dependency direction one-way.
- Keep scientific data handling explicit with `xarray`, `numpy`, `pandas`,
  `shapely` and `Path` APIs. Contract violations raise `ContractError`.
- Type annotations are expected for new public functions and methods.
- Use concise Google-style docstrings matching the Ruff configuration.
- Import order: standard library, third party, local package.
- Keep comments focused on non-obvious geometry, radiative-transfer, unit or
  batch-processing behaviour.
- This repository **does** keep a test suite: it is the regression oracle for
  code moved out of `aviris4_ch4`. Add tests for new behaviour.
- Avoid committing generated caches, local environment files, large outputs or
  bulky hyperspectral products. The pre-commit hook rejects files above 6000 KB.

## Workflow Notes

- Dimension names are part of the contracts: sensor geometry uses
  `(across, along, band)`, map products `(y, x)`, plume columns `(time, y, x)`.
- Columns are stored in `kg m-2`; conversions go through `ucat`.
- Do not run remote jobs, submit scheduler scripts, transfer data or access
  external scientific resources unless the user explicitly asks for that.
