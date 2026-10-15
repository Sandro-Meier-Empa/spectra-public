# Changelog

## 2026-10-14

### Added

- Initial package: data contracts (`PlumeColumn`, `SensorScene`,
  `ColumnProduct`, absorption table) with validators, shared types,
  structural protocols and the entry-point plugin registry for sensors,
  retrievers, unit-absorption providers and regridders.
- Absorption look-up tables, Lambert-Beer radiance embedding and additive
  product embedding moved from `aviris4_ch4.embedding` (byte-identical, golden
  fixture in `tests/golden/`); `libradtran_arts` provider wrapping
  `aviris4_ch4.mf.UnitAbsorption`; analytic `toy` provider.
- Plume placement (anchor corner, reflection, rotation, random anchor draws
  with outside-mass bounds), CDO target/source preparation, `cdo remapcon`
  runner, conservative-mass validation and scene/window merging moved from
  `aviris4_ch4.regridding` and generalised to any `SensorScene`.
- Pure-Python conservative regridders: exact polygon overlap (`polygon`,
  default) and supersampled bucket (`bucket`), for sensor windows and map grids.
- AVIRIS-4 adapter (`aviris4`) over the `aviris4_ch4` NetCDF products and the
  `aviris4_mf` retriever wrapping the matched filter; `toy` sensor and
  retriever plus an adapter conformance suite.
- `spectra.pipeline` (place, regrid, embed, retrieve, export) and the CLI:
  `plugins`, `validate`, `embed radiance|product`, `catalog build|qa|status`,
  `generate`, `finalize`, `slurm`, `export-prism`.
- Scene catalogue (Parquet) with QA metrics and status lifecycle; recipe-driven
  seeded generation writing prism-compatible products and a pairings table.
- Plume-column I/O and augmentation (`flip`, emission scaling, wind transform,
  partial columns below an altitude).
- Scene tables (`spectra.sensors.table`): every scene is listed explicitly
  with the full path of each file role (CSV, YAML or in-memory rows; optional
  `scene_id`, `background_group` and filterable extra columns).

### Changed

- SPECTRA never derives file locations from directory layouts or file names.
  `SensorAdapter.list_scenes(scenes, **filters)` takes a scene table instead
  of a campaign root; `Aviris4Adapter` exposes `ROLES` and
  `record_from_files(radiance, metadata, shadow=, cloud=, product=,
  scene_id=, background_group=)` and takes only `chunks`/`shadow_threshold`.
  `spectra embed` takes `--radiance/--metadata/--shadow/--cloud/--product`
  or `--file ROLE=PATH` (plus `--scene-id`, `--background-group`);
  `spectra catalog build` takes `--scenes TABLE`; recipes use
  `sensor.scenes` (required when `catalog.path` is null);
  `SceneCatalog.build(adapter, scenes, records=, inspect=, ...)`. Tests read
  real flight lines from the scene table in `SPECTRA_TEST_AVIRIS4_SCENES`.
- Window attributes: regridders and `spectra.scene.window` use the `window_*`
  attributes only; a window is passed explicitly as a `Window` or carried
  as `window_*` attributes.

### Removed

- `--root`, `sensor.root`, `SPECTRA_TEST_AVIRIS4_ROOT`, `Aviris4Paths`, the
  altitude/line filters and the file-name parsing of the AVIRIS-4 adapter.
- The legacy `embedding_*` window attributes of remapped data.
