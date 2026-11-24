# Changelog

## 0.2.0 (unreleased)

SPECTRA is now embed-only: it imports a scene's native files, embeds N LES
plumes with Lambert-Beer attenuation and writes the attenuated radiance back
in the input's native format with the embedding metadata. Version 0.1.0
(tag `v0.1.0`) is the last release with the matched filter and the prism
dataset generation.

### Added

- `spectra.embed(run)` and `spectra embed run.yaml [--dry-run]`: one run file
  (sensor, scene files by role, output, seed, forward-model resource files by
  role, plumes) embeds N plumes into one scene. `spectra sensors` lists the
  file roles.
- Copy-and-patch export (`spectra.export.native`): the output is a byte copy
  of the native radiance file in which only the plume pixels of the
  attenuated bands change; a `source` dimension carries
  `injected_<gas>_mass` and per-plume `source_*` provenance, and `spectra_*`
  attributes hold the run provenance including the resolved run file
  (`spectra_run_config`), which reproduces the output.
- EMIT L1B adapter (`emit`, RAD + OBS) and AVIRIS-3 L1B adapter (`aviris3`,
  ORNL DAAC RDN + OBS granules, radiance in the `radiance` group, CRS from
  the granule's grid mapping).
- Geographic anchors (`{lon, lat}`) placed at the exact projected position,
  and the emission source as anchor point (`corner: source`, the default).
- Plumes with overlapping windows form clusters that are attenuated together
  with one absorption table at their median geometry.
- `wavelength_ranges_nm` in the run file limits the attenuated bands to one
  or more disjoint ranges inside the forward model's range, e.g.
  `[[1400, 1800], [2000, 2500]]`; the output records them in
  `spectra_attenuated_wavelength_ranges_nm`. `ClusterResult.bands` holds the
  attenuated band indices and `ClusterResult.band_runs` their contiguous runs.
- libRadtran/ARTS forward model ported from `aviris4_ch4.mf.UnitAbsorption`
  (`spectra.spectral.libradtran`), bit-identical to the original on AVIRIS-4
  and EMIT band sets, with explicit resource files, shared resource arrays
  and memoised intermediates.
- `spectra.testing.native`: schema-exact synthetic AVIRIS-3, AVIRIS-4 and
  EMIT files and HDF5-level comparison; `run_adapter_conformance(adapter,
  files)`.

### Changed

- `source_<native dim>` in the output is the pixel of the emission source
  prescribed in the LES (the plume column's `source_x`/`source_y`, placed
  with the plume), whatever the anchor point; `-1` when the source lies
  outside the swath. The anchor pixel moved to `source_anchor_<native dim>`,
  and `source_anchor_point`, `source_model_x` and `source_model_y` record
  the anchor point and the LES source position.
- The forward model takes the solar zenith angle from the scene (AVIRIS-4
  `cast_shadow`, AVIRIS-3 and EMIT OBS) instead of an ephemeris, so
  AVIRIS-4 tables differ slightly from v0.1.0 where the two disagreed.
- Surface elevations below sea level are treated as 0 m in the forward model
  (the original selected the top layer as surface layer there).
- Sensor adapters take `open_scene(files)` with every file given by role and
  declare a `NativeLayout`; the AVIRIS-4 adapter reads radiance and metadata
  only, requires a scalar `time` and the `cast_shadow` solar angles, reads the
  newer chunked converter output too, and keeps radiance in native units.
- Swath outlines that cross themselves at folded edges are repaired instead of
  rejected.
- Random anchors are drawn from a conservative map of the pixels that can
  meet the outside-fraction bounds (`spectra.geometry.feasibility`) instead
  of from the whole scene. Infeasible bounds fail at once with the reachable
  fractions, pixels without geolocation are never drawn, and a given seed
  draws different pixels than before.
- Plume-column validation reduces chunked files lazily; plume files are
  NetCDF only.
- Embedding version `lambert-beer-v2`: the optical depth between absorption
  table nodes is interpolated with PCHIP instead of linearly. The band optical
  depth is convex in the path amount (lines saturate), so linear
  interpolation under-attenuated plumes between nodes. With the default 12
  log-spaced nodes from 0.01 to 200 ppm, the embedded AVIRIS-4 CH4 signal was
  up to about 3 % too weak at 0.5 to 64 ppm, with radiance errors of 1 to 3 %
  in the strongest bands; PCHIP keeps the error below 0.4 %. Tables built by
  `build_slant_absorption_lut` carry the new version, and
  `embed_column_in_radiance` interpolates as the table's `embedding_version`
  prescribes, so `lambert-beer-v1` tables (and the golden fixtures) still
  embed linearly and bit for bit as before. Unknown versions raise a
  `ValueError`. `spectra_embedding_version` in the output records the
  versions of the tables used.

### Removed

- Matched filter and retrievers, column products and product embedding,
  prism/column-product/radiance exporters and pairings, scene catalogue and
  QA, recipe-driven generation, finalize and Slurm.
- CDO and bucket regridders (the exact polygon regridder is within a factor
  of two of bucket on large plumes), map-grid regridding, scene tables,
  `SceneRecord`, the toy adapter, quality masks and background groups.
- The entry-point plugin registry, unit-absorption providers, the slant
  column convention, plume mirroring/rescaling/wind transforms, the AVIRIS-4
  file-name time fallback, and the `aviris4`, `emit`, `prism`, `slurm` and
  `microhh` extras; `aviris4_ch4`, pyarrow, zarr and pandas are no longer
  dependencies.

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
