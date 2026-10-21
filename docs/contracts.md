# SPECTRA data contracts

Every hand-off is an explicit xarray contract with a validator in
`spectra.contracts`; contract violations raise `ContractError`. Dimension
names are part of the contracts.

## `PlumeColumn` (LES chain -> SPECTRA)

One file = one LES case x one tracer x one source. Producer:
`microhh_processing_chain.export.export_plume_column`, walked through in
`notebooks/microhh_plume_column.ipynb`. Plume files are NetCDF.

| Item | Spec |
|---|---|
| dims | `time`, `y`, `x` |
| coords | `x`, `y`: 1-D uniform cell centres in m (model frame); `time`: datetime64 UTC; `simulation_time(time)`: s since LES start |
| `column(time, y, x)` | float `kg m-2`, finite, >= 0, zeros outside the plume |
| `emission_rate(time)` | `kg s-1` |
| optional | `column_below_altitude(time, altitude, y, x)`, `concentration(time, z, y, x)` `kg m-3`, `wind_u`/`wind_v` [`m s-1`] |
| attrs | `gas`, `les_model`, `case_id`, `tracer`, `source_id`, `source_x/y/z` [m, model frame], `emission_rate_units="kg s-1"`, `dx`, `dy`, `reference_time`, `column_convention="vertical"` |

Helpers: `validate_plume_column` (lazy for chunked files), `select_time`,
`scale_emission`, `plume_support_bounds`; partial columns in
`spectra.plume.augment`. `spectra validate PLUME.nc --contract plume-column`
checks a file.

## `SensorScene` (adapter -> SPECTRA)

The common in-memory format every sensor adapter opens its native files into.
Sensor geometry, dims `(across, along, band)`.

| Item | Spec |
|---|---|
| coords | `wavelength(band)`, `fwhm(band)` nm; `lat`, `lon`, `x`, `y` on `(across, along)`; `spatial_ref` (CF grid mapping, projected metre CRS); `time` scalar or `(along)` |
| `radiance` | `(across, along, band)`, dask-backed, **native units** (a `units` attribute is required; Lambert-Beer only multiplies the radiance with a transmittance) |
| geometry | `sza`, `saa`, `vza`, `vaa` (scalar or 2-D) [degrees], `surface_elevation`, `sensor_altitude` [m ASL] |
| `valid` | bool on `(across, along)` |
| `pixel_area` | m2 on `(across, along)` |
| attrs | `sensor`, `platform` (`airborne`/`spaceborne`), `scene_id`, `acquisition_time`, `geometry_space="sensor"` |

`SceneGeometry.from_scene` extracts the projected pixel geometry used by the
placement and the regridder; `ViewingGeometry.from_scene` the scalar geometry
(SZA, VZA, surface elevation, sensor altitude) of an absorption table. Which
native variables each adapter maps onto these is described in
[`sensors.md`](sensors.md).

## Embedded output (SPECTRA -> user)

The output of a run is a **byte copy of the native radiance file** in which
only the plume pixels of the attenuated bands are rewritten. Fill values,
bands outside the forward model's range and pixels outside the plumes keep
their bytes, and every native variable, group, attribute and storage property
(chunking, compression, fill value) stays as it was. SPECTRA adds to the root
group:

| Item | Spec |
|---|---|
| dimension `source` | one entry per embedded plume (no coordinate variable) |
| `injected_<gas>_mass(source, <native y>, <native x>)` | float32 `kg m-2` on the native spatial dimensions in native order, zero outside each plume, zlib-compressed; the summed column is `.sum("source")` |
| dimension `source_strlen` | width of the string variables |
| `source_plume_file`, `source_case_id` | UTF-8 character arrays on `(source, source_strlen)` (`_Encoding`; xarray decodes them to strings) |
| `source_anchor_mode` | int8 (`0 lonlat`, `1 pixel`, `2 random`; `flag_values`/`flag_meanings`) |
| `source_time_index` | int32 LES snapshot index |
| `source_emission_rate`, `source_emission_scale` | float64 `kg s-1` (after scaling), factor |
| `source_lon`, `source_lat` | float64, emission source after placement |
| `source_<native dim>` | int32 anchor pixel, one variable per native spatial dimension (e.g. `source_downtrack`, `source_crosstrack`) |
| `source_rotation`, `source_flip` | float64 degrees (counterclockwise), int8 (`0 none`, `1 x`, `2 y`; `flag_values`/`flag_meanings`) |
| `source_injected_mass`, `source_outside_fraction` | float64 kg on the sensor grid, plume mass fraction outside the swath |
| `source_cluster` | int32 index of the plume cluster (plumes with overlapping windows) |
| `source_lut_sza`, `source_lut_vza`, `source_lut_amf`, `source_lut_surface_elevation` | geometry of the cluster's absorption table |
| `spectra_*` global attributes | version, gas, regridder, seed, source radiance path/size/mtime, attenuated wavelength range and ranges, forward-model settings (`spectra_forward_model_*`) and `spectra_run_config`, the fully resolved run file (seed and drawn anchors filled in) |

A file carrying a `source` dimension, `spectra_*` attributes or colliding
`source_*`/`injected_*_mass` variables is refused as input
(`spectra.contracts.embedded.check_not_embedded`), so plumes are never
embedded twice. The companion files (AVIRIS-4 metadata, EMIT and AVIRIS-3
OBS) are not copied; they apply to the output unchanged.

The strings are fixed-width character arrays rather than netCDF
variable-length strings: the variable-length fill value of the latter crashes
HDF5 1.14 when a file is reopened while another handle to it is open.

## Absorption table

Output of `spectra.spectral.build_slant_absorption_lut`: `unit_abs_spect`,
`sigma_slant`, `optical_depth` on `(enhancement, wavelength)` with
`vertical_amount` and `slant_amount` [ppm-m] coordinates and the geometry
attributes (`sza`, `vza`, `geometric_air_mass_factor`, `layer_height`,
`effective_layer_height`, ...). The forward model
(`spectra.spectral.libradtran.LibRadtranArts`, or `ToyForwardModel` in tests)
builds one per plume cluster from the scene's bands and the cluster geometry.
`spectra validate LUT.nc --contract lut` checks a file.

## Units

Columns are stored in `kg m-2`; `spectra.contracts.convert_column_units`
wraps `ucat` (1 ppm-m = 6.78e-7 kg m-2 at 101325 Pa and 288.15 K for CH4).
