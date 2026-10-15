# SPECTRA data contracts

Every hand-off between repositories is an explicit xarray contract with a
validator in `spectra.contracts`. Dimension names are part of the contract.

## Contract A: `PlumeColumn` (LES chain -> SPECTRA)

One file = one LES case x one tracer x one source. Producer:
`microhh_processing_chain.export.export_plume_column`.

| Item | Spec |
|---|---|
| dims | `time`, `y`, `x` |
| coords | `x`, `y`: 1-D uniform cell centres in m; `time`: datetime64 UTC; `simulation_time(time)`: s since LES start |
| `column(time, y, x)` | float32 `kg m-2`, finite, >= 0, zeros outside the plume |
| `emission_rate(time)` | `kg s-1` |
| optional | `column_below_altitude(time, altitude, y, x)`, `concentration(time, z, y, x)` `kg m-3`, `wind_u`/`wind_v` (`m s-1`, attr `height`) |
| attrs | `gas`, `les_model`, `case_id`, `tracer`, `source_id`, `source_x/y/z`, `emission_rate_units="kg s-1"`, `dx`, `dy`, `reference_time`, `column_convention="vertical"` (+ provenance) |

Helpers: `validate_plume_column`, `select_time`, `scale_emission`,
`plume_support_bounds`; augmentation in `spectra.plume.augment`.

## Contract B: `SensorScene` (adapter -> SPECTRA)

Sensor geometry, dims `(across, along, band)`.

| Item | Spec |
|---|---|
| coords | `wavelength(band)`, `fwhm(band)` nm; `lat`, `lon`, `x`, `y` on `(across, along)`; `spatial_ref` (CF grid mapping, projected metre CRS); `time` scalar or `(along)` |
| `radiance` | `(across, along, band)` `W m-2 nm-1 sr-1`, dask-backed |
| geometry | `sza`, `saa`, `vza`, `vaa` (scalar or 2-D, degrees), `surface_elevation`, `sensor_altitude` (m ASL), optional `h2o_column` |
| masks | `valid` (required), `cloud`, `shadow`, `water`, `saturation` (bool) |
| `pixel_area` | m2 |
| attrs | `sensor`, `platform` (`airborne`/`spaceborne`), `scene_id`, `acquisition_time`, `background_group`, `geometry_space="sensor"` |

`SceneGeometry.from_scene` extracts the projected pixel geometry used by the
placement and the regridders.

Adapters take explicit files and never derive locations from directory
layouts or file names. Each adapter declares its file roles with a required
flag (`Aviris4Adapter.ROLES`: `radiance`, `metadata` required; `shadow`,
`cloud`, `product` optional); the user lists the files per scene in a scene
table (`spectra.sensors.table`, see `docs/recipes.md`), on the command line
(`spectra embed --radiance ... --metadata ... --file ROLE=PATH`) or with
`record_from_files`. `list_scenes(table, **filters)` returns the
`SceneRecord`s of a table, sorted by acquisition time, with the resolved paths
in `uris` and the `background_group` from the table (default
`<sensor>:<scene_id>`).

## Contract B': `ColumnProduct` (retriever output / L2B input / embedded product)

`geometry_space="sensor"` -> dims `(across, along)`; `geometry_space="map"`
-> dims `(y, x)` with 1-D `x`, `y` coordinates.

| Variable | Spec |
|---|---|
| `ch4_mass`, `ch4_mass_uncertainty` | `kg m-2`, uncertainty > 0 where valid |
| `retrieval_valid` | bool |
| `injected_ch4_mass` | `kg m-2`, present after embedding (label source) |
| optional | `sensitivity`, `background_valid`, `pixel_area`, `albedo`, `rgb_r/g/b`, `wind_u/v` |
| attrs | `sensor`, `scene_id`, `geometry_space`, retriever and embedding provenance |

`to_prism_grid` maps sensor space to `(y, x)` positionally (`across -> y`,
`along -> x`); `spectra.export.prism.to_prism_product` finalises a product for
prism and validates it with prism's own validator when prism is installed.

## Absorption table

Output of `spectra.spectral.build_slant_absorption_lut` (identical to the
former `aviris4_ch4.embedding` table): `unit_abs_spect`, `sigma_slant`,
`optical_depth` on `(enhancement, wavelength)` with `vertical_amount` and
`slant_amount` (ppm-m) coordinates and geometry attributes. Unit-absorption
providers (`spectra.spectral.providers`) build it for any sensor from a
`SpectralResponse` and a `ViewingGeometry`.

## Units

Columns are stored in `kg m-2`; `spectra.contracts.convert_column_units` wraps
`ucat` (1 ppm-m = 6.78e-7 kg m-2 at 101325 Pa and 288.15 K) and strips the
`units` attribute before the call because `ucat` rejects labelled arrays.
