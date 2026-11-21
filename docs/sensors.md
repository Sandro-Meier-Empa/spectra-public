# Sensors

Each sensor adapter opens a scene's native files, given explicitly by role,
as a `SensorScene` (see [`contracts.md`](contracts.md)) and declares in its
`LAYOUT` where the radiance lives in the native file. The exporter writes the
attenuated pixels back through the same layout, so import and export share
one dimension mapping. Adapters never search directories or parse file names.

```
spectra sensors
```

lists the built-in sensors and their file roles.

## AVIRIS-4 (`aviris4`)

L1 radiance and PARGE metadata NetCDF files in sensor geometry. Both the
files of the 2024/2025 campaigns (contiguous storage, int64 index
coordinates, `days since` time) and the newer converter output (chunked
radiance, int32 coordinates, epoch-seconds time, pixel sizes) are read.

| Role | Required | File |
|---|---|---|
| `radiance` | yes | `<scene>_rdn.nc` |
| `metadata` | yes | `<scene>_rdn_metadata.nc` |

| `SensorScene` | Native source |
|---|---|
| `radiance(across, along, band)` | `radiance(across, along, band)`, NaN fill, W m-2 nm-1 sr-1 |
| `wavelength`, `fwhm` | data variables of the radiance file |
| `time` | scalar `time` of the radiance file (required; must equal the metadata file's) |
| `lon`, `lat`, `x`, `y`, CRS | metadata `lon`, `lat`, `x`, `y`; CRS from the `epsg` attributes of `x`/`y` |
| `sza`, `saa` | `zenith` and `azimuth` attributes of the metadata `cast_shadow` (required) |
| `vza` | `abs(scan_zenith) / 100` |
| `vaa` | `absolute_azimuth / 10` |
| `surface_elevation` | `elevation` |
| `sensor_altitude` | `aircraft_altitude` |
| `pixel_area` | `pixel_size_across_track * pixel_size_along_track` when present, otherwise from the centre spacing |
| `valid` | finite radiance (middle band) and finite `x` |

The output is a copy of `<scene>_rdn.nc`; the metadata file applies to it
unchanged and is not copied.

## EMIT (`emit`)

EMIT L1B granules (collection 001) in sensor geometry. Both the older builds
(netCDF 4.7.4, HDF5 superblock v0, GLT fill -9999) and the newer ones (netCDF
4.9.3, superblock v2, GLT fill 0) are read and patched.

| Role | Required | File |
|---|---|---|
| `radiance` | yes | `EMIT_L1B_RAD_001_<time>_<orbit>_<scene>.nc` |
| `obs` | yes | the matching `EMIT_L1B_OBS_001_...nc` |

| `SensorScene` | Native source |
|---|---|
| `radiance(across, along, band)` | `radiance(downtrack, crosstrack, bands)`: downtrack is `along`, crosstrack `across`; -9999 fill, uW cm-2 sr-1 nm-1 (native units are kept) |
| `wavelength`, `fwhm` | `sensor_band_parameters/wavelengths`, `sensor_band_parameters/fwhm` (they differ slightly between granules) |
| `time` | global attribute `time_coverage_start` |
| `lon`, `lat` | `location/lon`, `location/lat` |
| `x`, `y`, CRS | `lon`/`lat` projected into the WGS 84 UTM zone of the granule centre |
| `sza`, `saa`, `vza`, `vaa` | OBS bands 4, 3, 2 and 1 (checked by name) |
| `surface_elevation` | `location/elev` |
| `sensor_altitude` | OBS path length (band 0) x cos(vza) + elevation (about 420 km) |
| `pixel_area` | from the centre spacing |
| `valid` | radiance not fill (middle band) and finite geolocation |

The RAD and OBS granules must share `flight_line` and `time_coverage_start`.
The output is a copy of the RAD granule (radiance, `flat_field_update`,
`sensor_band_parameters`, `location` with the GLT and all global attributes
unchanged); the OBS granule applies to it unchanged.

## AVIRIS-3 (`aviris3`)

AVIRIS-3 L1B granules of the ORNL DAAC (NetCDF4, one scene of a flight line
per granule) in sensor geometry: the radiance and the observation parameters
are on the raw `(lines, samples)` grid; the geolocation lookup table (GLT)
that orthorectifies them is not used.

| Role | Required | File |
|---|---|---|
| `radiance` | yes | `AV3<flight line>_<scene>_L1B_RDN_<hash>_RDN.nc` |
| `obs` | yes | the matching `AV3<flight line>_<scene>_L1B_ORT_<hash>_OBS.nc` |

| `SensorScene` | Native source |
|---|---|
| `radiance(across, along, band)` | `radiance/radiance(wavelength, lines, samples)` (band-major, chunked): lines are `along`, samples `across`; -9999 fill, uW nm-1 cm-2 sr-1 (native units are kept) |
| `wavelength`, `fwhm` | `radiance/wavelength`, `radiance/fwhm` [nm] |
| `time` | global attribute `time_coverage_start` |
| `lon`, `lat` | root `lon`, `lat` (float32) |
| `x`, `y`, CRS | `lon`/`lat` projected into the granule's own UTM zone, the `crs_wkt` of the root grid mapping (`transverse_mercator`) |
| `sza`, `saa`, `vza`, `vaa` | OBS `observation_parameters/to_sun_zenith`, `to_sun_azimuth`, `to_sensor_zenith`, `to_sensor_azimuth` |
| `surface_elevation` | root `elev` |
| `sensor_altitude` | OBS `path_length` x cos(vza) + elevation |
| `pixel_area` | from the centre spacing |
| `valid` | radiance not fill (middle band) and finite geolocation |

The RDN and OBS granules must share `flight_line`, `scene` and
`time_coverage_start`. The geolocation and observation geometry are read
when the scene is opened; only the radiance stays lazy. The longitudes and
latitudes are float32, which limits the pixel centres to a few decimetres
(several percent of a 4 m pixel); pixel areas from the centre spacing are
correspondingly noisy, and they also follow the real along-track jitter of
the aircraft. The output is a copy of the RDN granule (the radiance group, root
geolocation, grid mapping, GLT and all global attributes unchanged); the
embedding metadata goes to the root group on the `lines` and `samples`
dimensions (`injected_<gas>_mass(source, lines, samples)`, `source_lines`,
`source_samples`, `source_anchor_lines`, `source_anchor_samples`). The OBS
granule applies to it unchanged.

## Adding a sensor

An adapter is any object with `name`, `platform`, `ROLES`, `LAYOUT` and
`open_scene(files, *, chunks=None)` (`spectra.protocols.SensorAdapter`). Pass
it as `sensor` to `spectra.embed` (the run-file `sensor` key names only the
built-in adapters). The native file must be NETCDF4 with a floating-point
radiance variable on three dimensions; `run_adapter_conformance(adapter,
files)` checks that the opened radiance equals a direct read of the file
through `LAYOUT`.

## Large files

- Outputs are byte copies (AVIRIS-4 lines reach several hundred GB); the
  copy goes to a hidden staging file next to the output and is moved into
  place when complete. On Lustre, a striped output directory speeds up large copies
  (for example `lfs setstripe -c 4 /path/to/out`).
- On NFSv3 file systems HDF5 file locking can fail when the copy is opened
  for writing; set `HDF5_USE_FILE_LOCKING=FALSE` in that case.
- Only the plume windows of the radiance are read and written; the scene
  geometry is read lazily.
