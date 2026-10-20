# SPECTRA

**S**ynthetic **P**lume **E**mbedding in **C**ubes for **T**race-gas
**R**emote-sensing **A**pplications.

`spectra` embeds simulated trace-gas plumes from large-eddy simulations into
real imaging-spectrometer radiance and returns the attenuated radiance **in
the input's native file format**, with the embedding metadata attached. It
supports AVIRIS-3, AVIRIS-4 and EMIT. It does not retrieve or prepare data
for any downstream
application: feed the output into your own retrieval, detection or training
pipeline as you would the original file.

## What it does

1. **Import** (`spectra.sensors`): an adapter per sensor opens the native
   files of one scene, given explicitly by role, as a common `SensorScene`
   (sensor geometry, native radiance units).
2. **Place and regrid** (`spectra.geometry`, `spectra.regrid`): each LES plume
   snapshot is anchored at a geographic position, a pixel or a seeded random
   pixel (by its emission source or a grid corner), reflected and rotated,
   and regridded onto the pixel grid with exact, mass-conserving polygon
   overlap.
3. **Embed** (`spectra.pipeline`, `spectra.spectral`): plumes with
   overlapping windows form clusters; each cluster gets one absorption table
   from the libRadtran/ARTS forward model at its geometry, and the radiance
   of its pixels is attenuated (Lambert-Beer) in every band the model covers.
4. **Export** (`spectra.export.native`): the output is a byte copy of the
   native radiance file in which only the plume pixels change, plus a
   `source` dimension with the injected column of every plume
   (`injected_<gas>_mass`), per-plume provenance (`source_*`) and run
   provenance (`spectra_*`, including the resolved run file that reproduces
   the output).

## Installation

```bash
python -m pip install -e ".[dev]"
```

On CSCS Alps prefix Python commands with `OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1`.

## Quick start

Describe one scene and its plumes in a run file
([`docs/run-file.md`](docs/run-file.md), examples in `examples/`):

```yaml
sensor: emit
files:
  radiance: /path/to/EMIT_L1B_RAD_001_20260819T180247_2623112_032.nc
  obs: /path/to/EMIT_L1B_OBS_001_20260819T180247_2623112_032.nc
output: /path/to/EMIT_L1B_RAD_001_20260819T180247_2623112_032_embedded.nc
seed: 2026
forward_model:
  solar_reference: /path/to/spectrum/hybrid_reference_spectrum_p005nm_resolution_c2021-03-04_with_unc.nc
  tau:
    all: /path/to/spectrum/UVSPEC_moltau_midlatitude-summer_all_1000-2500nm-0.001nm.nc
    CH4: /path/to/spectrum/UVSPEC_moltau_midlatitude-summer_CH4_1000-2500nm-0.001nm.nc
  profiles:
    z: /path/to/arts/midlatitude-summer/midlatitude-summer.z.xml
    t: /path/to/arts/midlatitude-summer/midlatitude-summer.t.xml
    CH4: /path/to/arts/midlatitude-summer/midlatitude-summer.CH4.xml
  backgrounds: {CH4: 1.93}
plumes:
  - {file: /path/to/plume.plumecolumn.nc, time_index: 120, anchor: {lon: -102.9, lat: 32.1}, rotation: 30}
  - {file: /path/to/plume.plumecolumn.nc, time_index: 60, anchor: random}
```

```bash
spectra sensors                       # sensors and their file roles
spectra embed run.yaml --dry-run      # resolve anchors, print the resolved run
spectra embed run.yaml                # write the embedded native file
spectra validate plume.nc --contract plume-column
```

In Python:

```python
import spectra
import xarray as xr

outcome = spectra.embed("run.yaml")
embedded = xr.open_dataset(outcome.output)
truth = embedded["injected_ch4_mass"].sum("source")   # kg m-2, native dims
```

`spectra.embed_scene(scene, plumes, forward_model, seed=...)` returns the
in-memory result; `result.to_dataset(scene)` gives the embedded scene without
writing a file. The notebooks in `notebooks/` walk through AVIRIS-3, AVIRIS-4
and EMIT.

## Documentation

- [`docs/run-file.md`](docs/run-file.md): run files, anchors, the forward
  model, reproducibility.
- [`docs/sensors.md`](docs/sensors.md): file roles and the native-to-common
  mapping of each sensor, adding a sensor, large files.
- [`docs/contracts.md`](docs/contracts.md): plume columns, sensor scenes, the
  embedded output schema, absorption tables.

## Testing

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest
```

`tests/golden/` holds frozen reference outputs: the `aviris4_ch4` code the
embedding and the forward model were ported from (byte-identical), and the
v0.1.0 single-plume pipeline. Synthetic native files with the exact AVIRIS-3,
AVIRIS-4 and EMIT schemas (`spectra.testing.native`) back the round-trip
tests.
Optional environment variables enable tests on real resources:
`SPECTRA_TEST_RESOURCES` (libRadtran/ARTS bundle root),
`SPECTRA_TEST_<SENSOR>_FILES` (a YAML `{files: {role: /path/to/file}}` of one
real scene), `SPECTRA_TEST_DATA` (regridding comparison files),
`SPECTRA_TEST_MICROHH_CASE` (a MicroHH case directory).

See `AGENTS.md` for repository conventions.
