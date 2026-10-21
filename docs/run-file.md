# Run files

A run embeds N plumes into one scene and writes one file in the scene's
native format. It is described by a YAML file (or the same structure as a
Python mapping) and run with

```bash
spectra embed run.yaml                      # write the output
spectra embed run.yaml --dry-run            # print the resolved run, write nothing
spectra embed run.yaml --output /path/to/other.nc --seed 3 --overwrite
```

or `spectra.embed("run.yaml")` / `spectra.embed(mapping)`. Relative paths are
resolved against the run file's directory (the working directory for a
mapping). Every file is named explicitly; nothing is discovered.

```yaml
sensor: emit                         # aviris3 | aviris4 | emit
files:                               # every file of the scene by role (spectra sensors)
  radiance: /path/to/EMIT_L1B_RAD_001_20260819T180247_2623112_032.nc
  obs: /path/to/EMIT_L1B_OBS_001_20260819T180247_2623112_032.nc
output: /path/to/EMIT_L1B_RAD_001_20260819T180247_2623112_032_embedded.nc
overwrite: false                     # refuse an existing output (default)
seed: 2026                           # random anchors; omitted -> drawn and recorded
forward_model:
  solar_reference: /path/to/spectrum/hybrid_reference_spectrum_p005nm_resolution_c2021-03-04_with_unc.nc
  tau:                               # libRadtran molecular optical depths
    all: /path/to/spectrum/UVSPEC_moltau_midlatitude-summer_all_1000-2500nm-0.001nm.nc
    CH4: /path/to/spectrum/UVSPEC_moltau_midlatitude-summer_CH4_1000-2500nm-0.001nm.nc
  profiles:                          # ARTS standard-atmosphere profiles
    z: /path/to/arts/midlatitude-summer/midlatitude-summer.z.xml
    t: /path/to/arts/midlatitude-summer/midlatitude-summer.t.xml
    CH4: /path/to/arts/midlatitude-summer/midlatitude-summer.CH4.xml
  backgrounds: {CH4: 1.93}           # ppm; required for the plume gas
  scenario: midlatitude-summer       # label recorded as provenance
  # rho: 1.0                         # albedo of the radiance model (cancels)
  # wmin / wmax: band range in nm    # default: coverage of the 'all' tau file
  # enhancement_min_ppm: 0.01        # absorption-table nodes
  # enhancement_max_ppm: 200
  # n_enhancements: 12
  # layer_height_m: 1000
plumes:
  - file: /path/to/plume_a.plumecolumn.nc
    time_index: 120                  # LES snapshot (required)
    anchor: {lon: -102.9, lat: 32.1} # emission source placed exactly here
    rotation: 30                     # degrees, counterclockwise
    emission_scale: 2.0
  - file: /path/to/plume_b.plumecolumn.nc
    time_index: 90
    anchor: {along: 812, across: 233, corner: lower left}
    flip: x                          # none | x | y, before the rotation
  - file: /path/to/plume_b.plumecolumn.nc
    time_index: 60
    anchor: {random: true, max_outside_fraction: 0.05}
  - file: /path/to/plume_b.plumecolumn.nc
    time_index: 61
    anchor: random                   # short form, fully inside the swath
```

## Plumes and anchors

Each plume is a snapshot (`time_index`) of a `PlumeColumn` file. Its anchor
point is pinned to one of:

| Anchor | Meaning |
|---|---|
| `{lon, lat}` | a geographic position, projected exactly (not snapped to a pixel) |
| `{along, across}` | the centre of a pixel of the scene |
| `random` / `{random: true}` | a uniformly random pixel among those where the plume mass fraction outside the swath lies in `[min_outside_fraction, max_outside_fraction]` (default `[0, 0]`, fully inside); up to `max_attempts` candidate pixels are tested |

A random anchor is drawn from a map of the pixels that can meet the
outside-fraction bounds for the plume's rotation and reflection. The map is
built from the swath outline and one read of the pixel centres. If no pixel
can meet the bounds, the run stops at once and reports the outside fractions
the placement can reach, rather than drawing in vain.

`corner` selects the point of the LES grid that is pinned: `source` (the
emission source, `source_x`/`source_y` of the plume file; default) or a grid
corner cell (`lower left`, `lower right`, `upper left`, `upper right`).
Reflection (`flip`) and rotation pivot about the anchor.

Plume `i` draws its random anchor from the `i`-th child of the run seed, so
adding or removing other plumes does not change it. All plumes of a run must
carry the same gas.

## What happens

1. The adapter opens the scene lazily (`SensorScene`).
2. Every plume is anchored, placed (reflection, rotation) and regridded onto
   the sensor grid with exact polygon overlap, which conserves mass; its
   column is converted to kg m-2.
3. Plumes whose windows overlap form a cluster: their columns are summed
   and attenuated together, so overlapping plumes saturate correctly.
4. Each cluster gets one absorption table from the forward model, at the
   median geometry (SZA, VZA, surface elevation, sensor altitude) of its
   window, for the scene bands inside the forward model's range; other bands
   pass through unchanged.
5. The output is a byte copy of the native radiance file with the plume
   pixels attenuated (Lambert-Beer) and the embedding metadata added
   ([`contracts.md`](contracts.md#embedded-output-spectra---user)).

The surface is treated as lying in the lowest 1 km model layer: the plume
is spread over the part of that layer above the surface
(`layer_height_m - elevation mod layer_height_m`), so the absorption table
covers up to `enhancement_max_ppm` x that height in ppm-m. A plume beyond it
stops the run before the forward model runs, naming the
`enhancement_max_ppm` needed. Elevations below sea level are treated as
0 m. For airborne sensors the whole column attenuates both photon legs; if
the plume file has partial columns, the share of the mass above the sensor is
logged.

## Reproducibility

The output's `spectra_run_config` attribute holds the resolved run: the seed
and every drawn random anchor (as `random: true` with the drawn `along`/
`across`, which replays as a pixel anchor). Running it again reproduces the
output; `spectra embed run.yaml --dry-run` prints it without running the
forward model.

## Forward-model resources

The libRadtran/ARTS forward model needs, by role: the TSIS-1 solar reference
spectrum, the libRadtran optical depths of all gases (`tau.all`) and of each
gas with a background (`tau.<GAS>`), and the ARTS profiles of altitude
(`profiles.z`), temperature (`profiles.t`) and each gas with a background
(`profiles.<GAS>`). Backgrounds can be given for CH4, CO2 and H2O. The model
runs once per cluster (about a minute and a half for 200 bands); its arrays
are loaded once per run.

For tests and examples without resources, `forward_model: {model: toy}`
selects an analytic model (`spectra.spectral.toy.ToyForwardModel`).
