# Golden fixture

Reference outputs of the **current `aviris4_ch4` code** on small crops of real data,
recorded by `scripts/archive/build_golden_fixture.py` (run against aviris4-ch4
commit `2b1d8c5`, the last one carrying the migrated code). SPECTRA's ports are tested against
these files for byte-identical results. Rebuild only when the reference has to
change, and record why in the CHANGELOG.

- Built (UTC): `2026-10-14T00:13:12.407563+00:00` with `aviris4_ch4 1.1.1.dev13+gd28d8d547`.
- Flight line: `M014_240916_FRA_Pau_Methane_3000ft_Line_0001_133313_000_rdn.nc` (ARES 2024 Pau campaign),
  window across `20:60`, along
  `6980:7140`, bands 2080-2500 nm every 2nd band.
- MicroHH case: `pau_controlled_release.20240918_133000_290kgh_15min_v2_release`, snapshot index 120
  (t = 6000.0 s), cells x `512:542`, y `505:517` at 1 m.
- Absorption table: libRadtran/ARTS resources in `/capstor/store/cscs/c2sm/amrs/ARES/data-unit-abs-spec`;
  sza 41.421 deg, altitude 1058 m, elevation 95 m,
  8 log-spaced nodes 0.01-200 ppm.

## Files

| File | Content |
|---|---|
| `inputs/metadata_window.nc` | PARGE metadata crop re-indexed to a standalone 40 x 160 swath |
| `inputs/radiance_window.nc` | Radiance crop, 29 SWIR bands |
| `inputs/microhh_raw.nc` | MicroHH `ch4_path` snapshot crop (12 x 30 cells, no attrs) |
| `inputs/lut.nc`, `inputs/target_10ppm.nc` | Real absorption table and 10 ppm unit absorption spectrum |
| `inputs/column_on_grid.nc` | Synthetic Gaussian column on the radiance grid (kg m-2) |
| `expected/embedded_radiance_{vertical,slant}.nc` | `embed_column_in_radiance` outputs |
| `expected/injected_ch4_mass.nc` | `injected_column` output |
| `record.json` | Inputs, placement results, provenance attributes and SHA-256 digests |

## Placements

| Name | Anchor | Rotation | Flip | along, across | Outside fraction |
|---|---|---|---|---|---|
| rot | upper left | -20 | None | 15, 2 | 0 |
| flip | lower right | 10 | x | 20, 26 | 0 |
| partial | lower left | 0 | None | 148, 36 | 7.54e-09 |

The real-data files are small crops of campaign data stored for regression
testing only; they are not a data product. Fixtures of the removed matched
filter, product embedding and CDO preparation were dropped with that code
(their digests stay in `record.json`); they are kept at tag `v0.1.0`.

## Forward-model references

`reference/` freezes the output of `aviris4_ch4.mf.UnitAbsorption` (aviris4-ch4
commit `2b1d8c5`, `mf.py` unchanged since `6d4ed8b`) for three cases, so the
libRadtran/ARTS port in `spectra.spectral.libradtran` is checked bit for bit on
two sensors' band sets. Recorded with
`scripts/golden/record_forward_model_reference.py`; `reference/record.json`
lists the geometry (including the ephemeris solar zenith angle the original
model derived), backgrounds, versions and SHA-256 digests.

| Case | Bands | Geometry | Backgrounds |
|---|---|---|---|
| `emit` | 201 EMIT bands, 1000-2500 nm | vza 8 deg, elevation 150 m, altitude 400 km | CH4, CO2, H2O |
| `aviris4` | 202 AVIRIS-4 bands, 1000-2500 nm | vza 12 deg, elevation 280 m, altitude 4280 m | CH4 |
| `aviris4_high` | as `aviris4` | vza 3 deg, elevation 1350 m (second layer), altitude 5200 m | CH4 |

Each case has `{case}_lut.nc` (absorption table, 12 nodes 0.01-200 ppm) and
`{case}_target_10ppm.nc` (unit absorption spectrum at 10 ppm with the band
`fwhm`). `reference/profiles.json` holds the ARTS profiles parsed with `lxml`
and their column averages; `inputs/arts/midlatitude-summer/` holds the ARTS
profile files themselves (2 KB each) so the parser tests run without the full
resources.
