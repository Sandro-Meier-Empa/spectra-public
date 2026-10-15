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
| `inputs/lut.nc`, `inputs/target_10ppm.nc` | Real absorption table and 10 ppm target spectrum |
| `inputs/column_on_grid.nc` | Synthetic Gaussian column on the radiance grid (kg m-2) |
| `inputs/observed_product.nc` | Synthetic matched-filter column in ppm-m |
| `inputs/fake_remapped.nc` | Synthetic remapped window with embedding attributes |
| `expected/embedded_radiance_{vertical,slant}.nc` | `embed_column_in_radiance` outputs |
| `expected/injected_ch4_mass.nc` | `injected_column` output |
| `expected/product_embedded_{ppm,kg}.nc` | `embed_column_in_product` outputs |
| `expected/placement_{rot,flip,partial}_cdo_{source,target}.nc` | `prepare_microhh_for_aviris_cdo` outputs |
| `expected/merged.nc` | `merge_aviris_with_microhh` output without radiance |
| `record.json` | Inputs, placement results, provenance attributes and SHA-256 digests |

## Placements

| Name | Anchor | Rotation | Flip | along, across | Outside fraction |
|---|---|---|---|---|---|
| rot | upper left | -20 | None | 15, 2 | 0 |
| flip | lower right | 10 | x | 20, 26 | 0 |
| partial | lower left | 0 | None | 148, 36 | 7.54e-09 |

The real-data files are small crops of campaign data stored for regression
testing only; they are not a data product.
