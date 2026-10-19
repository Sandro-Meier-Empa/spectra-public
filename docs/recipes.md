# Recipe-driven dataset generation

A recipe is a YAML file that, together with a scene catalogue (or a scene
table) and a set of plume-column files, fully determines a synthetic dataset.
Everything random is derived from the recipe seed, so a dataset can be
regenerated sample by sample on any number of processes and the result is
identical.

```
spectra generate recipe.yaml --out RUN --dry-run    # context, parameters, LUT check
spectra generate recipe.yaml --out RUN              # all samples, one process
spectra slurm recipe.yaml --out RUN                 # job array; sbatch it
spectra finalize recipe.yaml --out RUN              # merge shards -> pairings.csv
spectra export-prism recipe.yaml --out RUN --compose
```

`--out RUN` is the run directory (defaults to `output.root`); products are
written to `output.root/samples/<sample_id>.nc|.zarr`. Examples:
`examples/recipe_toy.yaml` (no external data) and
`examples/recipe_aviris4.yaml`.

## Schema reference

Validation happens in `spectra.recipe.schema.load_recipe`; every error is a
`ContractError` naming the offending key. Relative paths (`sensor.scenes`,
`catalog.path`, `output.root`, `plumes.files`) are resolved against the
recipe file. Defaults are shown; keys without a default are required.

| Key | Default | Meaning |
|---|---|---|
| `recipe_version` | `1` | Schema version. |
| `name` | required | Dataset name; sample ids are `<name>-<index:06d>`. |
| `seed` | required | Master seed of every draw. |
| `n_samples` | required | Number of samples (>= 1). |
| `method` | `radiance` | `radiance` (Lambert-Beer into the cube, then retrieval) or `product` (additive onto the adapter's retrieved product). `product` forbids `retrieval`. |
| `sensor.name` | required | Registered sensor adapter (`spectra plugins`). |
| `sensor.scenes` | `null` | Scene table (see below) passed to `list_scenes`; required when `catalog.path` is null. |
| `sensor.options` | `{}` | Adapter constructor options. |
| `catalog.path` | `null` | Catalogue directory (`spectra catalog build`); null lists scenes with the adapter. |
| `catalog.status` | `[eligible]` | Accepted catalogue statuses. |
| `catalog.filters` | `{}` | `SceneCatalog.filter` criteria (or `list_scenes` filters: `scene_id` and equality on extra scene-table columns). |
| `plumes.files` | required | Plume-column files (Contract A); globs allowed. |
| `plumes.time_policy.mode` | `uniform` | `uniform` over snapshots at or after `min_simulation_time_s`; `fixed` (one index in `indices`); `list` (uniform over `indices`). |
| `plumes.time_policy.min_simulation_time_s` | `0.0` | Spin-up excluded by `uniform`. |
| `plumes.time_policy.indices` | `null` | Snapshot indices of `fixed`/`list`. |
| `plumes.emission.distribution` | `loguniform` | `fixed`, `uniform`, `loguniform` (`low`..`high`) or `list` (`values`). |
| `plumes.emission.units` | `kg h-1` | `kg h-1` or `kg s-1`; converted to kg s-1 internally. |
| `plumes.column_convention` | `vertical` | Convention of the injected column (`vertical` or `slant`). |
| `placement.mode` | `anchor` | Only anchor placement is supported. |
| `placement.rotations_deg` | `{mode: uniform_int, low: 0, high: 359}` | Whole-degree rotation: `uniform_int`, `fixed` (`value`) or `list` (`values`). |
| `placement.flips` | `{none: 1.0}` | Weights of the reflection axes `none`, `x`, `y`. |
| `placement.anchor_corners` | all four | LES corner cells drawn uniformly. |
| `placement.min_outside_fraction`, `max_outside_fraction` | `0.0`, `0.0` | Accepted plume mass fraction outside the swath (`0, 0` = fully inside). |
| `placement.max_attempts` | `10000` | Anchor draws before giving up. |
| `placement.window_shape` | `null` | Minimum `[n_across, n_along]` of the sample window. |
| `embedding.regridder`, `regridder_options` | `polygon`, `{}` | Conservative regridder plugin. |
| `embedding.lut` | `{0.01, 200.0, 12, 1000.0, both}` | `enhancement_min_ppm`, `enhancement_max_ppm`, `n_nodes` (log-spaced), `layer_height_m`, `radiance_leg`. |
| `embedding.unit_absorption` | `null` | `{provider, options}`; required for `radiance`. |
| `retrieval` | `null` | `{retriever, options, statistics_scope: window, target_enhancement_ppm: 10, band_windows_nm, min_n_samples: 7000, exclude_injected_from_background: false, plume_mask_threshold_kg_m2: 0}`; required for `radiance` unless the exporter writes embedded radiance (then it must be omitted). With `exclude_injected_from_background`, pixels whose injected column exceeds `plume_mask_threshold_kg_m2` are excluded from the background statistics. |
| `negatives.fraction` | `0.0` | Probability that a sample carries no plume. |
| `labels.detection_snr`, `ambiguous_snr` | `2.0`, `1.0` | prism label thresholds (recorded on products and passed to `prism-compose-dataset`). |
| `split.ratios` | `{train: 0.7, val: 0.15, test: 0.15}` | Group and sample fractions; must sum to 1. |
| `split.group_by` | `[background_group, les_case]` | Group columns assigned independently. |
| `split.seed` | `null` | Seed of the group assignment (defaults to `seed`). |
| `output.root` | required | Dataset root; samples go to `<root>/samples`. |
| `output.product_format` | `netcdf` | `netcdf` or `zarr`. |
| `output.dataset_version` | required | Immutable version label for prism. |
| `output.exporter`, `exporter_options` | `prism`, `{}` | Registered exporter (`spectra.exporters`): `prism` (prism `(y, x)` sample contract), `column_product` (retrieved `ColumnProduct` in sensor geometry, for other ML frameworks), `radiance` (embedded `SensorScene`, no retrieval) or a third-party one. |
| `output.extent` | `window` | `window` writes the sample window; `scene` the full scene with the window embedded (radiance method). |
| `compute.chunks` | `{}` | Dask chunks passed to `open_scene`. |
| `compute.slurm` | `null` | `{account, run_time, array_size, tasks_per_job: 1, uenv, partition, cluster: eiger, processor: cpu}`. |

`recipe_hash` is the SHA-256 of the canonical JSON of the fully resolved
recipe (`recipe.resolved.yaml` reloads to the same hash). It is written to
every product and pairings row.

## Determinism rules

1. **Context** (`context.json`): the eligible scenes (catalogue rows with an
   accepted status, or `adapter.list_scenes`), the resolved plume files with
   their per-snapshot emission rates and column maxima, and the group-to-split
   assignment. `context_hash` covers all of it.
2. **Splits are assigned to groups before pairing.** `background_group`s and
   `les_case`s are each ordered by `sha256("<seed>:<group>")`, cut by the
   cumulative ratios with largest-remainder rounding, and every split with a
   positive ratio receives at least one group when enough groups exist (the
   rule of `prism.data.splits.assign_groups`, made process independent).
   Samples only pair scenes and plumes of the same split, so prism's
   leakage rule holds by construction. The dry run fails when a split would
   have no scene or no plume file.
3. **Per-sample draws** use
   `numpy.random.default_rng(SeedSequence(seed, spawn_key=(index,)))` in the
   fixed order split -> negative? -> scene -> plume file -> time index ->
   emission -> rotation -> flip -> anchor corner -> anchor pixel
   (`place_in_scene`, uniform over the swath until the outside-mass bounds
   hold). Changing `n_samples` never changes existing samples; changing the
   seed changes all of them.
4. **Negatives** draw everything like a positive (identical stream) and then
   drop the plume: `emission_scale = 0`, `emission_rate_kg_s = 0`,
   `source_id = none`, `tracer = none`, `les_case = none:<sample_id>` (one
   group per negative, as prism does for real observations, so the
   group-independence rule holds). The window and `plume_uri` of the drawn
   plume are kept so the sample has the same window statistics as a positive.
   The product keeps the recipe's `embedding_method` (prism rejects `none`
   for composed samples) with a zero `injected_ch4_mass`.
5. **Sample `i` reads only row `i`** of `params.parquet` (`SampleParams`:
   scene, plume, time index, emission rate and scale, placement with anchor
   indices, window, `pair_id`, `params_hash =
   sha256(recipe_hash:context_hash:index)`). `pair_id` hashes the physical
   draw without the method, so a `product` twin of a `radiance` recipe with the
   same seed shares pair ids.
6. The dry run also writes `lut_coverage.json`: the largest column of every
   plume file at the largest emission rate, in ppm-m, against
   `enhancement_max_ppm * layer_height_m`. A radiance recipe that cannot
   cover its plumes fails with `extend the LUT nodes ...`. The effective
   layer height of a scene is at most the nominal one; the embedding
   re-checks the coverage per sample.

## Outputs

| File | Content |
|---|---|
| `RUN/context.json`, `RUN/params.parquet` | Context and parameter table of the dry run. |
| `RUN/recipe.resolved.yaml`, `RUN/recipe_hash.txt`, `RUN/lut_coverage.json` | Resolved recipe, its hash, LUT check. |
| `output.root/samples/<sample_id>.nc|.zarr` | Samples written by `output.exporter` (prism products by default) with the provenance attributes `sensor, scene_id, background_group, les_model, les_case, tracer, source_id, time_index, simulation_time_s, emission_rate_kg_s, emission_scale, rotation_deg, flip, anchor_*, window_*, pixel_size_m, injected_source_y/x, recipe_name, recipe_hash, params_hash, pair_id, spectra_version, created_utc, labels_*` plus the `embedding_*`/`regrid_*` attributes of the pipeline; `wind_u`/`wind_v` channels are the plume wind transformed with the placement. |
| `RUN/shards/pairings_<shard|all>.csv` | Pairings rows of one `generate` call. |
| `RUN/pairings.csv`, `RUN/dataset.json` | Merged table (`finalize`: completeness and group-independence checks) and summary. |

The pairings columns (`spectra.export.pairings.PAIRINGS_COLUMNS`) start with
the prism requirements `sample_id, product_uri, background_group, les_case,
source_id, split, sensor`, then `source_y, source_x` (the placed LES source
mapped to the nearest window pixel in prism's `(y, x)` layout, or scene pixel
with `output.extent: scene`; empty for negatives) and the provenance columns `scene_id, tracer, time_index,
simulation_time_s, emission_rate_kg_h, emission_scale, rotation_deg, flip,
anchor_corner, anchor_along, anchor_across, window_across_start,
window_along_start, embedding_method, retriever, unit_absorption_provider,
regridder, mf_statistics_scope, uncertainty_source, column_convention,
pixel_size_m, pair_id, plume_uri, scene_uri, recipe_hash, params_hash,
spectra_version`.

`spectra export-prism` needs `output.exporter: prism`; the pairings table is
written for every exporter and serves as the sample index of other
consumers.

## Slurm usage (CSCS Alps)

`spectra slurm recipe.yaml --out RUN` writes `RUN/generate.sbatch` with
`cscs.AlpsSetup.create_sbatch_script` (install the `slurm` extra). The array
has `compute.slurm.array_size` jobs; job `j` runs `tasks_per_job` shards
`k = j * tasks_per_job + t` concurrently on its exclusive node, and shard `k`
generates the index range `[k * per_shard, (k + 1) * per_shard)` with
`python -m spectra.cli generate RUN/recipe.resolved.yaml --out RUN
--index-range START:STOP --shard k`. Run the dry run before submitting (the
jobs re-run it otherwise, concurrently), submit with `sbatch`, then
`spectra finalize` merges `RUN/shards/pairings_*.csv`. Re-running a shard
with `--overwrite` regenerates identical products; nothing is submitted by
SPECTRA itself.

## Scene tables

SPECTRA never derives file locations from directory layouts or file names:
the user lists every file of every scene explicitly, and adapters only open
what they are given. A scene table is a CSV file (one row per scene), a YAML
list of mappings (optionally under a `scenes` key) or, in Python, a list of
dicts (`spectra.sensors.table.read_scene_table`). Its columns are the roles
the adapter declares (`Aviris4Adapter.ROLES`: `radiance` and `metadata`
required; `shadow`, `cloud`, `product` optional), plus

- `scene_id` (optional; defaults to the stem of the first required file),
- `background_group` (optional; the leakage group, default
  `<sensor>:<scene_id>`),
- any other column, kept as string metadata in `SceneRecord.extra` and usable
  as an equality filter (`--filter altitude=3000ft`, `catalog.filters`).

Empty cells omit optional roles; relative paths resolve against the table's
directory; required roles must be present and every listed file must exist;
scene ids must be unique. `examples/scenes_aviris4.example.csv`:

```
scene_id,radiance,metadata,shadow,cloud,product,background_group
line_0001,/path/to/L1/line_0001_rdn.nc,/path/to/L1/line_0001_rdn_metadata.nc,/path/to/L2/line_0001_rdn_shadow.nc,,/path/to/L2/line_0001_rdn_CH4.nc,aviris4:campaign-day1
```

The same table feeds `spectra catalog build --scenes`, `sensor.scenes` of a
recipe without a catalogue, and `adapter.list_scenes(table, **filters)`. A
single scene is given on the command line (`spectra embed ... --radiance
PATH --metadata PATH [--shadow PATH] [--cloud PATH] [--product PATH]
[--file ROLE=PATH]`) or in Python with
`adapter.record_from_files(radiance, metadata, shadow=..., ...)`.

## Catalogue commands

```
spectra catalog build --sensor aviris4 --scenes SCENES.csv --out CATALOG [--filter k=v] [--no-inspect]
spectra catalog qa CATALOG [--scene-id ID ...] [--subsample N] [--qa-version V]
spectra catalog status CATALOG --scene-id ID --set eligible --reason TEXT --by NAME
```

`build` catalogues every scene of the table as `discovered` (`--no-inspect`
skips opening the scenes for their sizes); `qa` computes
`spectra.catalog.qa.compute_scene_qa` per scene and applies the default
`QaThresholds`; `status` moves single scenes along the lifecycle.


Snapshots whose emission rate is zero (before a ramped release starts) are never drawn in `uniform` mode; `fixed`/`list` indices pointing at such snapshots raise an error.
