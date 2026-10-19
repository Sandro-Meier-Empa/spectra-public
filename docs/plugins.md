# Combining sensors, retrievers and exporters

SPECTRA's chain has four independent plugin slots, each an entry-point group
selected by name on the command line and in recipes:

| Slot | Group | Meets the rest of the chain through |
|---|---|---|
| Sensor adapter | `spectra.sensors` | `SensorScene` (radiance, bands, geometry, masks) |
| Unit-absorption provider | `spectra.unit_absorption` | `SpectralResponse` + `ViewingGeometry` -> spectrum / LUT |
| Retriever | `spectra.retrievers` | `SensorScene` + target spectrum -> `ColumnProduct` |
| Exporter | `spectra.exporters` | embedded sample -> files of a downstream consumer |

No slot knows which plugin fills another, so any combination works, e.g.:

- AVIRIS-4 radiance, EMIT matched filter, prism:
  `sensor: aviris4`, `retrieval.retriever: <emit retriever>`, `output.exporter: prism`.
- EMIT radiance, AVIRIS-4 matched filter, another ML framework:
  `sensor: <emit adapter>`, `retrieval.retriever: aviris4_mf`,
  `output.exporter: column_product` (or a framework-specific exporter).
- AVIRIS-3 radiance, embedding only:
  `sensor: <aviris3 adapter>`, no `retrieval`, `output.exporter: radiance`,
  `output.extent: window` or `scene`.

Only `aviris4` and `toy` adapters and the `aviris4_mf` and `toy` retrievers
ship today; EMIT and AVIRIS-3 adapters and an EMIT retriever are added as
plugins without touching the rest of SPECTRA.

## Embedded radiance follows the input

`spectra.pipeline.embed_radiance_sample` keeps every band of the scene it is
given. Bands outside the provider's `spectral_range_nm` (libradtran_arts:
`wmin`..`wmax`) pass through unattenuated; `band_windows_nm` restricts the
output bands explicitly. `extent="window"` returns the plume window,
`extent="scene"` the full scene (lazy) with the window embedded and
`injected_<gas>_mass` zero elsewhere. `spectra.export.io.write_dataset`
writes either as NetCDF or Zarr (`load=False` streams full scenes).

## Adding a retriever (e.g. emit-ghg)

A retriever implements `spectra.protocols.Retriever`:

```python
class EmitGhgMatchedFilter:
    name = "emit_ghg_mf"

    def retrieve(self, scene, target, *, settings, background_stats=None):
        subset = select_bands(scene, settings.band_windows_nm)
        output = ...  # call emit-ghg on subset["radiance"] (across, along, band)
        return matched_filter_to_column_product(
            output,
            attrs={k: scene.attrs[k] for k in ("sensor", "platform", "scene_id",
                                                "background_group")},
            geometry=scene,
            retriever=self.name,
            column_variable="<emit-ghg column variable>",
            uncertainty_variable="<emit-ghg uncertainty variable>",
        )

    def background_statistics(self, scene, *, settings):
        ...  # per-column mean/covariance for statistics_scope="full_scene"
```

registered in the wrapper package's `pyproject.toml`:

```toml
[project.entry-points."spectra.retrievers"]
emit_ghg_mf = "my_package.retriever:EmitGhgMatchedFilter"
```

Notes for emit-ghg:

- emit-ghg estimates the background per detector column over the whole scene;
  embed with `extent="scene"` (or `output.extent: scene`) and/or implement
  `background_statistics` for `statistics_scope: full_scene`.
- `target` is the provider's unit absorption spectrum on the scene's
  retrieval bands; emit-ghg may instead use its own target.
- `matched_filter_to_column_product` converts ppm-m outputs to kg m-2 and
  adds the geometry, pixel areas and validity mask.
- If emit-ghg only reads files, write the embedded scene first with
  `spectra.export.io.write_dataset(..., load=False)`.
- `spectra.testing.conformance.run_retriever_conformance` checks the plugin.

## Adding an exporter

An exporter implements `spectra.protocols.Exporter` (`name`,
`needs_retrieval`, `prepare(sample, *, gas, extra_attrs)`,
`write(dataset, path, *, overwrite)`) and registers under
`spectra.exporters`. With `needs_retrieval = False` it receives the embedded
`SensorScene` and recipes run without a retriever.
