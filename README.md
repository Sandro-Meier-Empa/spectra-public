# SPECTRA

**S**ynthetic **P**lume **E**mbedding in **C**ubes for **T**race-gas
**R**emote-sensing **A**pplications.

`spectra` embeds simulated trace-gas plumes from large-eddy simulations into
real observations of imaging spectrometers and generates synthetic
segmentation datasets, independently of the sensor. It sits between

- an LES processing chain producing plume columns
  (`microhh-processing-chain`, `export_plume_column`),
- sensor-specific packages providing readers and retrievals
  (`aviris4-ch4` today; EMIT via `ddeq` and ports of `emit-ghg` next), and
- the `prism` machine-learning repository consuming the canonical sample
  contract.

## What it does

1. **Contracts** (`spectra.contracts`, `docs/contracts.md`): `PlumeColumn`
   (LES output), `SensorScene` (sensor-geometry radiance, masks, geolocation),
   `ColumnProduct` (retrieved or embedded columns in `kg m-2`) and the
   absorption look-up table. Every hand-off between repositories is validated.
2. **Placement and regridding** (`spectra.geometry`, `spectra.regrid`): anchor,
   reflect and rotate a plume in the sensor's projected CRS, draw random anchor
   pixels with a bounded outside-mass fraction, and remap conservatively onto
   the pixel grid (exact polygon overlap, bucket, or `cdo remapcon`).
3. **Embedding** (`spectra.spectral`, `spectra.pipeline`): Lambert-Beer
   attenuation of the radiance with a nonlinear absorption table followed by
   the sensor's real retrieval, or additive embedding into a column product.
4. **Plugins** (`spectra.sensors`, `spectra.retrievers`,
   `spectra.spectral.providers`, `spectra.regrid`): sensor adapters,
   retrievals, unit-absorption providers and regridders registered through
   entry points (`spectra plugins`). First party: `aviris4`, `aviris4_mf`,
   `libradtran_arts`, `polygon`/`bucket`/`cdo`, plus `toy` test doubles.
5. **Catalogue and recipes** (`spectra.catalog`, `spectra.recipe`,
   `docs/recipes.md`): plume-free background scenes with QA status, and
   seeded, reproducible dataset generation that writes prism-compatible
   products and a pairings table for `prism-compose-dataset`.

## Installation

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[aviris4]"      # AVIRIS-4 adapter and retrieval
```


## Quick start

```bash
spectra plugins
spectra validate pau.plumecolumn.nc --contract plume-column

# one sample, radiance method, explicit anchor pixel; every scene file is
# given explicitly (SPECTRA never searches directories or parses file names)
spectra embed radiance --sensor aviris4 \
  --radiance /path/to/L1/line_0001_rdn.nc \
  --metadata /path/to/L1/line_0001_rdn_metadata.nc \
  --shadow /path/to/L2/line_0001_rdn_shadow.nc \
  --plume pau_crop.plumecolumn.nc --time-index 120 \
  --anchor-along 7000 --anchor-across 50 --anchor-corner "upper left" --rotation -20 \
  --uvspec-path /path/to/data-unit-abs-spec/spectrum \
  --sol-reference-path /path/to/data-unit-abs-spec/spectrum \
  --arts-path /path/to/data-unit-abs-spec/arts --xch4-bg 1.93191 \
  --noise-model-file /path/to/emit_noise.txt --min-n-samples 7000 \
  --save-path embedded.nc --prism
# product method: add the column onto the retrieved product
spectra embed product --sensor aviris4 \
  --radiance /path/to/L1/line_0001_rdn.nc \
  --metadata /path/to/L1/line_0001_rdn_metadata.nc \
  --product /path/to/L2/line_0001_rdn_CH4.nc \
  --plume pau_crop.plumecolumn.nc --time-index 120 --seed 1 \
  --save-path embedded_product.nc --prism

# datasets: list the flight lines in a scene table (one row per scene with
# the full path of every file; examples/scenes_aviris4.example.csv)
spectra catalog build --sensor aviris4 --scenes /path/to/scenes.csv --out catalog/
spectra catalog qa catalog/
spectra generate recipe.yaml --out dataset/ --dry-run
spectra generate recipe.yaml --out dataset/
spectra finalize recipe.yaml --out dataset/
prism-compose-dataset dataset/pairings.csv $PRISM_DATA_ROOT/v1 --dataset-version v1
```

`notebooks/embed_aviris4.ipynb` walks through the same chain in Python.

## Testing

`tests/golden/` holds reference outputs of the former `aviris4_ch4` embedding
code on real data crops; the ports are checked for byte-identical results.
Optional environment variables enable tests on real resources:
`SPECTRA_TEST_RESOURCES` (libRadtran/ARTS), `SPECTRA_TEST_DATA` (CDO files),
`SPECTRA_TEST_AVIRIS4_SCENES` (a scene table CSV listing real flight lines
with full paths), `SPECTRA_TEST_MICROHH_CASE` (a MicroHH case directory).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest
```

See `AGENTS.md` for repository conventions.
