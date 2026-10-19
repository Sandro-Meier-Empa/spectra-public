"""Free combination of sensor adapters, retrievers and exporters.

Every sensor reaches every retriever through the ``SensorScene`` contract and
every exporter through the embedded sample, so any sensor x retriever x
exporter pairing must work. Two toy sensors stand in for different
instruments: an airborne SWIR-only one and a spaceborne VSWIR one whose bands
partly lie outside the provider's spectral range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spectra import cli
from spectra.contracts import validate_column_product, validate_sensor_scene
from spectra.contracts.errors import ContractError
from spectra.export.exporters import (
    ColumnProductExporter,
    PrismExporter,
    RadianceExporter,
)
from spectra.pipeline import (
    EmbeddingSpec,
    embed_radiance_sample,
    embed_sample,
    place_and_regrid,
    scene_window,
    snapshot_column,
)
from spectra.plume.io import write_plume_column
from spectra.protocols import Exporter, Retriever
from spectra.recipe import build_plugins, generate, recipe_from_mapping
from spectra.registry import get_exporter, list_plugins
from spectra.regrid.polygon import PolygonRegridder
from spectra.retrievers.convert import matched_filter_to_column_product
from spectra.retrievers.toy import ToyRetriever
from spectra.sensors.toy import ToySensorAdapter
from spectra.spectral.providers.toy import ToyUnitAbsorption
from spectra.testing import make_plume_column
from spectra.types import Placement, RetrievalSettings

SPEC = EmbeddingSpec(enhancements_ppm=(1.0, 10.0, 100.0, 1000.0))
PROVIDER_RANGE = (1000.0, 2500.0)
SENSORS: dict[str, dict[str, Any]] = {
    "airborne_swir": {"n_across": 32, "n_along": 48},
    "spaceborne_vswir": {
        "n_across": 32,
        "n_along": 48,
        "platform": "spaceborne",
        "pixel_across_m": 60.0,
        "sensor_altitude": 400_000.0,
        "wavelengths": np.arange(400.0, 2500.0, 10.0).tolist(),
    },
}


def _scene(sensor: str) -> xr.Dataset:
    adapter = ToySensorAdapter(**SENSORS[sensor])
    return adapter.open_scene(adapter.list_scenes(None, n=1)[0])


@pytest.fixture(scope="module")
def column() -> xr.DataArray:
    return snapshot_column(
        make_plume_column(nx=24, ny=24, dx=2.0, n_time=1, seed=1), time_index=0
    )


def _provider() -> ToyUnitAbsorption:
    return ToyUnitAbsorption(spectral_range_nm=PROVIDER_RANGE)


def _injected(scene: xr.Dataset, column: xr.DataArray) -> xr.DataArray:
    return place_and_regrid(
        scene,
        column,
        Placement(anchor_along=10, anchor_across=10),
        regridder=PolygonRegridder(),
    )


def _retrievers(golden: Path | None) -> dict[str, Retriever]:
    retrievers: dict[str, Retriever] = {"toy": ToyRetriever()}
    if golden is not None:
        from spectra.retrievers.aviris4_mf import Aviris4MatchedFilter

        retrievers["aviris4_mf"] = Aviris4MatchedFilter(
            noise_model_file=str(golden / "emit_noise.txt"),
            mask_shadows=False,
            parallel=False,
        )
    return retrievers


@pytest.fixture(scope="module")
def noise_inputs() -> Path | None:
    try:
        import aviris4_ch4.mf  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError:
        return None
    inputs = Path(__file__).parent / "golden" / "inputs"
    return inputs if (inputs / "emit_noise.txt").exists() else None


def test_exporters_are_registered() -> None:
    assert {"prism", "column_product", "radiance"} <= set(list_plugins("exporters"))
    for name in ("prism", "column_product", "radiance"):
        exporter = get_exporter(name)
        assert isinstance(exporter, Exporter)
    assert get_exporter("radiance").needs_retrieval is False


@pytest.mark.parametrize("sensor", sorted(SENSORS))
@pytest.mark.parametrize("retriever_name", ["toy", "aviris4_mf"])
def test_every_sensor_retriever_exporter_pairing(
    sensor: str,
    retriever_name: str,
    column: xr.DataArray,
    noise_inputs: Path | None,
    tmp_path: Path,
) -> None:
    retrievers = _retrievers(noise_inputs)
    if retriever_name not in retrievers:
        pytest.skip("aviris4_ch4 or the golden noise model is not available")
    scene = _scene(sensor).load()
    settings = RetrievalSettings(
        statistics_scope="window",
        band_windows_nm=((2080.0, 2500.0),),
        min_n_samples=1,
    )
    product = embed_sample(
        scene,
        column,
        Placement(anchor_along=10, anchor_across=10),
        method="radiance",
        regridder=PolygonRegridder(),
        spec=SPEC,
        provider=_provider(),
        retriever=retrievers[retriever_name],
        settings=settings,
    )
    validate_column_product(product)
    assert product.attrs["retriever"] == retriever_name
    assert product.attrs["sensor"] == "toy"
    for exporter in (PrismExporter(), ColumnProductExporter()):
        prepared = exporter.prepare(product, extra_attrs={"note": "x"})
        assert prepared.attrs["note"] == "x"
        path = exporter.write(prepared, tmp_path / f"{exporter.name}.nc")
        with xr.open_dataset(path) as written:
            assert "injected_ch4_mass" in written

    embedded = embed_radiance_sample(scene, _injected(scene, column), _provider())
    radiance = RadianceExporter()
    path = radiance.write(radiance.prepare(embedded), tmp_path / "radiance.zarr")
    with xr.open_zarr(path) as written:
        validate_sensor_scene(written)
        assert written.sizes["band"] == scene.sizes["band"]


def test_bands_outside_provider_range_pass_through(column: xr.DataArray) -> None:
    scene = _scene("spaceborne_vswir").load()
    injected = _injected(scene, column)
    embedded = embed_radiance_sample(scene, injected, _provider(), spec=SPEC)
    window, _ = scene_window(scene, injected)
    assert embedded.sizes["band"] == scene.sizes["band"]
    np.testing.assert_array_equal(
        embedded["wavelength"].values, scene["wavelength"].values
    )
    wavelength = scene["wavelength"].values
    outside = wavelength < PROVIDER_RANGE[0]
    np.testing.assert_array_equal(
        embedded["radiance"].isel(band=np.flatnonzero(outside)).values,
        window["radiance"].isel(band=np.flatnonzero(outside)).values,
    )
    ratio = (embedded["radiance"] / window["radiance"]).isel(
        band=np.flatnonzero(~outside)
    )
    assert float(ratio.min()) < 1.0
    assert embedded.attrs["embedding_attenuated_wavelength_min_nm"] >= 1000.0
    assert embedded.attrs["embedding_attenuated_band_count"] == int((~outside).sum())

    selected = embed_radiance_sample(
        scene, injected, _provider(), spec=SPEC, band_windows_nm=((2080.0, 2500.0),)
    )
    assert float(selected["wavelength"].min()) >= 2080.0


def test_full_scene_extent_embeds_only_the_window(column: xr.DataArray) -> None:
    scene = _scene("spaceborne_vswir")
    injected = _injected(scene, column)
    window_result = embed_radiance_sample(scene, injected, _provider(), spec=SPEC)
    full = embed_radiance_sample(
        scene, injected, _provider(), spec=SPEC, extent="scene"
    )
    assert full["radiance"].chunks is not None
    assert full.sizes == scene.sizes
    validate_sensor_scene(full)
    _, window = scene_window(scene, injected)

    loaded = full["radiance"].values
    original = scene["radiance"].values
    inside = np.zeros(original.shape[:2], dtype=bool)
    inside[window.across, window.along] = True
    np.testing.assert_array_equal(loaded[~inside], original[~inside])
    np.testing.assert_array_equal(
        full["radiance"].isel(window.indexers).values,
        window_result["radiance"].values,
    )
    mass = full["injected_ch4_mass"].values
    assert float(mass[~inside].max()) == 0.0
    np.testing.assert_allclose(
        full["injected_ch4_mass"].isel(window.indexers).values,
        window_result["injected_ch4_mass"].values,
    )
    with pytest.raises(ContractError, match="extent"):
        embed_radiance_sample(scene, injected, _provider(), extent="cube")  # type: ignore[arg-type]


def test_retrieval_on_full_embedded_scene(column: xr.DataArray) -> None:
    scene = _scene("airborne_swir").load()
    settings = RetrievalSettings(statistics_scope="window", min_n_samples=1)
    product = embed_sample(
        scene,
        column,
        Placement(anchor_along=10, anchor_across=10),
        method="radiance",
        regridder=PolygonRegridder(),
        spec=SPEC,
        provider=_provider(),
        retriever=ToyRetriever(),
        settings=settings,
        extent="scene",
    )
    validate_column_product(product)
    assert product["ch4_mass"].shape == (scene.sizes["across"], scene.sizes["along"])


def test_matched_filter_converter_accepts_other_variable_names() -> None:
    scene = _scene("airborne_swir").load()
    shape = (scene.sizes["across"], scene.sizes["along"])
    output = xr.Dataset(
        {
            "xch4_enhancement": (("across", "along"), np.full(shape, 100.0)),
            "sigma": (("across", "along"), np.full(shape, 50.0)),
        },
        coords={"across": scene["across"], "along": scene["along"]},
    )
    output["xch4_enhancement"].attrs["units"] = "ppm m"
    output["sigma"].attrs["units"] = "ppm m"
    product = matched_filter_to_column_product(
        output,
        attrs={"sensor": "toy", "scene_id": "toy-0001"},
        geometry=scene,
        retriever="external_mf",
        column_variable="xch4_enhancement",
        uncertainty_variable="sigma",
    )
    validate_column_product(product)
    assert product.attrs["retriever"] == "external_mf"
    assert product["ch4_mass"].attrs["native_variable"] == "xch4_enhancement"


# --- recipes ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def plume_files(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("modular")
    files = []
    # Two LES cases: each positive split of the 0.5/0.5 ratios gets one.
    for case, seed in (("caseA", 1), ("caseB", 2)):
        dataset = make_plume_column(
            nx=48, ny=48, dx=2.0, n_time=3, case_id=case, seed=seed
        )
        files.append(write_plume_column(dataset, root / f"{case}.plumecolumn.nc"))
    return files


@pytest.fixture(scope="module")
def plume_file(plume_files: list[Path]) -> Path:
    return plume_files[0]


def _recipe(plumes: list[Path], root: Path, **output: Any) -> dict[str, Any]:
    return {
        "name": "modular",
        "seed": 3,
        "n_samples": 2,
        "method": "radiance",
        "sensor": {"name": "toy", "scenes": "unused"},
        "catalog": {"path": None},
        "plumes": {
            "files": [str(plume) for plume in plumes],
            "emission": {"distribution": "uniform", "low": 500, "high": 1500},
        },
        "embedding": {"unit_absorption": {"provider": "toy"}},
        "negatives": {"fraction": 0.5},
        "split": {"ratios": {"train": 0.5, "val": 0.5, "test": 0.0}},
        "output": {"root": str(root), "dataset_version": "t", **output},
    }


@pytest.mark.parametrize("extent", ["window", "scene"])
def test_radiance_recipe_without_retrieval(
    plume_files: list[Path], tmp_path: Path, extent: str
) -> None:
    recipe = recipe_from_mapping(
        _recipe(plume_files, tmp_path / "data", exporter="radiance", extent=extent)
    )
    plugins = build_plugins(recipe)
    assert plugins.retriever is None and plugins.exporter.name == "radiance"
    table = pd.read_csv(generate(recipe, tmp_path / "run", plugins=plugins))
    assert len(table) == 2
    assert (table["retriever"].fillna("") == "").all()
    adapter = ToySensorAdapter()
    full_sizes = adapter.open_scene(adapter.list_scenes(None, n=1)[0]).sizes
    for row in table.to_dict(orient="records"):
        with xr.open_dataset(str(row["product_uri"])) as sample:
            validate_sensor_scene(sample)
            assert sample.attrs["embedding_method"] == "radiance"
            if extent == "scene":
                assert sample.sizes == full_sizes
            if row["source_id"] == "none":
                assert float(sample["injected_ch4_mass"].max()) == 0.0
            else:
                assert float(sample["injected_ch4_mass"].max()) > 0.0
                assert np.isfinite(row["source_y"])


def test_column_product_recipe(plume_files: list[Path], tmp_path: Path) -> None:
    mapping = _recipe(plume_files, tmp_path / "data", exporter="column_product")
    mapping["retrieval"] = {"retriever": "toy", "min_n_samples": 1}
    recipe = recipe_from_mapping(mapping)
    table = pd.read_csv(generate(recipe, tmp_path / "run"))
    for uri in table["product_uri"]:
        with xr.open_dataset(str(uri)) as sample:
            info = validate_column_product(sample)
            assert info.geometry_space == "sensor"


def test_recipe_rejects_inconsistent_exporters(
    plume_files: list[Path], tmp_path: Path
) -> None:
    root = tmp_path / "data"
    with pytest.raises(ContractError, match="retrieval is required"):
        recipe_from_mapping(_recipe(plume_files, root))
    with_retrieval = _recipe(plume_files, root, exporter="radiance")
    with_retrieval["retrieval"] = {"retriever": "toy"}
    with pytest.raises(ContractError, match="retrieval must be omitted"):
        recipe_from_mapping(with_retrieval)
    product = _recipe(plume_files, root, exporter="radiance")
    product.update(method="product", embedding={})
    with pytest.raises(ContractError, match="needs method 'radiance'"):
        recipe_from_mapping(product)
    with pytest.raises(ContractError, match=r"output\.exporter"):
        recipe_from_mapping(_recipe(plume_files, root, exporter="nonexistent"))
    with pytest.raises(ContractError, match=r"output\.extent"):
        recipe_from_mapping(_recipe(plume_files, root, exporter="radiance", extent="x"))


# --- command line ------------------------------------------------------------------


def _cli_args(plume: Path, save_path: Path, *extra: str) -> list[str]:
    return [
        "embed",
        "radiance",
        "--sensor",
        "toy",
        "--adapter-option",
        "n_across=32",
        "--adapter-option",
        "n_along=48",
        "--file",
        "radiance=unused",
        "--plume",
        str(plume),
        "--anchor-along",
        "10",
        "--anchor-across",
        "10",
        "--provider",
        "toy",
        "--retriever",
        "toy",
        "--min-n-samples",
        "1",
        "--enhancement-min",
        "1",
        "--enhancement-max",
        "1000",
        "--enhancement-nodes",
        "4",
        "--save-path",
        str(save_path),
        *extra,
    ]


def test_cli_exporters_and_extent(plume_file: Path, tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.nc"
    cli.main(
        _cli_args(plume_file, scene_path, "--exporter", "radiance", "--extent", "scene")
    )
    with xr.open_dataset(scene_path) as written:
        validate_sensor_scene(written)
        assert written.sizes["across"] == 32 and written.sizes["along"] == 48

    product_path = tmp_path / "product.nc"
    cli.main(_cli_args(plume_file, product_path))
    with xr.open_dataset(product_path) as written:
        assert validate_column_product(written).geometry_space == "sensor"

    with pytest.raises(SystemExit):
        cli.main(
            _cli_args(
                plume_file, tmp_path / "x.nc", "--exporter", "radiance", "--prism"
            )
        )
