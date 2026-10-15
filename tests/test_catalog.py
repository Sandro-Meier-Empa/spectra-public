"""Scene catalogue, QA metrics and status lifecycle on a fake adapter."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from spectra.catalog import (
    COLUMNS,
    CatalogError,
    QaResult,
    QaSettings,
    QaThresholds,
    SceneCatalog,
    SceneRow,
    compute_scene_qa,
    compute_window_qa,
    record_to_row,
    row_to_record,
)
from spectra.contracts import validate_column_product, validate_sensor_scene
from spectra.geometry.crs import spatial_ref_coordinate, to_lonlat
from spectra.protocols import SensorAdapter, SupportsProduct
from spectra.types import Platform, SceneRecord, Window

N_ACROSS, N_ALONG, N_BAND = 16, 20, 5
SIGMA = 1.0e-5  # kg m-2 background noise of the fake product
CRS_EPSG = 32631
T0 = datetime(2024, 9, 16, 12, 35, 59, tzinfo=UTC)
# In-memory scene table of the fake adapter (its scenes need no files).
SCENE_ROWS: list[dict[str, object]] = [
    {"scene_id": name} for name in ("clean", "plume", "cloudy")
]


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pyproj

    x = 500_000.0 + np.arange(N_ACROSS, dtype=float)[:, None] + np.zeros((1, N_ALONG))
    y = 5_000_000.0 + np.zeros((N_ACROSS, 1)) + np.arange(N_ALONG, dtype=float)[None]
    lon, lat = to_lonlat(pyproj.CRS.from_epsg(CRS_EPSG), x, y)
    return x, y, lon, lat


def make_scene(
    scene_id: str, *, cloud_fraction: float = 0.0, seed: int = 0
) -> xr.Dataset:
    import pyproj

    rng = np.random.default_rng(seed)
    x, y, lon, lat = _grid()
    dims = ("across", "along")
    crs = pyproj.CRS.from_epsg(CRS_EPSG)
    cloud = np.zeros((N_ACROSS, N_ALONG), dtype=bool)
    cloud[:, : round(cloud_fraction * N_ALONG)] = True
    scene = xr.Dataset(
        data_vars={
            "radiance": (
                ("across", "along", "band"),
                rng.uniform(0.01, 0.05, (N_ACROSS, N_ALONG, N_BAND)).astype("float32"),
                {"units": "W m-2 nm-1 sr-1"},
            ),
            "sza": xr.DataArray(40.0, attrs={"units": "degrees"}),
            "saa": xr.DataArray(150.0, attrs={"units": "degrees"}),
            "vza": (dims, np.zeros((N_ACROSS, N_ALONG)), {"units": "degrees"}),
            "vaa": (dims, np.zeros((N_ACROSS, N_ALONG)), {"units": "degrees"}),
            "surface_elevation": (dims, np.full((N_ACROSS, N_ALONG), 200.0)),
            "sensor_altitude": xr.DataArray(1100.0, attrs={"units": "m"}),
            "pixel_area": (dims, np.ones((N_ACROSS, N_ALONG)), {"units": "m2"}),
            "valid": (dims, np.ones((N_ACROSS, N_ALONG), dtype=bool)),
            "cloud": (dims, cloud),
        },
        coords={
            "across": np.arange(N_ACROSS),
            "along": np.arange(N_ALONG),
            "band": np.arange(N_BAND),
            "wavelength": ("band", np.linspace(2100.0, 2400.0, N_BAND)),
            "fwhm": ("band", np.full(N_BAND, 5.0)),
            "x": (dims, x, {"epsg": CRS_EPSG, "units": "m"}),
            "y": (dims, y, {"epsg": CRS_EPSG, "units": "m"}),
            "lon": (dims, lon),
            "lat": (dims, lat),
            "spatial_ref": spatial_ref_coordinate(crs),
            "time": np.datetime64(T0.replace(tzinfo=None), "ns"),
        },
        attrs={
            "sensor": "fake",
            "platform": "airborne",
            "scene_id": scene_id,
            "acquisition_time": T0.isoformat(),
            "background_group": f"fake:{scene_id}",
            "geometry_space": "sensor",
        },
    )
    return scene


def make_product(scene_id: str, *, plume: bool = False, seed: int = 0) -> xr.Dataset:
    import pyproj

    rng = np.random.default_rng(seed)
    x, y, _, _ = _grid()
    dims = ("across", "along")
    mass = rng.normal(0.0, SIGMA, (N_ACROSS, N_ALONG))
    if plume:
        across, along = np.meshgrid(
            np.arange(N_ACROSS), np.arange(N_ALONG), indexing="ij"
        )
        radius2 = (across - 8.0) ** 2 + (along - 10.0) ** 2
        mass += 20.0 * SIGMA * np.exp(-radius2 / (2.0 * 1.5**2))
    product = xr.Dataset(
        data_vars={
            "ch4_mass": (dims, mass, {"units": "kg m-2"}),
            "ch4_mass_uncertainty": (
                dims,
                np.full((N_ACROSS, N_ALONG), SIGMA),
                {"units": "kg m-2"},
            ),
            "retrieval_valid": (dims, np.ones((N_ACROSS, N_ALONG), dtype=bool)),
        },
        coords={
            "across": np.arange(N_ACROSS),
            "along": np.arange(N_ALONG),
            "x": (dims, x, {"epsg": CRS_EPSG}),
            "y": (dims, y, {"epsg": CRS_EPSG}),
            "spatial_ref": spatial_ref_coordinate(pyproj.CRS.from_epsg(CRS_EPSG)),
        },
        attrs={
            "sensor": "fake",
            "scene_id": scene_id,
            "background_group": f"fake:{scene_id}",
            "geometry_space": "sensor",
            "gas": "CH4",
        },
    )
    return product


class FakeAdapter:
    """In-memory adapter with a clean, a plume and a cloudy scene."""

    name: str = "fake"
    platform: Platform = "airborne"

    def __init__(self) -> None:
        specs = {
            "clean": {"cloud_fraction": 0.0, "plume": False, "seed": 1},
            "plume": {"cloud_fraction": 0.0, "plume": True, "seed": 2},
            "cloudy": {"cloud_fraction": 0.3, "plume": False, "seed": 3},
        }
        self.scenes: dict[str, xr.Dataset] = {}
        self.products: dict[str, xr.Dataset] = {}
        self._records: list[SceneRecord] = []
        for offset, (scene_id, spec) in enumerate(specs.items()):
            scene = make_scene(
                scene_id, cloud_fraction=spec["cloud_fraction"], seed=spec["seed"]
            )
            self.scenes[scene_id] = scene
            self.products[scene_id] = make_product(
                scene_id, plume=spec["plume"], seed=spec["seed"]
            )
            lon, lat = scene["lon"].values, scene["lat"].values
            record = SceneRecord(
                sensor=self.name,
                scene_id=scene_id,
                platform="airborne",
                acquisition_time=T0 + timedelta(hours=offset),
                background_group=f"fake:{scene_id}",
                uris={
                    "radiance": f"memory://{scene_id}",
                    "product": f"memory://{scene_id}/ch4",
                },
                center_lonlat=(float(lon.mean()), float(lat.mean())),
                bbox_lonlat=(
                    float(lon.min()),
                    float(lat.min()),
                    float(lon.max()),
                    float(lat.max()),
                ),
                has_radiance=True,
                has_product=True,
                extra={"line": f"{offset + 1:04d}", "kind": scene_id},
            )
            self._records.append(record)

    def list_scenes(
        self,
        scenes: str | Path | Sequence[Mapping[str, object]] | None = None,
        **filters: object,
    ) -> list[SceneRecord]:
        """Return the in-memory records; a rows table selects scenes by id."""
        records = self._records
        if scenes is not None:
            assert not isinstance(scenes, (str, Path)), "the fake takes rows"
            wanted = {str(row["scene_id"]) for row in scenes}
            records = [r for r in records if r.scene_id in wanted]
        if "kind" in filters:
            records = [r for r in records if r.extra["kind"] == filters["kind"]]
        return list(records)

    def open_scene(
        self, record: SceneRecord, *, chunks: Mapping[str, int] | None = None
    ) -> xr.Dataset:
        return self.scenes[record.scene_id]

    def quality_masks(self, scene: xr.Dataset, record: SceneRecord) -> xr.Dataset:
        return scene[[name for name in ("valid", "cloud") if name in scene]]

    def background_group(self, record: SceneRecord) -> str:
        return f"fake:{record.scene_id}"

    def open_product(self, record: SceneRecord) -> xr.Dataset:
        return self.products[record.scene_id]


@pytest.fixture(scope="module")
def adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def catalog(adapter: FakeAdapter) -> SceneCatalog:
    return SceneCatalog.build(adapter, SCENE_ROWS, inspect=True)


def _qa_all(catalog: SceneCatalog, adapter: FakeAdapter, qa_version: str = "1") -> None:
    for record in catalog.records():
        result = compute_scene_qa(
            adapter.open_scene(record), adapter.open_product(record), QaSettings()
        )
        catalog.update_qa(record.scene_id, result, qa_version=qa_version)


# -- fixtures and schema --------------------------------------------------------


def test_fake_adapter_satisfies_protocols_and_contracts(adapter: FakeAdapter) -> None:
    assert isinstance(adapter, SensorAdapter)
    assert isinstance(adapter, SupportsProduct)
    for record in adapter.list_scenes():
        info = validate_sensor_scene(adapter.open_scene(record), require_lazy=False)
        assert (info.n_across, info.n_along, info.n_band) == (N_ACROSS, N_ALONG, N_BAND)
        product_info = validate_column_product(adapter.open_product(record))
        assert product_info.scene_id == record.scene_id


def test_schema_columns_match_row_fields() -> None:
    from dataclasses import fields

    assert tuple(f.name for f in fields(SceneRow)) == tuple(COLUMNS)
    assert COLUMNS["scene_id"] == "string"
    assert COLUMNS["acquisition_time"] == "datetime64[ns, UTC]"


def test_record_row_round_trip(adapter: FakeAdapter) -> None:
    record = adapter.list_scenes()[0]
    row = record_to_row(record)
    assert row.status == "discovered"
    assert json.loads(row.uris) == dict(record.uris)
    assert row_to_record(row) == record
    naive = SceneRecord(
        **{**record.__dict__, "acquisition_time": T0.replace(tzinfo=None)}
    )
    assert row_to_record(record_to_row(naive)).acquisition_time == T0


# -- build / persistence --------------------------------------------------------


def test_build_discovers_three_scenes(catalog: SceneCatalog) -> None:
    frame = catalog.frame
    assert len(catalog) == 3
    assert list(frame.columns) == list(COLUMNS)
    assert set(frame["status"]) == {"discovered"}
    assert list(frame["n_across"]) == [N_ACROSS] * 3
    assert list(frame["n_along"]) == [N_ALONG] * 3
    assert list(frame["n_band"]) == [N_BAND] * 3
    assert np.allclose(frame["pixel_size_m_median"], 1.0)
    assert list(frame["background_group"]) == [
        "fake:clean",
        "fake:plume",
        "fake:cloudy",
    ]
    assert str(frame["acquisition_time"].dtype) == "datetime64[ns, UTC]"
    assert frame["has_product"].dtype == bool
    assert bool(frame["valid_fraction"].isna().all())
    assert bool(frame["n_detections"].isna().all())


def test_build_without_inspect_and_with_filters(adapter: FakeAdapter) -> None:
    catalog = SceneCatalog.build(adapter, SCENE_ROWS, inspect=False, kind="plume")
    assert len(catalog) == 1
    assert bool(catalog.frame["n_across"].isna().all())
    assert bool(catalog.frame["pixel_size_m_median"].isna().all())
    subset = SceneCatalog.build(adapter, SCENE_ROWS[1:], inspect=False)
    assert [r.scene_id for r in subset.records()] == ["plume", "cloudy"]
    records = adapter.list_scenes()[:2]
    from_records = SceneCatalog.build(adapter, None, records=records, inspect=False)
    assert [r.scene_id for r in from_records.records()] == ["clean", "plume"]
    with pytest.raises(CatalogError, match="scene table or records"):
        SceneCatalog.build(adapter, None, inspect=False)


def test_duplicate_scene_ids_are_rejected(adapter: FakeAdapter) -> None:
    record = adapter.list_scenes()[0]
    with pytest.raises(CatalogError, match="unique"):
        SceneCatalog.build(adapter, None, records=[record, record], inspect=False)


def test_write_read_round_trip(
    catalog: SceneCatalog, adapter: FakeAdapter, tmp_path: Path
) -> None:
    catalog.write(tmp_path / "catalog")
    _qa_all(catalog, adapter)
    catalog.apply_thresholds(QaThresholds())
    directory = catalog.write()
    assert directory == tmp_path / "catalog"
    assert (directory / "scenes.parquet").exists()
    meta = json.loads((directory / "catalog.json").read_text())
    assert meta["n_scenes"] == 3
    assert meta["catalog_version"] == "1.0"
    loaded = SceneCatalog.read(directory)
    pd.testing.assert_frame_equal(loaded.frame, catalog.frame)
    assert loaded.frame.dtypes.astype(str).to_dict() == {
        name: str(pd.Series(dtype=dtype).dtype) for name, dtype in COLUMNS.items()
    }
    assert loaded.directory == directory
    assert [r.scene_id for r in loaded.records()] == [
        r.scene_id for r in catalog.records()
    ]


# -- QA metrics ----------------------------------------------------------------


def test_scene_qa_clean_scene(adapter: FakeAdapter) -> None:
    result = compute_scene_qa(
        adapter.scenes["clean"], adapter.products["clean"], QaSettings()
    )
    assert result.valid_fraction == 1.0
    assert result.cloud_fraction == 0.0
    assert np.isnan(result.shadow_fraction)
    assert result.n_detections == 0
    assert np.isnan(result.max_detection_snr)
    assert abs(result.bg_median_kgm2) < 3 * SIGMA / np.sqrt(N_ACROSS * N_ALONG) * 3
    assert 0.5 * SIGMA < result.bg_std_kgm2 < 1.5 * SIGMA
    assert 0.3 * SIGMA < result.bg_mad_kgm2 < 1.2 * SIGMA
    assert result.uncertainty_median_kgm2 == SIGMA
    assert result.detections is not None and not result.detections.any()


def test_scene_qa_plume_scene(adapter: FakeAdapter) -> None:
    result = compute_scene_qa(
        adapter.scenes["plume"], adapter.products["plume"], QaSettings()
    )
    assert result.n_detections is not None and result.n_detections >= 1
    assert result.max_detection_snr > 3.0
    assert result.detections is not None
    assert tuple(result.detections.dims) == ("across", "along")
    assert bool(result.detections.sel(across=8, along=10))
    assert int(result.detections.sum()) >= 9
    without = compute_scene_qa(
        adapter.scenes["plume"], adapter.products["plume"], QaSettings(detection="none")
    )
    assert without.n_detections is None and without.detections is None
    assert without.bg_median_kgm2 == result.bg_median_kgm2


def test_scene_qa_cloudy_scene(adapter: FakeAdapter) -> None:
    result = compute_scene_qa(
        adapter.scenes["cloudy"], adapter.products["cloudy"], QaSettings()
    )
    assert result.cloud_fraction == pytest.approx(0.3)
    assert result.valid_fraction == 1.0
    assert result.n_detections == 0


def test_scene_qa_without_product_and_with_subsample(adapter: FakeAdapter) -> None:
    result = compute_scene_qa(adapter.scenes["cloudy"], None, QaSettings(subsample=2))
    assert result.cloud_fraction == pytest.approx(0.3)
    assert np.isnan(result.bg_median_kgm2)
    assert result.n_detections is None
    assert result.subsample == 2
    sub = compute_scene_qa(
        adapter.scenes["plume"], adapter.products["plume"], QaSettings(subsample=2)
    )
    assert sub.detections is not None and sub.detections.shape == (
        N_ACROSS // 2,
        N_ALONG // 2,
    )


def test_scene_qa_handles_dask_masks_and_invalid_pixels(adapter: FakeAdapter) -> None:
    pytest.importorskip("dask")
    scene = adapter.scenes["plume"].chunk({"across": 4, "along": 10, "band": -1})
    scene["valid"] = scene["valid"].copy(data=np.ones(scene["valid"].shape, dtype=bool))
    scene["valid"][:, :4] = False
    product = adapter.products["plume"].copy(deep=True)
    product["ch4_mass"][:, :4] = np.nan
    product["ch4_mass"][0, 5] = np.nan
    result = compute_scene_qa(scene, product, QaSettings())
    assert result.valid_fraction == pytest.approx(0.8)
    assert np.isfinite(result.bg_median_kgm2)
    assert result.n_detections == 1
    assert result.detections is not None and not result.detections[:, :4].any()


def test_window_qa(adapter: FakeAdapter) -> None:
    window = Window(across=slice(4, 13), along=slice(5, 16))
    product = adapter.products["plume"]
    settings = QaSettings(background_filter_px=5)
    inside = compute_window_qa(product, window, settings, scene=adapter.scenes["plume"])
    assert inside.valid_fraction == 1.0
    assert inside.n_detections is not None and inside.n_detections >= 1
    assert inside.detections is not None and inside.detections.shape == window.shape
    outside = compute_window_qa(product, Window(slice(0, 4), slice(0, 20)), settings)
    assert outside.n_detections == 0
    assert outside.valid_fraction == 1.0


def test_ddeq_detection_is_experimental_but_runs(adapter: FakeAdapter) -> None:
    pytest.importorskip("ddeq")
    settings = QaSettings(detection="ddeq", background_filter_px=11)
    result = compute_scene_qa(
        adapter.scenes["plume"], adapter.products["plume"], settings
    )
    assert result.detection_method == "ddeq"
    assert result.n_detections is not None and result.n_detections >= 1
    assert result.max_detection_snr > 3.0
    clean = compute_scene_qa(
        adapter.scenes["clean"], adapter.products["clean"], settings
    )
    assert clean.n_detections == 0


def test_scene_qa_mad_noise_model(adapter: FakeAdapter) -> None:
    scene, product = adapter.scenes["plume"], adapter.products["plume"]
    inflated = product.copy(deep=True)
    inflated["ch4_mass_uncertainty"] = inflated["ch4_mass_uncertainty"] * 100.0
    assert compute_scene_qa(scene, inflated, QaSettings()).n_detections == 0
    robust = compute_scene_qa(scene, inflated, QaSettings(noise="mad"))
    assert robust.n_detections is not None and robust.n_detections >= 1
    assert robust.max_detection_snr > 3.0
    clean = compute_scene_qa(
        adapter.scenes["clean"], adapter.products["clean"], QaSettings(noise="mad")
    )
    assert clean.n_detections == 0


def test_qa_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="detection"):
        QaSettings(detection="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subsample"):
        QaSettings(subsample=0)


# -- lifecycle ---------------------------------------------------------------------


def test_update_qa_and_thresholds(
    catalog: SceneCatalog, adapter: FakeAdapter, tmp_path: Path
) -> None:
    catalog.write(tmp_path / "cat")
    _qa_all(catalog, adapter)
    assert set(catalog.frame["status"]) == {"qa_done"}
    assert catalog.frame["qa_version"].tolist() == ["1", "1", "1"]
    assert bool(catalog.frame["qa_utc"].notna().all())
    assert catalog.frame["status_set_by"].tolist() == ["spectra.catalog"] * 3
    detection_uri = catalog.row("plume").detection_uri
    assert detection_uri is not None and Path(detection_uri).exists()
    assert Path(detection_uri).parent == tmp_path / "cat" / "detections"
    mask = xr.open_dataarray(detection_uri).load()
    assert mask.dtype == bool and bool(mask.any())

    outcomes = catalog.apply_thresholds(QaThresholds())
    assert outcomes == {
        "clean": "eligible",
        "plume": "manual_review",
        "cloudy": "rejected",
    }
    assert catalog.row("cloudy").status_reason is not None
    assert "cloud_fraction" in str(catalog.row("cloudy").status_reason)
    assert catalog.apply_thresholds(QaThresholds()) == {}

    eligible = catalog.records(status="eligible")
    assert len(eligible) == 1
    expected = adapter.list_scenes()[0]
    assert eligible[0] == expected
    assert eligible[0].acquisition_time == expected.acquisition_time
    assert eligible[0].extra == expected.extra
    assert {r.scene_id for r in catalog.records(status=["eligible", "rejected"])} == {
        "clean",
        "cloudy",
    }

    filtered = catalog.filter(max_cloud_fraction=0.1)
    assert set(filtered["scene_id"]) == {"clean", "plume"}
    assert set(catalog.filter(status="rejected")["scene_id"]) == {"cloudy"}
    assert set(catalog.filter(status=("eligible", "manual_review"))["scene_id"]) == {
        "clean",
        "plume",
    }
    assert set(catalog.filter(date_min=T0 + timedelta(hours=1))["scene_id"]) == {
        "plume",
        "cloudy",
    }
    assert set(catalog.filter(date_max="2024-09-16T13:00:00")["scene_id"]) == {"clean"}
    assert set(catalog.filter(min_n_detections=1)["scene_id"]) == {"plume"}
    assert set(catalog.filter(has_product=True, sensor="fake")["scene_id"]) == {
        "clean",
        "plume",
        "cloudy",
    }
    with pytest.raises(CatalogError, match="criterion"):
        catalog.filter(nonsense=1)

    summary = catalog.summary()
    assert summary.loc["fake", "eligible"] == 1
    assert summary.loc["fake", "manual_review"] == 1
    assert summary.loc["fake", "rejected"] == 1
    assert summary.loc["fake", "total"] == 3
    assert summary.loc["total", "total"] == 3
    assert summary.loc["fake", "discovered"] == 0


def test_thresholds_quarantine_and_background_std(
    catalog: SceneCatalog, adapter: FakeAdapter
) -> None:
    _qa_all(catalog, adapter)
    catalog.frame.loc[catalog.frame["scene_id"] == "clean", "annotation_uri"] = (
        "file:///ann.nc"
    )
    outcomes = catalog.apply_thresholds(QaThresholds(max_bg_std_kgm2=0.5 * SIGMA))
    assert outcomes["clean"] == "quarantined"
    assert outcomes["cloudy"] == "rejected"
    assert outcomes["plume"] == "rejected"
    assert "bg_std" in str(catalog.row("plume").status_reason)


def test_thresholds_send_unknown_metrics_to_manual_review(
    catalog: SceneCatalog, adapter: FakeAdapter
) -> None:
    _qa_all(catalog, adapter)
    catalog.frame.loc[catalog.frame["scene_id"] == "clean", "cloud_fraction"] = np.nan
    outcomes = catalog.apply_thresholds(QaThresholds())
    assert outcomes["clean"] == "manual_review"
    assert "cloud_fraction" in str(catalog.row("clean").status_reason)


def test_set_status_transitions(catalog: SceneCatalog, adapter: FakeAdapter) -> None:
    with pytest.raises(CatalogError, match="not allowed"):
        catalog.set_status("clean", "eligible", reason="skip qa", by="me")
    with pytest.raises(CatalogError, match="Unknown status"):
        catalog.set_status("clean", "wonderful", reason="x", by="me")
    with pytest.raises(CatalogError, match="Unknown scene_id"):
        catalog.set_status("missing", "qa_pending", reason="x", by="me")
    catalog.set_status("clean", "qa_pending", reason="queued", by="me")
    assert catalog.row("clean").status == "qa_pending"
    _qa_all(catalog, adapter)
    catalog.apply_thresholds(QaThresholds())
    before = datetime.now(tz=UTC)
    catalog.set_status("clean", "quarantined", reason="real plume seen", by="sandro")
    row = catalog.row("clean")
    assert row.status == "quarantined"
    assert row.status_reason == "real plume seen"
    assert row.status_set_by == "sandro"
    assert row.status_set_utc is not None and row.status_set_utc >= before
    assert str(catalog.frame["status_set_utc"].dtype) == "datetime64[ns, UTC]"
    catalog.set_status("clean", "eligible", reason="false alarm", by="sandro")
    assert catalog.row("clean").status == "eligible"


def test_update_qa_is_append_only(catalog: SceneCatalog, adapter: FakeAdapter) -> None:
    record = catalog.records()[0]
    result = compute_scene_qa(adapter.open_scene(record), adapter.open_product(record))
    catalog.update_qa(record.scene_id, result, qa_version="1.0")
    with pytest.raises(CatalogError, match="append-only"):
        catalog.update_qa(record.scene_id, result, qa_version="1.0")
    with pytest.raises(CatalogError, match="append-only"):
        catalog.update_qa(record.scene_id, result, qa_version="0.9")
    catalog.apply_thresholds()
    assert catalog.row(record.scene_id).status == "eligible"
    newer = QaResult(valid_fraction=0.5, cloud_fraction=0.0, n_detections=0)
    catalog.update_qa(record.scene_id, newer, qa_version="1.1")
    row = catalog.row(record.scene_id)
    assert row.status == "qa_done"
    assert row.qa_version == "1.1"
    assert row.valid_fraction == 0.5
    assert np.isnan(row.bg_median_kgm2)
    assert row.detection_uri is None
    assert catalog.apply_thresholds() == {record.scene_id: "rejected"}
    catalog.update_qa(record.scene_id, result, qa_version="1.1", force=True)
    assert catalog.row(record.scene_id).valid_fraction == 1.0


# -- real data -----------------------------------------------------------------------


@pytest.mark.data
def test_aviris4_catalogue_and_qa(aviris4_scenes_table: Path, tmp_path: Path) -> None:
    pytest.importorskip("aviris4_ch4")
    from spectra.sensors.aviris4 import Aviris4Adapter

    adapter = Aviris4Adapter()
    n_listed = len(pd.read_csv(aviris4_scenes_table, dtype=str, keep_default_na=False))
    started = time.perf_counter()
    catalog = SceneCatalog.build(adapter, aviris4_scenes_table, inspect=False)
    build_seconds = time.perf_counter() - started
    assert len(catalog) == n_listed >= 1
    assert set(catalog.frame["status"]) == {"discovered"}
    assert catalog.frame["acquisition_time"].is_monotonic_increasing
    directory = catalog.write(tmp_path / "aviris4")
    loaded = SceneCatalog.read(directory)
    pd.testing.assert_frame_equal(loaded.frame, catalog.frame)

    with_product = [r for r in loaded.records() if r.has_product]
    assert with_product, "no flight line with a matched-filter product"
    record = with_product[0]
    started = time.perf_counter()
    scene = adapter.open_scene(record)
    product = adapter.open_product(record)
    open_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = compute_scene_qa(scene, product, QaSettings(subsample=8))
    qa_seconds = time.perf_counter() - started
    print(
        f"\naviris4 catalogue: {len(catalog)} scenes in {build_seconds:.1f} s; "
        f"open {open_seconds:.1f} s; QA(subsample=8) {qa_seconds:.1f} s; {result}"
    )
    assert qa_seconds < 120.0
    assert 0.0 < result.valid_fraction <= 1.0
    assert np.isfinite(result.bg_median_kgm2)
    assert np.isfinite(result.bg_std_kgm2)
    assert (
        np.isfinite(result.uncertainty_median_kgm2)
        and result.uncertainty_median_kgm2 > 0
    )
    assert result.n_detections is not None and result.n_detections >= 0
    for name in ("cloud_fraction", "shadow_fraction"):
        value = getattr(result, name)
        assert np.isnan(value) or 0.0 <= value <= 1.0
    loaded.update_qa(record.scene_id, result, qa_version="1")
    outcomes = loaded.apply_thresholds(QaThresholds())
    assert outcomes[record.scene_id] in {"eligible", "rejected", "manual_review"}
    assert Path(str(loaded.row(record.scene_id).detection_uri)).exists()
    loaded.write()
    again = SceneCatalog.read(directory)
    pd.testing.assert_frame_equal(again.frame, loaded.frame)
