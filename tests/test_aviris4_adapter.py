"""AVIRIS-4 adapter: explicit scene files, scene tables and the CLI entry.

The adapter never discovers files; tests with synthetic NetCDF headers run
everywhere, the contract checks on real flight lines need the scene table in
``SPECTRA_TEST_AVIRIS4_SCENES``.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from spectra import cli
from spectra.contracts import validate_column_product, validate_sensor_scene
from spectra.contracts.errors import ContractError
from spectra.sensors.aviris4 import ROLES, Aviris4Adapter
from spectra.types import Platform, SceneRecord, SpectralResponse, ViewingGeometry


# -- synthetic file headers ------------------------------------------------------
def write_scene_files(
    directory: Path, stem: str, time: str, *, lon0: float = 0.0
) -> tuple[Path, Path]:
    """Write a radiance header and a metadata file the record builder can read."""
    directory.mkdir(parents=True, exist_ok=True)
    radiance = xr.Dataset(
        {"radiance": (("across", "along", "band"), np.zeros((2, 3, 4), "float32"))},
        coords={"time": np.datetime64(time, "ns")},
    )
    lon = lon0 + np.linspace(-0.01, 0.01, 6).reshape(2, 3)
    lat = 43.0 + np.linspace(-0.01, 0.01, 6).reshape(2, 3)
    metadata = xr.Dataset(
        {"lon": (("across", "along"), lon), "lat": (("across", "along"), lat)}
    )
    radiance_path = directory / f"{stem}.nc"
    metadata_path = directory / f"{stem}_metadata.nc"
    radiance.to_netcdf(radiance_path)
    metadata.to_netcdf(metadata_path)
    return radiance_path, metadata_path


@pytest.fixture
def synthetic(tmp_path: Path) -> dict[str, Path]:
    late = write_scene_files(tmp_path / "L1", "late_rdn", "2024-09-16T13:33:13")
    early = write_scene_files(tmp_path / "L1", "early_rdn", "2024-09-16T12:29:20")
    product = tmp_path / "L2" / "early_rdn_CH4.nc"
    product.parent.mkdir()
    product.write_bytes(b"")
    return {
        "late_radiance": late[0],
        "late_metadata": late[1],
        "early_radiance": early[0],
        "early_metadata": early[1],
        "early_product": product,
    }


def test_roles_and_record_from_files(synthetic: dict[str, Path]) -> None:
    assert ROLES == {
        "radiance": True,
        "metadata": True,
        "shadow": False,
        "cloud": False,
        "product": False,
    }
    adapter = Aviris4Adapter()
    record = adapter.record_from_files(
        synthetic["late_radiance"], synthetic["late_metadata"]
    )
    assert record.sensor == "aviris4" and record.platform == "airborne"
    assert record.scene_id == "late_rdn"
    assert record.background_group == "aviris4:late_rdn"
    assert adapter.background_group(record) == "aviris4:late_rdn"
    assert record.acquisition_time == datetime(2024, 9, 16, 13, 33, 13, tzinfo=UTC)
    assert record.uris == {
        "radiance": str(synthetic["late_radiance"].resolve()),
        "metadata": str(synthetic["late_metadata"].resolve()),
    }
    assert record.has_radiance and not record.has_product
    assert record.center_lonlat == pytest.approx((0.0, 43.0))
    assert record.bbox_lonlat == pytest.approx((-0.01, 42.99, 0.01, 43.01))

    explicit = adapter.record_from_files(
        str(synthetic["early_radiance"]),
        str(synthetic["early_metadata"]),
        product=synthetic["early_product"],
        scene_id="pau-0002",
        background_group="aviris4:pau-day1",
    )
    assert explicit.scene_id == "pau-0002"
    assert explicit.background_group == "aviris4:pau-day1"
    assert explicit.has_product
    assert explicit.uris["product"] == str(synthetic["early_product"].resolve())
    assert adapter.background_group(explicit) == "aviris4:pau-day1"

    with pytest.raises(ContractError, match="shadow file .* is missing"):
        adapter.record_from_files(
            synthetic["late_radiance"],
            synthetic["late_metadata"],
            shadow=synthetic["late_radiance"].with_name("absent_shadow.nc"),
        )


def test_list_scenes_from_table_is_sorted_and_filtered(
    tmp_path: Path, synthetic: dict[str, Path]
) -> None:
    table = tmp_path / "scenes.csv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "radiance", "metadata", "product", "line"])
        # Relative paths resolve against the table; the later scene is listed first.
        writer.writerow(["late", "L1/late_rdn.nc", "L1/late_rdn_metadata.nc", "", "1"])
        writer.writerow(
            ["early", "L1/early_rdn.nc", "L1/early_rdn_metadata.nc", "L2/early_rdn_CH4.nc", "2"]
        )  # fmt: skip
    adapter = Aviris4Adapter()
    records = adapter.list_scenes(table)
    assert [record.scene_id for record in records] == ["early", "late"]
    assert [record.has_product for record in records] == [True, False]
    assert [record.extra for record in records] == [{"line": "2"}, {"line": "1"}]
    assert all(
        record.background_group == f"aviris4:{record.scene_id}" for record in records
    )
    assert [r.scene_id for r in adapter.list_scenes(table, scene_id="late")] == ["late"]
    assert [r.scene_id for r in adapter.list_scenes(table, line="2")] == ["early"]
    assert adapter.list_scenes(table, scene_id=["x"]) == []


# -- command line ----------------------------------------------------------------
class RecordingAdapter:
    """Fake adapter that keeps the rows ``spectra embed`` hands to ``list_scenes``."""

    name: str = "fake"
    platform: Platform = "airborne"

    def __init__(self) -> None:
        self.rows: list[Mapping[str, object]] = []

    def list_scenes(
        self,
        scenes: str | Path | Sequence[Mapping[str, object]],
        **filters: object,
    ) -> list[SceneRecord]:
        assert not isinstance(scenes, (str, Path))
        self.rows.extend(scenes)
        return [
            SceneRecord(
                sensor=self.name,
                scene_id=str(row.get("scene_id") or "default-id"),
                platform=self.platform,
                acquisition_time=datetime(2024, 9, 16, tzinfo=UTC),
                background_group=str(row.get("background_group") or "fake:default"),
                uris={
                    key: str(value)
                    for key, value in row.items()
                    if key not in ("scene_id", "background_group")
                },
                center_lonlat=(0.0, 0.0),
            )
            for row in scenes
        ]

    def open_scene(
        self, record: SceneRecord, *, chunks: Mapping[str, int] | None = None
    ) -> xr.Dataset:
        raise NotImplementedError

    def quality_masks(self, scene: xr.Dataset, record: SceneRecord) -> xr.Dataset:
        raise NotImplementedError

    def background_group(self, record: SceneRecord) -> str:
        return record.background_group


def embed_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "file": None,
        "radiance": None,
        "metadata": None,
        "shadow": None,
        "cloud": None,
        "product": None,
        "scene_id": None,
        "background_group": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cli_scene_record_merges_file_flags() -> None:
    adapter = RecordingAdapter()
    record = cli._scene_record(
        adapter,
        embed_args(
            radiance="/path/to/a.nc",
            metadata="/path/to/a_metadata.nc",
            file=["shadow=/path/to/a_shadow.nc", "dem = /path/to/dem.nc "],
            scene_id="pau-1",
            background_group="aviris4:day1",
        ),
    )
    assert adapter.rows == [
        {
            "shadow": "/path/to/a_shadow.nc",
            "dem": "/path/to/dem.nc",
            "radiance": "/path/to/a.nc",
            "metadata": "/path/to/a_metadata.nc",
            "scene_id": "pau-1",
            "background_group": "aviris4:day1",
        }
    ]
    assert record.scene_id == "pau-1" and record.background_group == "aviris4:day1"
    assert set(record.uris) == {"shadow", "dem", "radiance", "metadata"}

    # --radiance wins over a --file radiance=... entry; defaults stay empty.
    adapter = RecordingAdapter()
    record = cli._scene_record(
        adapter, embed_args(file=["radiance=/old.nc"], radiance="/new.nc")
    )
    assert adapter.rows == [
        {"radiance": "/new.nc", "scene_id": "", "background_group": ""}
    ]
    assert record.scene_id == "default-id"


def test_cli_scene_record_rejects_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli._scene_record(RecordingAdapter(), embed_args())
    assert "--file ROLE=PATH" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli._scene_record(RecordingAdapter(), embed_args(file=["radiance"]))
    assert "ROLE=PATH" in capsys.readouterr().err
    # The real adapter checks the files before anything heavy happens.
    with pytest.raises(SystemExit):
        cli._scene_record(
            Aviris4Adapter(),
            embed_args(radiance="/path/to/absent.nc", metadata="/path/to/absent_m.nc"),
        )
    assert "is missing" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["radiance", "product"])
def test_cli_embed_help_lists_file_flags(
    method: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["embed", method, "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--file ROLE=PATH", "--radiance PATH", "--metadata PATH"):
        assert flag in out
    for flag in (
        "--shadow",
        "--cloud",
        "--product",
        "--scene-id",
        "--background-group",
    ):
        assert flag in out
    assert "--root" not in out


# -- real flight lines -------------------------------------------------------------
@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("aviris4_ch4")
    return Aviris4Adapter()


@pytest.fixture(scope="module")
def records(adapter: Aviris4Adapter, aviris4_scenes_table: Path) -> list[SceneRecord]:
    records = adapter.list_scenes(aviris4_scenes_table)
    assert records, f"{aviris4_scenes_table} lists no scenes"
    return records


@pytest.fixture(scope="module")
def record(records: list[SceneRecord]) -> SceneRecord:
    return records[0]


@pytest.mark.data
def test_records_are_sorted_and_grouped(
    adapter: Aviris4Adapter, records: list[SceneRecord], aviris4_scenes_table: Path
) -> None:
    times = [item.acquisition_time for item in records]
    assert times == sorted(times)
    assert all({"radiance", "metadata"} <= set(item.uris) for item in records)
    assert all(item.background_group == f"aviris4:{item.scene_id}" for item in records)
    assert all(
        item.background_group == adapter.background_group(item) for item in records
    )
    first = adapter.list_scenes(aviris4_scenes_table, scene_id=records[0].scene_id)
    assert [item.scene_id for item in first] == [records[0].scene_id]


@pytest.mark.data
def test_background_group_comes_from_the_table(
    adapter: Aviris4Adapter,
    records: list[SceneRecord],
    aviris4_scenes_table: Path,
    tmp_path: Path,
) -> None:
    with aviris4_scenes_table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        fields = [name for name in reader.fieldnames if name != "background_group"]
        rows = list(reader)
    base = aviris4_scenes_table.parent
    copy = tmp_path / "grouped.csv"
    with copy.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, "background_group"])
        writer.writeheader()
        for row in rows:
            out = {
                key: str(base / row[key]) if key in ROLES and row[key] else row[key]
                for key in fields
            }
            writer.writerow({**out, "background_group": "aviris4:pau-2024"})
    grouped = adapter.list_scenes(copy)
    assert [item.scene_id for item in grouped] == [item.scene_id for item in records]
    assert {item.background_group for item in grouped} == {"aviris4:pau-2024"}
    assert all(adapter.background_group(item) == "aviris4:pau-2024" for item in grouped)
    assert [item.uris for item in grouped] == [item.uris for item in records]


@pytest.mark.data
def test_scene_follows_contract(adapter: Aviris4Adapter, record: SceneRecord) -> None:
    scene = adapter.open_scene(record)
    info = validate_sensor_scene(scene, require_lazy=True)
    assert info.sensor == "aviris4"
    assert info.scene_id == record.scene_id
    masks = adapter.quality_masks(scene, record)
    assert "valid" in masks
    assert all(masks[name].dtype.kind == "b" for name in masks.data_vars)
    geometry = ViewingGeometry.from_scene(scene, reducer="center")
    assert 0.0 <= geometry.sza < 90.0
    assert geometry.sensor_altitude > geometry.surface_elevation
    response = SpectralResponse.from_scene(scene)
    assert response.select([(2080.0, 2500.0)]).size > 20
    assert float(scene["pixel_area"][::50, ::500].median()) > 0
    if "shadow" in record.uris:
        assert "shadow" in masks


@pytest.mark.data
def test_product_follows_contract(adapter: Aviris4Adapter, record: SceneRecord) -> None:
    if not record.has_product:
        pytest.skip("flight line has no matched-filter product")
    product = adapter.open_product(record)
    info = validate_column_product(product)
    assert info.geometry_space == "sensor"
    assert info.crs is not None
    sample = product["ch4_mass_uncertainty"][::50, ::500].values
    assert np.all(sample[np.isfinite(sample)] > 0)


@pytest.mark.data
def test_scene_geometry_is_lazy(adapter: Aviris4Adapter, record: SceneRecord) -> None:
    geometry = adapter.scene_geometry(record)
    assert geometry.shape[0] > 1000
    assert geometry.crs.is_projected
