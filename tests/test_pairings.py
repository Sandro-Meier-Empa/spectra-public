"""Pairings table: columns, row construction, round trip and leakage rule."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.export.pairings import (
    PAIRINGS_COLUMNS,
    PRISM_REQUIRED_COLUMNS,
    assert_group_independence,
    pairings_row,
    read_pairings,
    scene_uri,
    write_pairings,
)
from spectra.recipe.params import SampleParams
from spectra.recipe.schema import Recipe, recipe_from_mapping
from spectra.types import Placement, Window


def _recipe(method: str = "product") -> Recipe:
    mapping = {
        "name": "unit",
        "seed": 1,
        "n_samples": 1,
        "method": method,
        "sensor": {"name": "toy", "scenes": "unused"},
        "plumes": {
            "files": ["plume.nc"],
            "emission": {"distribution": "fixed", "values": [100]},
        },
        "output": {"root": "out", "dataset_version": "v"},
    }
    if method == "radiance":
        mapping["embedding"] = {"unit_absorption": {"provider": "toy"}}
        mapping["retrieval"] = {"retriever": "toy"}
    return recipe_from_mapping(mapping)


def _params(*, negative: bool = False) -> SampleParams:
    return SampleParams(
        sample_id="unit-000000",
        index=0,
        split="train",
        sensor="toy",
        scene_id="toy-0001",
        background_group="toy:toy-0001",
        plume_uri="/data/plume.nc",
        les_case="none:unit-000000" if negative else "caseA",
        tracer="none" if negative else "ch4",
        source_id="none" if negative else "caseA:ch4",
        time_index=1,
        simulation_time_s=120.0,
        emission_rate_kg_s=0.0 if negative else 0.5,
        emission_scale=0.0 if negative else 0.5,
        placement=Placement(
            anchor_along=10,
            anchor_across=4,
            anchor_corner="upper left",
            rotation_deg=30,
            flip="x",
            emission_scale=1.0 if negative else 0.5,
            time_index=1,
        ),
        window=Window(across=slice(2, 12), along=slice(5, 25)),
        is_negative=negative,
        pair_id="abc",
        params_hash="def",
    )


def _product(**attrs: object) -> xr.Dataset:
    base: dict[str, object] = {
        "sensor": "toy",
        "embedding_method": "product",
        "retriever": "toy",
        "uncertainty_source": "constant",
        "pixel_size_m": 10.0,
        "scene_uri": scene_uri({"radiance": "/a.nc"}),
        "recipe_hash": "r" * 64,
    }
    base.update(attrs)
    return xr.Dataset(attrs=base)


def test_columns_start_with_prism_requirements() -> None:
    assert PAIRINGS_COLUMNS[: len(PRISM_REQUIRED_COLUMNS)] == PRISM_REQUIRED_COLUMNS
    assert {"source_y", "source_x", "pair_id", "recipe_hash", "params_hash"} <= set(
        PAIRINGS_COLUMNS
    )
    assert len(set(PAIRINGS_COLUMNS)) == len(PAIRINGS_COLUMNS)
    prism = pytest.importorskip("prism.cli.prepare_dataset")
    assert set(prism.REQUIRED_PAIRING_COLUMNS) <= set(PAIRINGS_COLUMNS)


def test_pairings_row_positive() -> None:
    row = pairings_row(
        _product(injected_source_y=3, injected_source_x=7),
        _params(),
        _recipe(),
        Path("/out/samples/unit-000000.nc"),
    )
    assert list(row) == list(PAIRINGS_COLUMNS)
    assert row["product_uri"] == "/out/samples/unit-000000.nc"
    assert row["source_y"] == 3.0 and row["source_x"] == 7.0
    assert row["emission_rate_kg_h"] == pytest.approx(1800.0)
    assert row["rotation_deg"] == 30 and row["flip"] == "x"
    assert row["window_across_start"] == 2 and row["window_along_start"] == 5
    assert row["embedding_method"] == "product"
    assert row["unit_absorption_provider"] == ""
    assert row["mf_statistics_scope"] == ""
    assert row["retriever"] == "toy"
    assert row["pixel_size_m"] == 10.0
    assert row["scene_uri"] == '{"radiance": "/a.nc"}'
    assert row["recipe_hash"] == "r" * 64


def test_pairings_row_negative_and_radiance_columns() -> None:
    row = pairings_row(
        _product(embedding_method="radiance"),
        _params(negative=True),
        _recipe("radiance"),
        "/out/x.zarr",
    )
    assert math.isnan(row["source_y"]) and math.isnan(row["source_x"])
    assert row["les_case"] == "none:unit-000000"
    assert row["source_id"] == "none"
    assert row["emission_rate_kg_h"] == 0.0
    assert row["unit_absorption_provider"] == "toy"
    assert row["mf_statistics_scope"] == "window"


def test_write_read_round_trip(tmp_path: Path) -> None:
    rows = [
        pairings_row(
            _product(injected_source_y=1, injected_source_x=2),
            _params(),
            _recipe(),
            "/out/a.nc",
        ),
        pairings_row(
            _product(),
            _params(negative=True),
            _recipe(),
            "/out/b.nc",
        ),
    ]
    path = write_pairings(rows, tmp_path / "pairings.csv")
    table = read_pairings(path)
    assert list(table.columns) == list(PAIRINGS_COLUMNS)
    assert table["sample_id"].dtype == "string"
    assert table["unit_absorption_provider"].tolist() == ["", ""]
    assert table["source_y"].tolist()[0] == 1.0
    assert math.isnan(table["source_y"].tolist()[1])
    with pytest.raises(ContractError, match="lacks pairings columns"):
        pd.DataFrame({"sample_id": ["a"]}).to_csv(tmp_path / "bad.csv", index=False)
        read_pairings(tmp_path / "bad.csv")


def test_group_independence_rule() -> None:
    frame = pd.DataFrame(
        {
            "background_group": ["g1", "g1", "g2"],
            "les_case": ["c1", "c2", "c2"],
            "split": ["train", "train", "val"],
            "sensor": ["toy", "toy", "toy"],
        }
    )
    with pytest.raises(ContractError, match="les_case values cross"):
        assert_group_independence(frame)
    frame["les_case"] = ["c1", "c1", "c2"]
    assert_group_independence(frame)
    frame["sensor"] = ["toy", "other", "toy"]
    with pytest.raises(ContractError, match="span more than one sensor"):
        assert_group_independence(frame)
