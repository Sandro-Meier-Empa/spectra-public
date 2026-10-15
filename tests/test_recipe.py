"""Recipe-driven generation on the toy plugins: dry run, determinism, shards."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from spectra.catalog.store import SceneCatalog
from spectra.contracts.errors import ContractError
from spectra.export.pairings import (
    PAIRINGS_COLUMNS,
    assert_group_independence,
    read_pairings,
)
from spectra.plume.io import write_plume_column
from spectra.protocols import SensorAdapter
from spectra.recipe import (
    assign_splits,
    build_context,
    build_plugins,
    derive_sample_params,
    dry_run,
    finalize,
    generate,
    generate_sample,
    load_recipe,
    lut_coverage,
    merge_shards,
    recipe_from_mapping,
    recipe_hash,
)
from spectra.recipe.generate import RunCache
from spectra.recipe.params import GenerationContext, params_from_table
from spectra.sensors.toy import ToySensorAdapter
from spectra.testing import make_plume_column

ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
SPECTRA_EXE = Path(sys.executable).parent / "spectra"
# Two LES cases and three toy scenes: with 0.5/0.5/0 ratios every positive
# split receives one case and at least one scene.
RATIOS = {"train": 0.5, "val": 0.5, "test": 0.0}


@pytest.fixture(scope="module")
def toy_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("recipe")
    plumes = []
    for case, seed in (("caseA", 1), ("caseB", 2)):
        dataset = make_plume_column(
            nx=48, ny=48, dx=2.0, n_time=3, case_id=case, seed=seed
        )
        plumes.append(str(write_plume_column(dataset, root / f"{case}.plumecolumn.nc")))
    toy = ToySensorAdapter()
    adapter = cast(SensorAdapter, toy)
    # The toy scenes need no files: catalogue the records directly.
    catalog = SceneCatalog.build(
        adapter, None, records=toy.list_scenes(), directory=root / "catalog"
    )
    for record in catalog.records():
        catalog.set_status(record.scene_id, "qa_done", reason="test", by="test")
        catalog.set_status(record.scene_id, "eligible", reason="test", by="test")
    catalog.write()
    return {"root": root, "plumes": plumes, "catalog": root / "catalog"}


def product_mapping(env: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "name": "toyprod",
        "seed": 7,
        "n_samples": 6,
        "method": "product",
        "sensor": {"name": "toy"},
        "catalog": {"path": str(env["catalog"])},
        "plumes": {
            "files": [str(env["root"] / "*.plumecolumn.nc")],
            "emission": {"distribution": "loguniform", "low": 100, "high": 2000},
        },
        "negatives": {"fraction": 0.34},
        "split": {"ratios": dict(RATIOS)},
        "output": {"root": str(env["root"] / "dataset"), "dataset_version": "t"},
    }
    mapping.update(overrides)
    return mapping


def radiance_mapping(env: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "name": "toyrad",
        "seed": 3,
        "n_samples": 2,
        "method": "radiance",
        # No catalogue: sensor.scenes is required, the toy adapter ignores it.
        "sensor": {"name": "toy", "scenes": "unused"},
        "catalog": {"path": None},
        "plumes": {
            "files": list(env["plumes"]),
            "emission": {"distribution": "uniform", "low": 500, "high": 1500},
        },
        "embedding": {"unit_absorption": {"provider": "toy"}},
        "retrieval": {
            "retriever": "toy",
            "statistics_scope": "window",
            "min_n_samples": 1,
        },
        "negatives": {"fraction": 0.5},
        "split": {"ratios": dict(RATIOS)},
        "output": {
            "root": str(env["root"] / "dataset_rad"),
            "dataset_version": "t",
            "product_format": "zarr",
        },
        "compute": {
            "slurm": {
                "account": "s1234",
                "run_time": "01:00:00",
                "array_size": 2,
                "tasks_per_job": 2,
            }
        },
    }
    mapping.update(overrides)
    return mapping


@pytest.fixture(scope="module")
def product_recipe_path(toy_env: dict[str, Any]) -> Path:
    path = toy_env["root"] / "product.yaml"
    path.write_text(yaml.safe_dump(product_mapping(toy_env)))
    return path


@pytest.fixture(scope="module")
def radiance_recipe_path(toy_env: dict[str, Any]) -> Path:
    path = toy_env["root"] / "radiance.yaml"
    path.write_text(yaml.safe_dump(radiance_mapping(toy_env)))
    return path


# --- schema -----------------------------------------------------------------


def test_load_recipe_fills_defaults_and_resolves_paths(
    product_recipe_path: Path, toy_env: dict[str, Any]
) -> None:
    recipe = load_recipe(product_recipe_path)
    assert recipe.method == "product"
    assert recipe.retrieval is None
    assert recipe.embedding.regridder == "polygon"
    assert recipe.placement.rotations_deg.mode == "uniform_int"
    assert recipe.plumes.emission.units == "kg h-1"
    assert recipe.plumes.emission.maximum_kg_s == pytest.approx(2000 / 3600)
    assert Path(recipe.catalog.path or "").is_absolute()
    assert len(recipe_hash(recipe)) == 64


def test_recipe_validation_names_keys(toy_env: dict[str, Any]) -> None:
    with pytest.raises(ContractError, match="retrieval must be omitted"):
        recipe_from_mapping(product_mapping(toy_env, retrieval={"retriever": "toy"}))
    with pytest.raises(ContractError, match=r"split\.ratios"):
        recipe_from_mapping(
            product_mapping(toy_env, split={"ratios": {"train": 0.9, "val": 0.2}})
        )
    with pytest.raises(ContractError, match="n_samples"):
        recipe_from_mapping(product_mapping(toy_env, n_samples=0))
    with pytest.raises(ContractError, match="unknown keys"):
        recipe_from_mapping(product_mapping(toy_env, bogus=1))
    with pytest.raises(ContractError, match=r"plumes\.emission\.low"):
        recipe_from_mapping(
            product_mapping(
                toy_env,
                plumes={"files": ["x"], "emission": {"distribution": "uniform"}},
            )
        )
    with pytest.raises(ContractError, match="unit_absorption is required"):
        recipe_from_mapping(radiance_mapping(toy_env, embedding={}))
    with pytest.raises(ContractError, match=r"sensor\.scenes .* catalog\.path"):
        recipe_from_mapping(radiance_mapping(toy_env, sensor={"name": "toy"}))
    with pytest.raises(ContractError, match=r"unknown keys \['root'\]"):
        recipe_from_mapping(
            radiance_mapping(toy_env, sensor={"name": "toy", "root": "unused"})
        )
    assert Path(
        recipe_from_mapping(radiance_mapping(toy_env), base=Path("/base")).sensor.scenes
        or ""
    ) == Path("/base/unused")


def test_assign_splits_is_deterministic_and_fills_positive_splits() -> None:
    groups = [f"g{i}" for i in range(10)]
    first = assign_splits(groups, {"train": 0.7, "val": 0.15, "test": 0.15}, 1)
    second = assign_splits(
        reversed(groups), {"train": 0.7, "val": 0.15, "test": 0.15}, 1
    )
    assert first == second
    assert set(first.values()) == {"train", "val", "test"}
    assert list(first.values()).count("train") == 7
    assert assign_splits(groups, {"train": 0.7, "val": 0.15, "test": 0.15}, 2) != first
    three = assign_splits(["a", "b", "c"], {"train": 0.7, "val": 0.15, "test": 0.15}, 0)
    assert sorted(three.values()) == ["test", "train", "val"]
    two = assign_splits(["a", "b"], RATIOS, 0)
    assert sorted(two.values()) == ["train", "val"]


# --- dry run and parameters --------------------------------------------------


@pytest.fixture(scope="module")
def product_run(product_recipe_path: Path, toy_env: dict[str, Any]) -> Path:
    recipe = load_recipe(product_recipe_path)
    return dry_run(recipe, toy_env["root"] / "run_product")


def test_dry_run_writes_files_and_coverage(
    product_run: Path, product_recipe_path: Path
) -> None:
    for name in (
        "context.json",
        "params.parquet",
        "recipe.resolved.yaml",
        "recipe_hash.txt",
        "lut_coverage.json",
    ):
        assert (product_run / name).exists()
    recipe = load_recipe(product_recipe_path)
    assert (product_run / "recipe_hash.txt").read_text().strip() == recipe_hash(recipe)
    resolved = load_recipe(product_run / "recipe.resolved.yaml")
    assert recipe_hash(resolved) == recipe_hash(recipe)
    params = pd.read_parquet(product_run / "params.parquet")
    assert len(params) == 6
    assert list(params["index"]) == list(range(6))
    assert set(params["split"]) <= {"train", "val"}
    assert params["is_negative"].sum() >= 1
    negatives = params.loc[params["is_negative"].astype(bool)]
    positives = params.loc[~params["is_negative"].astype(bool)]
    assert (negatives["emission_scale"] == 0).all()
    assert negatives["les_case"].str.startswith("none:").all()
    assert (negatives["source_id"] == "none").all()
    rates_kg_h = positives["emission_rate_kg_s"] * 3600
    assert rates_kg_h.between(100, 2000).all()
    coverage = json.loads((product_run / "lut_coverage.json").read_text())
    assert len(coverage["plumes"]) == 2
    assert all(entry["covered"] for entry in coverage["plumes"])
    context = GenerationContext.read(product_run / "context.json")
    assert len(context.records) == 3
    assert {p.case_id for p in context.plumes} == {"caseA", "caseB"}
    assert sorted(context.case_splits.values()) == ["train", "val"]


def test_lut_coverage_rejects_small_tables(toy_env: dict[str, Any]) -> None:
    recipe = recipe_from_mapping(
        radiance_mapping(
            toy_env,
            embedding={
                "unit_absorption": {"provider": "toy"},
                "lut": {"enhancement_min_ppm": 0.001, "enhancement_max_ppm": 0.01},
            },
        )
    )
    context = build_context(recipe)
    with pytest.raises(ContractError, match="extend the LUT nodes"):
        lut_coverage(recipe, context)


def test_derive_sample_params_is_deterministic(
    product_recipe_path: Path, product_run: Path
) -> None:
    recipe = load_recipe(product_recipe_path)
    context = GenerationContext.read(product_run / "context.json")
    first = derive_sample_params(recipe, context, 2).to_dict()
    assert derive_sample_params(recipe, context, 2).to_dict() == first
    table = pd.read_parquet(product_run / "params.parquet")
    assert params_from_table(table, 2).to_dict() == first
    code = (
        "import json\n"
        "from spectra.recipe import load_recipe, derive_sample_params\n"
        "from spectra.recipe.params import GenerationContext\n"
        f"recipe = load_recipe({str(product_recipe_path)!r})\n"
        f"context = GenerationContext.read({str(product_run / 'context.json')!r})\n"
        "print(json.dumps(derive_sample_params(recipe, context, 2).to_dict()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=ENV
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == json.loads(
        json.dumps(first)
    )
    reseeded = dataclasses.replace(recipe, seed=8)
    other = derive_sample_params(reseeded, build_context(reseeded), 2).to_dict()
    drawn = ("anchor_along", "anchor_across", "rotation_deg", "emission_scale")
    assert any(other[key] != first[key] for key in drawn)
    assert other["params_hash"] != first["params_hash"]


# --- generation ----------------------------------------------------------------


def test_sharded_generation_matches_serial(
    product_recipe_path: Path, product_run: Path, toy_env: dict[str, Any]
) -> None:
    recipe = load_recipe(product_recipe_path)
    plugins = build_plugins(recipe)
    generate(recipe, product_run, index_range=(0, 3), shard=0, plugins=plugins)
    generate(recipe, product_run, index_range=(3, 6), shard=1, plugins=plugins)
    assert (product_run / "shards" / "pairings_0.csv").exists()
    assert (product_run / "shards" / "pairings_1.csv").exists()
    serial_run = toy_env["root"] / "run_product_serial"
    serial = generate(recipe, serial_run, overwrite=True, plugins=plugins)
    sharded = merge_shards(product_run).drop(columns=["product_uri"])
    expected = read_pairings(serial).drop(columns=["product_uri"])
    pd.testing.assert_frame_equal(
        sharded.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_finalize_pairings_and_prism_build(
    product_recipe_path: Path, product_run: Path, toy_env: dict[str, Any]
) -> None:
    recipe = load_recipe(product_recipe_path)
    pairings = finalize(recipe, product_run)
    assert pairings == product_run / "pairings.csv"
    table = read_pairings(pairings)
    assert list(table.columns) == list(PAIRINGS_COLUMNS)
    assert len(table) == 6
    assert_group_independence(table)
    assert (table["sensor"] == "toy").all()
    assert (table["embedding_method"] == "product").all()
    assert (table["pixel_size_m"] == 10.0).all()
    negatives = table.loc[table["source_id"] == "none"]
    positives = table.loc[table["source_id"] != "none"]
    assert len(negatives) >= 1 and len(positives) >= 1
    assert bool(negatives["source_y"].isna().all())
    assert bool(positives["source_y"].notna().all())
    assert (table["emission_rate_kg_h"] >= 0).all()
    summary = json.loads((product_run / "dataset.json").read_text())
    assert summary["n_samples"] == 6
    assert summary["recipe_hash"] == recipe_hash(recipe)
    assert sum(summary["n_samples_per_split"].values()) == 6

    for row in positives.to_dict(orient="records"):
        product = xr.open_dataset(str(row["product_uri"]))
        injected = product["injected_ch4_mass"].values
        y, x = int(row["source_y"]), int(row["source_x"])
        assert 0 <= y < injected.shape[0] and 0 <= x < injected.shape[1]
        assert float(injected[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].max()) > 0
        assert product.attrs["les_case"] == row["les_case"]
        assert product.attrs["recipe_hash"] == recipe_hash(recipe)
        product.close()

    pytest.importorskip("prism")
    from prism.cli.prepare_dataset import build_dataset

    manifest = build_dataset(
        pairings,
        toy_env["root"] / "prism_product",
        dataset_version="t",
        detection_snr=2.0,
        overwrite=True,
    )
    assert manifest.sensors == ("toy",)
    assert len(manifest.frame) == 6
    assert set(manifest.frame["embedding_method"]) == {"product"}


def test_negative_sample_has_zero_injected_column(
    product_recipe_path: Path, product_run: Path
) -> None:
    recipe = load_recipe(product_recipe_path)
    context = GenerationContext.read(product_run / "context.json")
    table = pd.read_parquet(product_run / "params.parquet")
    index = int(table.loc[table["is_negative"], "index"].iloc[0])
    params = params_from_table(table, index)
    assert params.is_negative and params.emission_scale == 0.0
    assert params.placement.emission_scale == 1.0
    sample = generate_sample(
        params, recipe, context, plugins=build_plugins(recipe), overwrite=True
    )
    assert sample.row["les_case"] == f"none:{params.sample_id}"
    assert np.isnan(sample.row["source_y"])
    product = xr.open_dataset(sample.path)
    assert float(product["injected_ch4_mass"].max()) == 0.0
    assert product.attrs["embedding_method"] == "product"
    assert product.attrs["emission_scale"] == 0.0
    assert product["ch4_mass"].shape == (
        params.window.across.stop - params.window.across.start,
        params.window.along.stop - params.window.along.start,
    )
    product.close()


def test_radiance_recipe_end_to_end(
    radiance_recipe_path: Path, toy_env: dict[str, Any]
) -> None:
    recipe = load_recipe(radiance_recipe_path)
    assert recipe.retrieval is not None
    out = toy_env["root"] / "run_radiance"
    plugins = build_plugins(recipe)
    assert plugins.provider is not None and plugins.retriever is not None
    cache = RunCache()
    shard = generate(recipe, out, plugins=plugins)
    cache.close()
    table = read_pairings(shard)
    assert len(table) == 2
    assert (table["embedding_method"] == "radiance").all()
    assert (table["retriever"] == "toy").all()
    assert (table["unit_absorption_provider"] == "toy").all()
    assert (table["mf_statistics_scope"] == "window").all()
    pairings = finalize(recipe, out)
    assert_group_independence(read_pairings(pairings))
    for row in table.to_dict(orient="records"):
        product = xr.open_zarr(str(row["product_uri"]))
        assert product["ch4_mass"].dims == ("y", "x")
        assert {"wind_u", "wind_v", "injected_ch4_mass"} <= set(product.data_vars)
        assert product.attrs["embedding_version"] == "lambert-beer-v1"
        if row["source_id"] != "none":
            assert float(product["injected_ch4_mass"].max()) > 0
        product.close()


def test_cli_dry_run_and_finalize(
    product_recipe_path: Path, product_run: Path, toy_env: dict[str, Any]
) -> None:
    if not SPECTRA_EXE.exists():
        pytest.skip("spectra executable is not installed next to the interpreter")
    out = toy_env["root"] / "run_cli"
    result = subprocess.run(
        [
            str(SPECTRA_EXE),
            "generate",
            str(product_recipe_path),
            "--out",
            str(out),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=ENV,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "params.parquet").exists()
    result = subprocess.run(
        [
            str(SPECTRA_EXE),
            "finalize",
            str(product_recipe_path),
            "--out",
            str(product_run),
        ],
        capture_output=True,
        text=True,
        env=ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "pairings.csv" in result.stdout
    result = subprocess.run(
        [str(SPECTRA_EXE), "finalize", str(product_recipe_path), "--out", str(out)],
        capture_output=True,
        text=True,
        env=ENV,
    )
    assert result.returncode == 1
    assert "No pairings shards" in result.stderr


def test_slurm_array_script(
    radiance_recipe_path: Path, toy_env: dict[str, Any]
) -> None:
    pytest.importorskip("cscs")
    from spectra.recipe.slurm import shard_layout, write_slurm_array

    recipe = load_recipe(radiance_recipe_path)
    out = toy_env["root"] / "run_slurm"
    script = write_slurm_array(recipe, out, python="python")
    text = script.read_text()
    assert shard_layout(recipe) == (4, 1)
    assert "#SBATCH --array=0-1" in text
    assert "#SBATCH --account=s1234" in text
    assert "#SBATCH --time=01:00:00" in text
    assert f"#SBATCH --chdir={out.resolve()}" in text
    assert "python -m spectra.cli generate" in text
    assert '--index-range "$START:$STOP" --shard "$SHARD"' in text
    assert "TASKS=2" in text
    assert (out / "recipe.resolved.yaml").exists()
    assert recipe_hash(load_recipe(out / "recipe.resolved.yaml")) == recipe_hash(recipe)
