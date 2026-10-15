"""Command-line interface of SPECTRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import spectra
from spectra.contracts import (
    validate_column_product,
    validate_lut,
    validate_plume_column,
    validate_sensor_scene,
)
from spectra.contracts.errors import ContractError
from spectra.registry import GROUPS, list_plugins
from spectra.protocols import SensorAdapter
from spectra.types import Placement, SceneRecord

CONTRACT_VALIDATORS = {
    "plume-column": validate_plume_column,
    "sensor-scene": validate_sensor_scene,
    "column-product": validate_column_product,
    "lut": validate_lut,
}
ANCHORS = ("lower left", "lower right", "upper left", "upper right")


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _parse_options(values: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` plugin options, decoding JSON values when possible."""
    options: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            _fail(f"Option {item!r} must have the form key=value.")
        key, raw = item.split("=", 1)
        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            options[key] = raw
    return options


# -- plugins -----------------------------------------------------------------
def cmd_plugins(args: argparse.Namespace) -> None:
    for group, entry_group in GROUPS.items():
        print(f"{group} ({entry_group}):")
        for name, point in sorted(list_plugins(group).items()):
            print(f"  {name:18s} {point.value}")


# -- validate ----------------------------------------------------------------
def cmd_validate(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        _fail(f"{path} does not exist.")
    dataset = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    validator = CONTRACT_VALIDATORS[args.contract]
    try:
        info = validator(dataset)
    except ContractError as error:
        _fail(f"{path} violates the {args.contract} contract: {error}")
        return
    print(f"{path}: valid {args.contract}")
    if info is not None:
        print(info)


# -- embed -------------------------------------------------------------------
def _add_common_embed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sensor", default="aviris4", help="Registered sensor adapter."
    )
    parser.add_argument(
        "--file",
        action="append",
        metavar="ROLE=PATH",
        help=(
            "Full path of one scene file by role (the adapter lists its roles, "
            "e.g. radiance, metadata, shadow, cloud, product); repeatable."
        ),
    )
    for role in ("radiance", "metadata", "shadow", "cloud", "product"):
        parser.add_argument(
            f"--{role}",
            default=None,
            metavar="PATH",
            help=f"Shortcut for --file {role}=PATH.",
        )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene identifier; defaults to the radiance stem.",
    )
    parser.add_argument(
        "--background-group",
        default=None,
        help="Leakage group; defaults to sensor:scene_id.",
    )
    parser.add_argument(
        "--adapter-option",
        action="append",
        metavar="KEY=VALUE",
        help="Adapter constructor option (JSON values allowed); repeatable.",
    )
    parser.add_argument(
        "--plume",
        required=True,
        help="PlumeColumn NetCDF/Zarr, or a raw (y, x) column file.",
    )
    parser.add_argument(
        "--column-variable", default=None, help="Variable of a raw column file."
    )
    parser.add_argument(
        "--column-units", default=None, help="Units of a raw column variable."
    )
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--emission-scale", type=float, default=1.0)
    parser.add_argument("--anchor-along", type=int, default=None)
    parser.add_argument("--anchor-across", type=int, default=None)
    parser.add_argument("--anchor-corner", choices=ANCHORS, default="lower left")
    parser.add_argument(
        "--rotation", type=int, default=0, help="Whole degrees, CCW positive."
    )
    parser.add_argument("--flip", choices=("none", "x", "y"), default="none")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed of the random anchor draw."
    )
    parser.add_argument("--min-outside-fraction", type=float, default=0.0)
    parser.add_argument("--max-outside-fraction", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=10_000)
    parser.add_argument("--regridder", default="polygon", help="Registered regridder.")
    parser.add_argument(
        "--regridder-option",
        action="append",
        metavar="KEY=VALUE",
        help="Regridder option.",
    )
    parser.add_argument(
        "--save-path", required=True, help="Output NetCDF (.nc) or Zarr (.zarr)."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prism",
        action="store_true",
        help="Write the output on the prism (y, x) grid through spectra.export.prism.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectra", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"spectra {spectra.__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plugins = subparsers.add_parser("plugins", help="List registered plugins.")
    plugins.set_defaults(func=cmd_plugins)

    validate = subparsers.add_parser(
        "validate", help="Validate a file against a contract."
    )
    validate.add_argument("path")
    validate.add_argument(
        "--contract", choices=sorted(CONTRACT_VALIDATORS), required=True
    )
    validate.set_defaults(func=cmd_validate)

    embed = subparsers.add_parser(
        "embed", help="Embed a plume snapshot into one scene."
    )
    embed_sub = embed.add_subparsers(dest="method", required=True)
    radiance = embed_sub.add_parser(
        "radiance", help="Lambert-Beer into the radiance cube, then retrieve."
    )
    _add_common_embed_arguments(radiance)
    radiance.add_argument("--provider", default="libradtran_arts")
    radiance.add_argument("--uvspec-path", default=None)
    radiance.add_argument("--sol-reference-path", default=None)
    radiance.add_argument("--arts-path", default=None)
    radiance.add_argument("--scenario", default="midlatitude-summer")
    radiance.add_argument("--xch4-bg", type=float, default=None)
    radiance.add_argument(
        "--provider-option",
        action="append",
        metavar="KEY=VALUE",
        help="Provider option.",
    )
    radiance.add_argument("--retriever", default="aviris4_mf")
    radiance.add_argument("--noise-model-file", default=None)
    radiance.add_argument(
        "--retriever-option",
        action="append",
        metavar="KEY=VALUE",
        help="Retriever option.",
    )
    radiance.add_argument("--mf-enhancement", type=float, default=10.0)
    radiance.add_argument("--enhancement-min", type=float, default=0.01)
    radiance.add_argument("--enhancement-max", type=float, default=200.0)
    radiance.add_argument("--enhancement-nodes", type=int, default=12)
    radiance.add_argument("--layer-height", type=float, default=1000.0)
    radiance.add_argument(
        "--column-convention", choices=("vertical", "slant"), default="vertical"
    )
    radiance.add_argument(
        "--band-window",
        nargs=2,
        type=float,
        action="append",
        metavar=("MIN_NM", "MAX_NM"),
        help=(
            "Retrieval wavelength window; repeatable. Default 1480-1800 and 2080-2500."
        ),
    )
    radiance.add_argument("--min-n-samples", type=int, default=7000)
    radiance.add_argument("--serial", action="store_true")
    radiance.set_defaults(func=cmd_embed, method="radiance")

    product = embed_sub.add_parser(
        "product", help="Add the column onto a retrieved product."
    )
    _add_common_embed_arguments(product)
    product.add_argument(
        "--product-path",
        default=None,
        help="Column product; defaults to the adapter's product.",
    )
    product.set_defaults(func=cmd_embed, method="product")

    from spectra.recipe_cli import register_recipe_commands

    register_recipe_commands(subparsers)
    return parser


def _load_column(args: argparse.Namespace) -> xr.DataArray:
    from spectra.pipeline import snapshot_column

    path = Path(args.plume)
    dataset = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    if args.column_variable is None:
        return snapshot_column(
            dataset, time_index=args.time_index, emission_scale=args.emission_scale
        )
    if args.column_variable not in dataset:
        _fail(f"'{args.column_variable}' is not in {path}.")
    column = dataset[args.column_variable]
    if "time" in column.dims:
        column = column.isel(time=args.time_index)
    units = args.column_units or column.attrs.get("units")
    if units is None:
        _fail("Raw columns need --column-units.")
    column = column * args.emission_scale if args.emission_scale != 1.0 else column
    column.attrs["units"] = units
    return column


def _resolve_placement(
    args: argparse.Namespace, scene: xr.Dataset, column: xr.DataArray
) -> Placement:
    from spectra.types import SceneGeometry

    if (args.anchor_along is None) != (args.anchor_across is None):
        _fail("--anchor-along and --anchor-across must be given together.")
    if args.anchor_along is not None:
        return Placement(
            anchor_along=args.anchor_along,
            anchor_across=args.anchor_across,
            anchor_corner=args.anchor_corner,
            rotation_deg=args.rotation,
            flip=args.flip,
            emission_scale=args.emission_scale,
            time_index=args.time_index,
        )
    from spectra.geometry.swath import place_in_scene

    placed = place_in_scene(
        column,
        SceneGeometry.from_scene(scene),
        anchor_corner=args.anchor_corner,
        rotation_deg=args.rotation,
        flip=args.flip,
        min_outside_fraction=args.min_outside_fraction,
        max_outside_fraction=args.max_outside_fraction,
        rng=np.random.default_rng(args.seed),
        max_attempts=args.max_attempts,
    )
    print(
        f"Random anchor along={placed.placement.anchor_along}, "
        f"across={placed.placement.anchor_across} after "
        f"{placed.draw.attempts} draw(s); "
        f"outside mass fraction {placed.draw.outside_fraction:.4f}."
    )
    return Placement(
        anchor_along=placed.placement.anchor_along,
        anchor_across=placed.placement.anchor_across,
        anchor_corner=args.anchor_corner,
        rotation_deg=args.rotation,
        flip=args.flip,
        emission_scale=args.emission_scale,
        time_index=args.time_index,
    )


def _scene_record(adapter: SensorAdapter, args: argparse.Namespace) -> SceneRecord:
    """Build the scene record from the explicit file paths on the command line."""
    files: dict[str, str] = {}
    for item in args.file or []:
        if "=" not in item:
            _fail(f"--file expects ROLE=PATH; got {item!r}.")
        role, path = item.split("=", 1)
        files[role.strip()] = path.strip()
    for role in ("radiance", "metadata", "shadow", "cloud", "product"):
        value = getattr(args, role, None)
        if value:
            files[role] = value
    if not files:
        _fail("Give the scene files with --file ROLE=PATH (or --radiance, ...).")
    row = {
        **files,
        "scene_id": args.scene_id or "",
        "background_group": args.background_group or "",
    }
    try:
        return adapter.list_scenes([row])[0]
    except ContractError as error:
        _fail(str(error))
        raise


def cmd_embed(args: argparse.Namespace) -> None:
    from spectra.export.prism import to_prism_product, write_product
    from spectra.pipeline import EmbeddingSpec, embed_sample, log_nodes
    from spectra.registry import (
        get_regridder,
        get_retriever,
        get_sensor,
        get_unit_absorption_provider,
    )
    from spectra.types import RetrievalSettings

    save_path = Path(args.save_path)
    if save_path.exists() and not args.overwrite:
        _fail(f"{save_path} exists; pass --overwrite to replace it.")

    adapter = get_sensor(args.sensor, **_parse_options(args.adapter_option))
    record = _scene_record(adapter, args)
    scene = adapter.open_scene(record)
    column = _load_column(args)
    placement = _resolve_placement(args, scene, column)
    regridder = get_regridder(args.regridder, **_parse_options(args.regridder_option))

    if args.method == "product":
        if args.product_path is not None:
            product = xr.open_dataset(args.product_path)
        else:
            if not hasattr(adapter, "open_product"):
                _fail(
                    f"Adapter {args.sensor} cannot open products; pass --product-path."
                )
            product = adapter.open_product(record)  # type: ignore[attr-defined]
        result = embed_sample(
            scene,
            column,
            placement,
            method="product",
            regridder=regridder,
            product=product,
        )
    else:
        provider_options = _parse_options(args.provider_option)
        if args.provider == "libradtran_arts":
            for key, value in (
                ("uvspec_path", args.uvspec_path),
                ("sol_reference_path", args.sol_reference_path),
                ("arts_path", args.arts_path),
                ("scenario", args.scenario),
                ("xch4_bg", args.xch4_bg),
            ):
                if value is not None:
                    provider_options.setdefault(key, value)
        provider = get_unit_absorption_provider(args.provider, **provider_options)
        retriever_options = _parse_options(args.retriever_option)
        if args.retriever == "aviris4_mf":
            if args.noise_model_file is not None:
                retriever_options.setdefault("noise_model_file", args.noise_model_file)
            retriever_options.setdefault("parallel", not args.serial)
        retriever = get_retriever(args.retriever, **retriever_options)
        windows = tuple(tuple(window) for window in (args.band_window or []))
        settings = RetrievalSettings(
            statistics_scope="window",
            band_windows_nm=windows or ((1480.0, 1800.0), (2080.0, 2500.0)),
            target_enhancement_ppm=args.mf_enhancement,
            min_n_samples=args.min_n_samples,
        )
        spec = EmbeddingSpec(
            column_convention=args.column_convention,
            enhancements_ppm=log_nodes(
                args.enhancement_min, args.enhancement_max, args.enhancement_nodes
            ),
            layer_height_m=args.layer_height,
        )
        result = embed_sample(
            scene,
            column,
            placement,
            method="radiance",
            regridder=regridder,
            spec=spec,
            provider=provider,
            retriever=retriever,
            settings=settings,
        )
    if args.prism:
        result = to_prism_product(result)
    write_product(result, save_path, overwrite=args.overwrite)
    print(f"Wrote {save_path}")


def main(argv: list[str] | None = None) -> None:
    """Run the SPECTRA command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
