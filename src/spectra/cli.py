"""Command-line interface of SPECTRA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

import spectra
from spectra.contracts import (
    validate_lut,
    validate_plume_column,
    validate_sensor_scene,
)
from spectra.contracts.errors import ContractError
from spectra.registry import GROUPS, list_plugins

CONTRACT_VALIDATORS = {
    "plume-column": validate_plume_column,
    "sensor-scene": validate_sensor_scene,
    "lut": validate_lut,
}


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the SPECTRA command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
