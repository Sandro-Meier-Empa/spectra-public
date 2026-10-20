"""Command-line interface of SPECTRA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

import spectra
from spectra.api import embed, resolve_run
from spectra.contracts import validate_lut, validate_plume_column
from spectra.contracts.errors import ContractError
from spectra.pipeline import open_plumes, resolve_plumes, resolved_plume
from spectra.run import dump_run
from spectra.sensors import SENSORS, get_sensor
from spectra.sensors.native import resolve_files

CONTRACT_VALIDATORS = {
    "plume-column": validate_plume_column,
    "lut": validate_lut,
}


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def cmd_embed(args: argparse.Namespace) -> None:
    """Embed the plumes of a run file and write the native output file."""
    try:
        if args.dry_run:
            config = resolve_run(args.run, output=args.output, seed=args.seed)
            adapter = get_sensor(config.sensor)
            scene = adapter.open_scene(resolve_files(adapter.ROLES, config.files))
            anchors = resolve_plumes(
                scene,
                config.plumes,
                seed=int(config.seed or 0),
                datasets=open_plumes(config.plumes),
            )
            plumes = tuple(
                resolved_plume(plume, anchor)
                for plume, anchor in zip(config.plumes, anchors, strict=True)
            )
            print(dump_run(config.replace(plumes=plumes)), end="")
            return
        outcome = embed(
            args.run,
            output=args.output,
            seed=args.seed,
            overwrite=True if args.overwrite else None,
        )
    except (ContractError, FileExistsError, NotImplementedError) as error:
        _fail(str(error))
        return
    print(f"Wrote {outcome.output}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a file against a SPECTRA contract."""
    path = Path(args.path)
    if not path.exists():
        _fail(f"{path} does not exist.")
    with xr.open_dataset(path) as dataset:
        try:
            info = CONTRACT_VALIDATORS[args.contract](dataset)
        except ContractError as error:
            _fail(f"{path} violates the {args.contract} contract: {error}")
            return
    print(f"{path}: valid {args.contract}")
    if info is not None:
        print(info)


def cmd_sensors(args: argparse.Namespace) -> None:
    """List the built-in sensors and their file roles."""
    del args
    for name, adapter in sorted(SENSORS.items()):
        roles = ", ".join(
            role if required else f"{role} (optional)"
            for role, required in adapter.ROLES.items()
        )
        print(f"{name:8s} {adapter.platform:10s} files: {roles}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectra", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"spectra {spectra.__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "embed", help="Embed the plumes of a run file into its scene."
    )
    run.add_argument("run", help="YAML run file.")
    run.add_argument("--output", default=None, help="Override the output path.")
    run.add_argument("--seed", type=int, default=None, help="Override the seed.")
    run.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output."
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the anchors and print the resolved run file; write nothing.",
    )
    run.set_defaults(func=cmd_embed)

    validate = subparsers.add_parser(
        "validate", help="Validate a file against a contract."
    )
    validate.add_argument("path")
    validate.add_argument(
        "--contract", choices=sorted(CONTRACT_VALIDATORS), required=True
    )
    validate.set_defaults(func=cmd_validate)

    sensors = subparsers.add_parser("sensors", help="List sensors and file roles.")
    sensors.set_defaults(func=cmd_sensors)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the SPECTRA command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
