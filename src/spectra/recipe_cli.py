"""Catalogue and recipe subcommands of the ``spectra`` command line."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from spectra.contracts.errors import ContractError, MissingExtraError

if TYPE_CHECKING:
    from spectra.recipe.schema import Recipe


def _fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _parse_options(values: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` options, decoding JSON values when possible."""
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


def _parse_index_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        start, stop = (int(part) for part in value.split(":", 1))
    except ValueError:
        _fail(f"--index-range must have the form START:STOP; got {value!r}.")
        return None
    if start < 0 or stop <= start:
        _fail("--index-range needs 0 <= START < STOP.")
    return start, stop


def _load(path: str) -> Recipe:
    from spectra.recipe.schema import load_recipe

    try:
        return load_recipe(path)
    except (ContractError, FileNotFoundError) as error:
        _fail(f"Invalid recipe {path}: {error}")


def _out_dir(args: argparse.Namespace, recipe: Recipe) -> Path:
    return Path(args.out) if args.out else Path(recipe.output.root)


# -- catalog -----------------------------------------------------------------
def cmd_catalog_build(args: argparse.Namespace) -> None:
    from spectra.catalog.store import SceneCatalog
    from spectra.registry import get_sensor

    adapter = get_sensor(args.sensor, **_parse_options(args.adapter_option))
    catalog = SceneCatalog.build(
        adapter,
        args.scenes,
        inspect=not args.no_inspect,
        directory=args.out,
        **_parse_options(args.filter),
    )
    catalog.write(args.out)
    print(f"Wrote {catalog!r} to {args.out}")


def cmd_catalog_qa(args: argparse.Namespace) -> None:
    from spectra.catalog.qa import QaSettings, compute_scene_qa
    from spectra.catalog.schema import QaThresholds
    from spectra.catalog.store import SceneCatalog
    from spectra.registry import get_sensor

    catalog = SceneCatalog.read(args.catalog)
    frame = catalog.frame
    scene_ids = list(args.scene_id or frame["scene_id"].astype(str))
    adapters: dict[str, Any] = {}
    settings = QaSettings(
        subsample=args.subsample, detection="none" if args.no_detection else "snr"
    )
    for scene_id in scene_ids:
        row = catalog.row(scene_id)
        if row.sensor not in adapters:
            adapters[row.sensor] = get_sensor(
                row.sensor, **_parse_options(args.adapter_option)
            )
        adapter = adapters[row.sensor]
        from spectra.catalog.schema import row_to_record

        record = row_to_record(row)
        scene = adapter.open_scene(record)
        product = None
        if row.has_product and hasattr(adapter, "open_product"):
            product = adapter.open_product(record)
        result = compute_scene_qa(scene, product, settings)
        catalog.update_qa(
            scene_id, result, qa_version=args.qa_version, force=args.force
        )
        print(
            f"{scene_id}: valid={result.valid_fraction:.3f} "
            f"cloud={result.cloud_fraction:.3f} detections={result.n_detections}"
        )
    outcomes = catalog.apply_thresholds(QaThresholds())
    catalog.write()
    for scene_id, status in outcomes.items():
        print(f"{scene_id}: {status}")
    print(f"Updated {catalog!r}")


def cmd_catalog_status(args: argparse.Namespace) -> None:
    from spectra.catalog.schema import CatalogError
    from spectra.catalog.store import SceneCatalog

    catalog = SceneCatalog.read(args.catalog)
    try:
        catalog.set_status(args.scene_id, args.set, reason=args.reason, by=args.by)
    except CatalogError as error:
        _fail(str(error))
    catalog.write()
    print(f"{args.scene_id}: {args.set} ({args.reason}, by {args.by})")


# -- recipes -----------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    from spectra.recipe.generate import dry_run, generate

    recipe = _load(args.recipe)
    out = _out_dir(args, recipe)
    try:
        if args.dry_run:
            dry_run(recipe, out)
            print(f"Dry run written to {out}")
            return
        shard = generate(
            recipe,
            out,
            index_range=_parse_index_range(args.index_range),
            shard=args.shard,
            overwrite=args.overwrite,
        )
    except ContractError as error:
        _fail(str(error))
        return
    print(f"Wrote {shard}")


def cmd_finalize(args: argparse.Namespace) -> None:
    from spectra.recipe.finalize import finalize

    recipe = _load(args.recipe)
    out = _out_dir(args, recipe)
    try:
        pairings = finalize(recipe, out)
    except ContractError as error:
        _fail(str(error))
        return
    print(f"Wrote {pairings}")


def cmd_slurm(args: argparse.Namespace) -> None:
    from spectra.recipe.slurm import usage_hint, write_slurm_array

    recipe = _load(args.recipe)
    out = _out_dir(args, recipe)
    try:
        script = write_slurm_array(recipe, out, python=args.python)
    except (ContractError, MissingExtraError) as error:
        _fail(str(error))
        return
    print(usage_hint(recipe, script, out.resolve()))


def cmd_export_prism(args: argparse.Namespace) -> None:
    from spectra.recipe.finalize import PAIRINGS_FILE, finalize

    recipe = _load(args.recipe)
    out = _out_dir(args, recipe)
    pairings = out / PAIRINGS_FILE
    if not pairings.exists():
        try:
            pairings = finalize(recipe, out)
        except ContractError as error:
            _fail(str(error))
            return
    prism_root = (
        Path(args.prism_root)
        if args.prism_root
        else out / "prism" / recipe.output.dataset_version
    )
    command = [
        "prism-compose-dataset",
        str(pairings),
        str(prism_root),
        "--dataset-version",
        recipe.output.dataset_version,
        "--detection-snr",
        str(recipe.labels.detection_snr),
        "--ambiguous-snr",
        str(recipe.labels.ambiguous_snr),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if not args.compose:
        print(f"Pairings: {pairings}")
        print("Compose with: " + " ".join(command))
        return
    # Prefer the console script of the running interpreter: background shells
    # on the cluster often lack the virtual environment on PATH.
    sibling = Path(sys.executable).parent / command[0]
    executable = str(sibling) if sibling.exists() else shutil.which(command[0])
    if executable is None:
        _fail(
            "prism-compose-dataset is not on PATH; install prism (pip install prism)."
        )
        return
    subprocess.run([executable, *command[1:]], check=True)
    print(f"Composed prism dataset in {prism_root}")


# -- registration --------------------------------------------------------------
def _add_recipe_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("recipe", help="Recipe YAML.")
    parser.add_argument(
        "--out",
        default=None,
        help="Run directory (dry-run files, shards, pairings); default output.root.",
    )


def register_recipe_commands(subparsers: argparse._SubParsersAction) -> None:
    """Add the catalogue and recipe subcommands to the ``spectra`` parser."""
    catalog = subparsers.add_parser(
        "catalog", help="Build and curate a scene catalogue."
    )
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)

    build = catalog_sub.add_parser(
        "build", help="Catalogue the scenes of an explicit scene table."
    )
    build.add_argument("--sensor", required=True, help="Registered sensor adapter.")
    build.add_argument(
        "--scenes",
        required=True,
        help="Scene table (CSV/YAML) listing the full path of every file per scene.",
    )
    build.add_argument("--out", required=True, help="Catalogue directory.")
    build.add_argument(
        "--filter", action="append", metavar="KEY=VALUE", help="list_scenes filter."
    )
    build.add_argument(
        "--adapter-option", action="append", metavar="KEY=VALUE", help="Adapter option."
    )
    build.add_argument(
        "--no-inspect", action="store_true", help="Do not open scenes for sizes."
    )
    build.set_defaults(func=cmd_catalog_build)

    qa = catalog_sub.add_parser("qa", help="Run background QA and apply thresholds.")
    qa.add_argument("catalog", help="Catalogue directory.")
    qa.add_argument("--scene-id", action="append", help="Restrict to scene ids.")
    qa.add_argument("--subsample", type=int, default=None, help="Pixel stride.")
    qa.add_argument("--qa-version", default="1", help="QA version label.")
    qa.add_argument("--force", action="store_true", help="Overwrite same version.")
    qa.add_argument("--no-detection", action="store_true", help="Skip detections.")
    qa.add_argument(
        "--adapter-option", action="append", metavar="KEY=VALUE", help="Adapter option."
    )
    qa.set_defaults(func=cmd_catalog_qa)

    status = catalog_sub.add_parser("status", help="Set the status of one scene.")
    status.add_argument("catalog", help="Catalogue directory.")
    status.add_argument("--scene-id", required=True)
    status.add_argument("--set", required=True, help="New status.")
    status.add_argument("--reason", required=True)
    status.add_argument("--by", required=True, help="Who sets the status.")
    status.set_defaults(func=cmd_catalog_status)

    generate = subparsers.add_parser("generate", help="Generate samples of a recipe.")
    _add_recipe_argument(generate)
    generate.add_argument(
        "--dry-run", action="store_true", help="Only resolve context and parameters."
    )
    generate.add_argument("--index-range", default=None, metavar="START:STOP")
    generate.add_argument("--shard", type=int, default=None, help="Shard number.")
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(func=cmd_generate)

    finalize = subparsers.add_parser("finalize", help="Merge shards into pairings.csv.")
    _add_recipe_argument(finalize)
    finalize.set_defaults(func=cmd_finalize)

    slurm = subparsers.add_parser("slurm", help="Write a Slurm array script.")
    _add_recipe_argument(slurm)
    slurm.add_argument(
        "--python", default=sys.executable, help="Interpreter used on the nodes."
    )
    slurm.set_defaults(func=cmd_slurm)

    export = subparsers.add_parser(
        "export-prism", help="Finalize and optionally compose the prism dataset."
    )
    _add_recipe_argument(export)
    export.add_argument(
        "--compose", action="store_true", help="Run prism-compose-dataset."
    )
    export.add_argument("--prism-root", default=None, help="prism dataset folder.")
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(func=cmd_export_prism)


__all__ = ["register_recipe_commands"]
