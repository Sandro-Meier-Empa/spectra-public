"""Merge pairings shards into the dataset pairings table."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import spectra
from spectra.contracts.errors import ContractError
from spectra.export.pairings import (
    assert_group_independence,
    pairings_frame,
    read_pairings,
    write_pairings,
)
from spectra.recipe.generate import RECIPE_HASH_FILE, SHARDS_DIR
from spectra.recipe.params import CONTEXT_FILE, PARAMS_FILE, read_parameter_table
from spectra.recipe.schema import Recipe, recipe_hash

PAIRINGS_FILE = "pairings.csv"
DATASET_FILE = "dataset.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def merge_shards(out_dir: str | Path) -> pd.DataFrame:
    """Concatenate ``shards/pairings_*.csv`` in sample order and check completeness.

    Raises:
        ContractError: If there are no shards, duplicated or missing samples,
            or a group crosses splits.
    """
    out = Path(out_dir)
    shards = sorted((out / SHARDS_DIR).glob("pairings_*.csv"))
    if not shards:
        raise ContractError(f"No pairings shards below {out / SHARDS_DIR}.")
    table = pd.concat([read_pairings(path) for path in shards], ignore_index=True)
    params = read_parameter_table(out / PARAMS_FILE)
    duplicated = table.loc[table["sample_id"].duplicated(), "sample_id"]
    if len(duplicated):
        raise ContractError(
            f"Samples appear in several shards: {sorted(set(duplicated))[:10]}."
        )
    expected = list(params["sample_id"].astype(str))
    present = set(table["sample_id"].astype(str))
    missing = [sample_id for sample_id in expected if sample_id not in present]
    if missing:
        raise ContractError(
            f"{len(missing)} of {len(expected)} samples have no pairings row "
            f"(first: {missing[:5]}); generate the missing indices first."
        )
    unknown = sorted(present - set(expected))
    if unknown:
        raise ContractError(f"Pairings rows of unknown samples: {unknown[:5]}.")
    ordered = table.set_index("sample_id").loc[expected].reset_index()
    assert_group_independence(ordered)
    return pairings_frame(ordered.to_dict(orient="records"))


def finalize(recipe: Recipe, out_dir: str | Path) -> Path:
    """Write ``pairings.csv`` and ``dataset.json`` of a completed run.

    Args:
        recipe: Recipe of the run.
        out_dir: Run directory holding the dry-run files and the shards.

    Returns:
        The pairings table path.
    """
    out = Path(out_dir)
    stored = (out / RECIPE_HASH_FILE).read_text().strip()
    if stored != recipe_hash(recipe):
        raise ContractError(
            f"{out / RECIPE_HASH_FILE} holds {stored}, but the recipe hashes to "
            f"{recipe_hash(recipe)}."
        )
    table = merge_shards(out)
    pairings = write_pairings(table.to_dict(orient="records"), out / PAIRINGS_FILE)
    context = json.loads((out / CONTEXT_FILE).read_text())
    per_split = {
        str(split): int(count)
        for split, count in table["split"].value_counts().sort_index().items()
    }
    summary: dict[str, Any] = {
        "name": recipe.name,
        "dataset_version": recipe.output.dataset_version,
        "recipe_hash": stored,
        "context_hash": str(context["context_hash"]),
        "n_samples": len(table),
        "n_samples_per_split": per_split,
        "n_negatives": int((table["source_id"] == "none").sum()),
        "embedding_method": recipe.method,
        "sensor": recipe.sensor.name,
        "output_root": recipe.output.root,
        "pairings": pairings.name,
        "pairings_sha256": _file_sha256(pairings),
        "spectra_version": spectra.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / DATASET_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return pairings


__all__ = ["DATASET_FILE", "PAIRINGS_FILE", "finalize", "merge_shards"]
