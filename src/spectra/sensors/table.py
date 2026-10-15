"""Scene tables: explicit file paths for every scene and role.

SPECTRA never derives file locations from directory layouts or file names.
A sensor adapter declares the roles it needs (``radiance``, ``metadata``, ...)
and the user lists the files per scene in a table: a CSV with one row per
scene, a YAML list of mappings, or in-memory rows. Required roles must be
present and the files must exist; optional roles may be empty.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from spectra.contracts.errors import ContractError

SCENE_ID_COLUMN = "scene_id"
BACKGROUND_GROUP_COLUMN = "background_group"
_RESERVED = (SCENE_ID_COLUMN, BACKGROUND_GROUP_COLUMN)


@dataclass(frozen=True)
class SceneTableRow:
    """One scene of a scene table with resolved absolute file paths.

    Attributes:
        scene_id: Scene identifier; defaults to the stem of the first
            required role's file.
        uris: Role name to absolute file path, roles with empty cells omitted.
        background_group: Leakage group given in the table, if any.
        extra: Remaining table columns, as strings.
    """

    scene_id: str
    uris: dict[str, str]
    background_group: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


def _load_rows(
    source: str | Path | Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], Path | None]:
    """Return the raw rows and the directory relative paths resolve against."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ContractError(f"Scene table {path} does not exist.")
        if path.suffix.lower() in (".yaml", ".yml"):
            loaded = yaml.safe_load(path.read_text()) or []
            if isinstance(loaded, Mapping) and "scenes" in loaded:
                loaded = loaded["scenes"]
            if not isinstance(loaded, list):
                raise ContractError(
                    f"Scene table {path} must be a YAML list of mappings "
                    "(optionally under a 'scenes' key)."
                )
            rows = [dict(item) for item in loaded]
        else:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            rows = [dict(record) for record in frame.to_dict(orient="records")]
        return rows, path.parent.resolve()
    return [dict(item) for item in source], None


def _clean(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in ("nan", "none", "null") else text


def read_scene_table(
    source: str | Path | Sequence[Mapping[str, object]],
    *,
    roles: Mapping[str, bool],
    check_exists: bool = True,
) -> list[SceneTableRow]:
    """Read a scene table and resolve its file paths.

    Args:
        source: CSV path, YAML path, or in-memory rows (mappings).
        roles: Role name to ``required`` flag, as declared by the adapter.
        check_exists: Verify that every listed file exists.

    Returns:
        One row per scene with absolute paths.

    Raises:
        ContractError: If the table is empty, a required role is missing, a
            file does not exist, or scene identifiers repeat.
    """
    rows, base = _load_rows(source)
    if not rows:
        raise ContractError("Scene table is empty.")
    required = [role for role, needed in roles.items() if needed]
    output: list[SceneTableRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        uris: dict[str, str] = {}
        for role in roles:
            text = _clean(raw.get(role))
            if not text:
                continue
            path = Path(text).expanduser()
            if not path.is_absolute() and base is not None:
                path = base / path
            path = path.resolve()
            if check_exists and not path.exists():
                raise ContractError(
                    f"Scene table row {index}: {role} file {path} is missing."
                )
            uris[role] = str(path)
        missing = [role for role in required if role not in uris]
        if missing:
            raise ContractError(
                f"Scene table row {index} lacks the required role(s) {missing}."
            )
        scene_id = _clean(raw.get(SCENE_ID_COLUMN)) or Path(uris[required[0]]).stem
        if scene_id in seen:
            raise ContractError(f"Scene table repeats scene_id {scene_id!r}.")
        seen.add(scene_id)
        group = _clean(raw.get(BACKGROUND_GROUP_COLUMN)) or None
        extra = {
            str(key): _clean(value)
            for key, value in raw.items()
            if key not in roles and key not in _RESERVED and _clean(value)
        }
        output.append(SceneTableRow(scene_id, uris, group, extra))
    return output


def filter_rows(
    rows: Sequence[SceneTableRow], **filters: object
) -> list[SceneTableRow]:
    """Apply ``scene_id`` (string or iterable) and extra-column equality filters."""
    selected = list(rows)
    wanted = filters.pop("scene_id", None)
    if wanted is not None:
        if isinstance(wanted, str):
            ids = {wanted}
        elif isinstance(wanted, Iterable):
            ids = {str(item) for item in wanted}
        else:
            raise ContractError("scene_id filter must be a string or an iterable.")
        selected = [row for row in selected if row.scene_id in ids]
    for key, value in filters.items():
        selected = [row for row in selected if row.extra.get(key) == str(value)]
    return selected


__all__ = [
    "BACKGROUND_GROUP_COLUMN",
    "SCENE_ID_COLUMN",
    "SceneTableRow",
    "filter_rows",
    "read_scene_table",
]
