"""Scene tables: explicit per-scene file lists in CSV, YAML or memory."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectra.contracts.errors import ContractError
from spectra.sensors.table import SceneTableRow, filter_rows, read_scene_table

ROLES = {"radiance": True, "metadata": True, "shadow": False, "product": False}


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    """Two scenes worth of empty files below ``tmp_path/data``."""
    data = tmp_path / "data"
    return {
        "rad1": touch(data / "line_0001_rdn.nc"),
        "meta1": touch(data / "line_0001_rdn_metadata.nc"),
        "shadow1": touch(data / "masks" / "line_0001_rdn_shadow.nc"),
        "rad2": touch(data / "line_0002_rdn.nc"),
        "meta2": touch(data / "line_0002_rdn_metadata.nc"),
        "prod2": touch(data / "line_0002_rdn_CH4.nc"),
    }


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_csv_with_absolute_and_relative_paths(
    tmp_path: Path, files: dict[str, Path]
) -> None:
    # Relative paths resolve against the table's directory, not the cwd.
    table = write_csv(
        tmp_path / "tables" / "scenes.csv",
        ["scene_id", "radiance", "metadata", "shadow", "product", "flight"],
        [
            ["one", str(files["rad1"]), "../data/line_0001_rdn_metadata.nc",
             "../data/masks/line_0001_rdn_shadow.nc", "", "A"],
            ["two", "../data/line_0002_rdn.nc", str(files["meta2"]), "",
             str(files["prod2"]), "B"],
        ],
    )  # fmt: skip
    rows = read_scene_table(table, roles=ROLES)
    assert [row.scene_id for row in rows] == ["one", "two"]
    assert rows[0].uris == {
        "radiance": str(files["rad1"].resolve()),
        "metadata": str(files["meta1"].resolve()),
        "shadow": str(files["shadow1"].resolve()),
    }
    assert rows[1].uris == {
        "radiance": str(files["rad2"].resolve()),
        "metadata": str(files["meta2"].resolve()),
        "product": str(files["prod2"].resolve()),
    }
    assert all(Path(uri).is_absolute() for row in rows for uri in row.uris.values())
    assert rows[0].background_group is None
    assert rows[0].extra == {"flight": "A"} and rows[1].extra == {"flight": "B"}


def test_scene_id_defaults_to_first_required_stem(files: dict[str, Path]) -> None:
    rows = read_scene_table(
        [{"radiance": str(files["rad1"]), "metadata": str(files["meta1"])}],
        roles=ROLES,
    )
    assert rows == [
        SceneTableRow(
            "line_0001_rdn",
            {
                "radiance": str(files["rad1"].resolve()),
                "metadata": str(files["meta1"].resolve()),
            },
        )
    ]


def test_yaml_list_and_scenes_key(tmp_path: Path, files: dict[str, Path]) -> None:
    entries = [
        {
            "scene_id": "one",
            "radiance": "line_0001_rdn.nc",
            "metadata": "line_0001_rdn_metadata.nc",
            "shadow": None,
            "background_group": "campaign:day1",
            "altitude_ft": 3000,
        },
        {
            "radiance": str(files["rad2"]),
            "metadata": str(files["meta2"]),
            "product": str(files["prod2"]),
        },
    ]
    plain = tmp_path / "data" / "scenes.yaml"
    plain.write_text(yaml.safe_dump(entries))
    nested = tmp_path / "data" / "nested.yml"
    nested.write_text(yaml.safe_dump({"scenes": entries, "comment": "ignored"}))
    for table in (plain, nested):
        rows = read_scene_table(table, roles=ROLES)
        assert [row.scene_id for row in rows] == ["one", "line_0002_rdn"]
        assert rows[0].uris["radiance"] == str(files["rad1"].resolve())
        assert "shadow" not in rows[0].uris
        assert rows[0].background_group == "campaign:day1"
        assert rows[0].extra == {"altitude_ft": "3000"}
        assert rows[1].background_group is None and rows[1].extra == {}
    wrong = tmp_path / "data" / "wrong.yaml"
    wrong.write_text(yaml.safe_dump({"radiance": "x"}))
    with pytest.raises(ContractError, match="YAML list"):
        read_scene_table(wrong, roles=ROLES)


def test_optional_empty_cells_are_omitted(files: dict[str, Path]) -> None:
    rows = read_scene_table(
        [
            {
                "radiance": str(files["rad1"]),
                "metadata": str(files["meta1"]),
                "shadow": "",
                "product": "  ",
                "note": "",
            },
            {
                "radiance": str(files["rad2"]),
                "metadata": str(files["meta2"]),
                "shadow": "nan",
                "product": None,
            },
        ],
        roles=ROLES,
    )
    assert set(rows[0].uris) == {"radiance", "metadata"}
    assert set(rows[1].uris) == {"radiance", "metadata"}
    assert rows[0].extra == {}


def test_missing_required_role(files: dict[str, Path]) -> None:
    with pytest.raises(ContractError, match=r"row 0 lacks .*\['metadata'\]"):
        read_scene_table([{"radiance": str(files["rad1"])}], roles=ROLES)
    with pytest.raises(ContractError, match=r"row 1 lacks"):
        read_scene_table(
            [
                {"radiance": str(files["rad1"]), "metadata": str(files["meta1"])},
                {"radiance": str(files["rad2"]), "metadata": ""},
            ],
            roles=ROLES,
        )


def test_missing_file_and_check_exists(tmp_path: Path, files: dict[str, Path]) -> None:
    rows = [{"radiance": str(files["rad1"]), "metadata": str(tmp_path / "nope.nc")}]
    with pytest.raises(ContractError, match=r"metadata file .*nope\.nc is missing"):
        read_scene_table(rows, roles=ROLES)
    unchecked = read_scene_table(rows, roles=ROLES, check_exists=False)
    assert unchecked[0].uris["metadata"] == str((tmp_path / "nope.nc").resolve())
    with pytest.raises(ContractError, match="does not exist"):
        read_scene_table(tmp_path / "absent.csv", roles=ROLES)


def test_duplicate_scene_ids_are_rejected(files: dict[str, Path]) -> None:
    row = {"radiance": str(files["rad1"]), "metadata": str(files["meta1"])}
    with pytest.raises(ContractError, match="repeats scene_id 'line_0001_rdn'"):
        read_scene_table([row, dict(row)], roles=ROLES)
    with pytest.raises(ContractError, match="repeats scene_id 'x'"):
        read_scene_table(
            [
                {**row, "scene_id": "x"},
                {
                    "scene_id": "x",
                    "radiance": str(files["rad2"]),
                    "metadata": str(files["meta2"]),
                },
            ],
            roles=ROLES,
        )


def test_empty_table(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="empty"):
        read_scene_table([], roles=ROLES)
    empty = write_csv(tmp_path / "empty.csv", ["scene_id", "radiance"], [])
    with pytest.raises(ContractError, match="empty"):
        read_scene_table(empty, roles=ROLES)


def test_filter_rows(files: dict[str, Path]) -> None:
    rows = read_scene_table(
        [
            {
                "scene_id": "a",
                "radiance": str(files["rad1"]),
                "metadata": str(files["meta1"]),
                "altitude": "3000ft",
            },
            {
                "scene_id": "b",
                "radiance": str(files["rad2"]),
                "metadata": str(files["meta2"]),
                "altitude": "3000ft",
                "line": "0002",
            },
            {
                "scene_id": "c",
                "radiance": str(files["rad2"]),
                "metadata": str(files["meta2"]),
                "altitude": "6000ft",
                "line": "0003",
            },
        ],
        roles=ROLES,
    )
    ids = lambda selected: [row.scene_id for row in selected]  # noqa: E731
    assert ids(filter_rows(rows)) == ["a", "b", "c"]
    assert ids(filter_rows(rows, scene_id="b")) == ["b"]
    assert ids(filter_rows(rows, scene_id=["c", "a"])) == ["a", "c"]
    assert ids(filter_rows(rows, scene_id=("missing",))) == []
    assert ids(filter_rows(rows, altitude="3000ft")) == ["a", "b"]
    assert ids(filter_rows(rows, altitude="3000ft", line="0002")) == ["b"]
    assert ids(filter_rows(rows, line=3)) == []  # compared as strings: "3" != "0003"
    assert ids(filter_rows(rows, line="0003")) == ["c"]
    assert ids(filter_rows(rows, scene_id="a", altitude="6000ft")) == []
    with pytest.raises(ContractError, match="scene_id filter"):
        filter_rows(rows, scene_id=3)
