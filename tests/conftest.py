"""Shared fixtures: golden reference files and environment-gated resources."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import xarray as xr
import yaml

GOLDEN = Path(__file__).parent / "golden"
INPUTS = GOLDEN / "inputs"
EXPECTED = GOLDEN / "expected"

# Real resources are supplied through environment variables so the suite runs
# without them; tests needing them are skipped. SPECTRA never searches
# directories for scene files: real scenes are listed file by file in a small
# YAML file ``{files: {role: /path/to/file}}`` named by SPECTRA_TEST_<SENSOR>_FILES.
RESOURCES_ENV = "SPECTRA_TEST_RESOURCES"  # libRadtran/ARTS/solar resource root
DATA_ENV = "SPECTRA_TEST_DATA"  # directory with real regridding comparison data
MICROHH_ENV = "SPECTRA_TEST_MICROHH_CASE"  # MicroHH case directory
OUTPUT_ENV = "SPECTRA_TEST_OUTPUT_DIR"  # where real-data outputs may be written


def require_golden() -> None:
    """Skip when the golden fixture has not been built."""
    if not (GOLDEN / "record.json").exists():
        pytest.skip("golden fixture missing; see tests/golden/README.md")


def env_path(name: str) -> Path:
    """Return the path in an environment variable or skip the test."""
    value = os.environ.get(name)
    if not value or not Path(value).exists():
        pytest.skip(f"{name} is not set or does not exist")
    return Path(value)


@pytest.fixture(scope="session")
def golden_record() -> dict:
    require_golden()
    return json.loads((GOLDEN / "record.json").read_text())


@pytest.fixture(scope="session")
def golden_inputs() -> Path:
    require_golden()
    return INPUTS


@pytest.fixture(scope="session")
def golden_expected() -> Path:
    require_golden()
    return EXPECTED


@pytest.fixture(scope="session")
def golden_radiance(golden_inputs: Path) -> xr.Dataset:
    return xr.open_dataset(golden_inputs / "radiance_window.nc").load()


@pytest.fixture(scope="session")
def golden_metadata(golden_inputs: Path) -> xr.Dataset:
    return xr.open_dataset(golden_inputs / "metadata_window.nc").load()


@pytest.fixture(scope="session")
def golden_microhh_raw(golden_inputs: Path) -> xr.Dataset:
    return xr.open_dataset(golden_inputs / "microhh_raw.nc").load()


@pytest.fixture(scope="session")
def golden_lut(golden_inputs: Path) -> xr.Dataset:
    return xr.open_dataset(golden_inputs / "lut.nc").load()


@pytest.fixture(scope="session")
def golden_column_on_grid(golden_inputs: Path) -> xr.DataArray:
    return xr.open_dataset(golden_inputs / "column_on_grid.nc").load()["column"]


@pytest.fixture(scope="session")
def resources_root() -> Path:
    return env_path(RESOURCES_ENV)


@pytest.fixture(scope="session")
def data_root() -> Path:
    return env_path(DATA_ENV)


@pytest.fixture(scope="session")
def microhh_case_dir() -> Path:
    return env_path(MICROHH_ENV)


def native_files(sensor: str) -> dict[str, Path]:
    """Return the files of one real scene listed in SPECTRA_TEST_<SENSOR>_FILES."""
    listing = env_path(f"SPECTRA_TEST_{sensor.upper()}_FILES")
    files = yaml.safe_load(listing.read_text())["files"]
    return {role: Path(path) for role, path in files.items()}
