"""The multi-plume pipeline reproduces the single-plume pipeline it replaced.

``expected/embedded_rot_window.nc`` holds the output of the v0.1.0 pipeline
(``place_and_regrid`` + ``embed_radiance_sample``) on the golden AVIRIS-4
crop with the golden MicroHH snapshot, the golden "rot" placement and the
golden absorption table. The current pipeline, run through the real AVIRIS-4
adapter, must reproduce it bit for bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from spectra.contracts.plume_column import plume_column_attrs
from spectra.pipeline import embed_scene
from spectra.plume.io import write_plume_column
from spectra.run import AnchorSpec, PlumeSpec
from spectra.sensors import Aviris4Adapter
from spectra.types import SpectralResponse, ViewingGeometry, Window

pytestmark = pytest.mark.golden


class FixedTable:
    """Forward model returning one precomputed absorption table."""

    name = "fixed"

    def __init__(self, lut: xr.Dataset) -> None:
        self.table = lut

    @property
    def spectral_range_nm(self) -> tuple[float, float]:
        wavelength = self.table["wavelength"].values
        return float(wavelength.min()), float(wavelength.max())

    @property
    def enhancements_ppm(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.table["enhancement"].values)

    @property
    def layer_height_m(self) -> float:
        return float(self.table.attrs["layer_height"])

    def check(self, gas: str) -> None:
        del gas

    def lut(
        self, response: SpectralResponse, geometry: ViewingGeometry, *, gas: str
    ) -> xr.Dataset:
        del geometry, gas
        assert np.array_equal(response.wavelength, self.table["wavelength"].values)
        return self.table

    def provenance(self) -> dict[str, Any]:
        return {"name": self.name}


@pytest.fixture(scope="module")
def plume_file(golden_microhh_raw: xr.Dataset, tmp_path_factory) -> Path:
    raw = golden_microhh_raw["ch4_path"]
    column = raw.astype(np.float64).expand_dims(
        time=[np.datetime64("2024-09-18T13:30")]
    )
    dataset = xr.Dataset(
        {
            "column": column.assign_attrs(units="kg m-2"),
            "emission_rate": ("time", [0.08], {"units": "kg s-1"}),
        },
        coords={
            "x": raw["x"].astype(np.float64),
            "y": raw["y"].astype(np.float64),
            "simulation_time": ("time", [6000.0], {"units": "s"}),
        },
        attrs=plume_column_attrs(
            gas="CH4",
            les_model="MicroHH",
            case_id="golden",
            tracer="ch4",
            source_id="golden",
            source_x=5.0,
            source_y=6.0,
            source_z=1.0,
            dx=1.0,
            dy=1.0,
            reference_time="2024-09-18T13:30:00",
        ),
    )
    path = tmp_path_factory.mktemp("plume") / "golden.plumecolumn.nc"
    write_plume_column(dataset, path)
    return path


def test_pipeline_reproduces_single_plume_reference(
    golden_inputs: Path,
    golden_expected: Path,
    golden_record: dict,
    golden_lut: xr.Dataset,
    plume_file: Path,
) -> None:
    files = {
        "radiance": golden_inputs / "radiance_window.nc",
        "metadata": golden_inputs / "metadata_window.nc",
    }
    scene = Aviris4Adapter().open_scene(files)
    record = golden_record["placements"]["rot"]
    expected = xr.open_dataset(golden_expected / "embedded_rot_window.nc").load()
    plume = PlumeSpec(
        file=plume_file,
        time_index=0,
        anchor=AnchorSpec(
            along=record["along_index"],
            across=record["across_index"],
            corner=record["anchor"],
        ),
        rotation=record["rotation"],
    )

    result, resolved = embed_scene(scene, [plume], FixedTable(golden_lut), seed=0)

    assert resolved == (plume,)
    (cluster,) = result.clusters
    (source,) = result.sources
    assert cluster.window == source.window == Window.from_attrs(expected.attrs)
    np.testing.assert_array_equal(
        source.injected, expected["injected_ch4_mass"].values.astype(np.float32)
    )
    assert source.injected_mass_kg == pytest.approx(
        expected.attrs["regrid_target_mass"], rel=1e-12
    )
    embedded = result.to_dataset(scene)["radiance"].isel(cluster.window.indexers)
    np.testing.assert_array_equal(
        embedded.transpose("across", "along", "band").values,
        expected["radiance"].values,
    )
