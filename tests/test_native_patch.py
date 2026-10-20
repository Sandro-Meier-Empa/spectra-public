"""The storage assumptions behind copy-and-patch export, on synthetic native files.

Opening a byte copy of a native file with ``netCDF4`` in ``r+`` mode, writing a
slab of the radiance and adding dimensions, variables and root attributes must
leave every native dataset (dtype, layout, chunks, filters, fill value, storage
offset, attributes and bytes outside the slab) unchanged. The same check ran on
real AVIRIS-4 (contiguous and chunked) and EMIT (superblock v0 and v2) files
while the exporter was designed.
"""

from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from spectra.testing.native import (
    assert_native_preserved,
    native_snapshot,
    write_aviris4_files,
    write_emit_files,
)

pytest.importorskip("h5py")

AVIRIS4_DIMS = ("across", "along")
EMIT_DIMS = ("downtrack", "crosstrack")
CASES = {
    "aviris4-2025": (partial(write_aviris4_files, schema="2025"), AVIRIS4_DIMS),
    "aviris4-2026": (partial(write_aviris4_files, schema="2026"), AVIRIS4_DIMS),
    "emit-010635": (partial(write_emit_files, build="010635"), EMIT_DIMS),
    "emit-010617": (partial(write_emit_files, build="010617"), EMIT_DIMS),
}
NEW_OBJECTS = (
    "source",
    "source_strlen",
    "injected_ch4_mass",
    "source_plume_file",
    "source_flip",
)
NEW_ATTRS = ("spectra_version", "spectra_seed", "spectra_run_config")


@pytest.mark.parametrize("case", CASES)
def test_rplus_patch_keeps_native_storage(tmp_path: Path, case: str) -> None:
    writer, (y_dim, x_dim) = CASES[case]
    files = writer(tmp_path / "in")
    target = tmp_path / "out.nc"
    shutil.copyfile(files["radiance"], target)

    with netCDF4.Dataset(target) as ds:
        dims = ds["radiance"].dimensions
        shape = ds["radiance"].shape
    window = {y_dim: slice(3, 7), x_dim: slice(5, 9)}
    selection = tuple(window.get(dim, slice(10, 20)) for dim in dims)
    mask = np.zeros(shape, dtype=bool)
    mask[selection] = True
    before = native_snapshot(target, masks={"radiance": mask})

    with netCDF4.Dataset(target, "r+") as ds:
        radiance = ds["radiance"]
        radiance.set_auto_maskandscale(False)
        original = radiance[selection]
        radiance[selection] = (original * np.float32(0.9)).astype(radiance.dtype)
        ds.createDimension("source", 2)
        field = ds.createVariable(
            "injected_ch4_mass",
            "f4",
            ("source", y_dim, x_dim),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=(1, 8, 8),
        )
        field[:] = np.zeros(field.shape, dtype=np.float32)
        ds.createDimension("source_strlen", 13)
        files_variable = ds.createVariable(
            "source_plume_file", "S1", ("source", "source_strlen")
        )
        files_variable._Encoding = "utf-8"
        files_variable[:] = np.array(["/path/to/a.nc", "/path/to/b.nc"], dtype="U13")
        flips = ds.createVariable("source_flip", "i1", ("source",))
        flips[:] = [0, 1]
        ds.setncattr("spectra_version", "0.2.0")
        ds.setncattr("spectra_seed", np.int64(1))
        ds.setncattr_string("spectra_run_config", "x" * 70_000)

    after = native_snapshot(target, masks={"radiance": mask})
    assert_native_preserved(
        before, after, new_objects=NEW_OBJECTS, new_root_attrs=NEW_ATTRS
    )
    with netCDF4.Dataset(target) as ds:
        ds["radiance"].set_auto_maskandscale(False)
        patched = ds["radiance"][selection]
        assert np.array_equal(
            patched, (original * np.float32(0.9)).astype(patched.dtype), equal_nan=True
        )
        assert ds.getncattr("spectra_run_config") == "x" * 70_000
        assert list(ds["source_plume_file"][:]) == ["/path/to/a.nc", "/path/to/b.nc"]


def test_snapshot_detects_changes(tmp_path: Path) -> None:
    files = write_aviris4_files(tmp_path)
    before = native_snapshot(files["radiance"])
    with netCDF4.Dataset(files["radiance"], "r+") as ds:
        ds["radiance"][0, 1, 0] = 1.0
    changed = native_snapshot(files["radiance"])
    with pytest.raises(AssertionError, match=r"radiance\.sha256"):
        assert_native_preserved(before, changed)
    with netCDF4.Dataset(files["radiance"], "r+") as ds:
        ds["wavelength"].units = "um"
    with pytest.raises(AssertionError, match="units"):
        assert_native_preserved(changed, native_snapshot(files["radiance"]))
