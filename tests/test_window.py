"""Scene window expansion and merging against the aviris4-ch4 golden fixture."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.scene import (
    expand_window,
    merge_scene_with_column,
    normalise_remapped_dimensions,
    remapped_window,
)
from spectra.types import SceneGeometry, Window

# The golden ``fake_remapped.nc`` records its scene window only in the legacy
# ``embedding_*`` attributes of the aviris4-ch4 CDO files. SPECTRA no longer
# reads those, so the tests build the window explicitly from them.
FAKE_WINDOW = Window(across=slice(5, 35), along=slice(10, 150))
NO_WINDOW = "must be given explicitly or carried as 'window_\\*' attributes"


def window_from_embedding_attrs(attrs: Mapping[str, Any]) -> Window:
    return Window(
        across=slice(
            int(attrs["embedding_across_start"]), int(attrs["embedding_across_stop"])
        ),
        along=slice(
            int(attrs["embedding_along_start"]), int(attrs["embedding_along_stop"])
        ),
    )


@pytest.fixture
def fake_remapped(golden_inputs: Path) -> xr.Dataset:
    with xr.open_dataset(golden_inputs / "fake_remapped.nc") as dataset:
        return dataset.load()


@pytest.fixture
def reference_merge(
    golden_radiance: xr.Dataset, fake_remapped: xr.Dataset
) -> xr.Dataset:
    return merge_scene_with_column(
        golden_radiance, fake_remapped, window=FAKE_WINDOW, suffix="_microhh"
    )


@pytest.mark.golden
def test_merge_matches_golden(
    golden_radiance: xr.Dataset, golden_inputs: Path, golden_expected: Path
) -> None:
    path = golden_inputs / "fake_remapped.nc"
    with xr.open_dataset(path) as dataset:
        window = window_from_embedding_attrs(dataset.attrs)
    assert window == FAKE_WINDOW
    merged = merge_scene_with_column(
        golden_radiance, path, window=window, suffix="_microhh"
    )
    merged = merged.drop_vars("radiance").load()
    with xr.open_dataset(golden_expected / "merged.nc") as dataset:
        expected = dataset.load()
    xr.testing.assert_identical(merged, expected)
    merged.close()


@pytest.mark.golden
def test_merge_reads_window_attributes(
    golden_radiance: xr.Dataset, fake_remapped: xr.Dataset, reference_merge: xr.Dataset
) -> None:
    # The ``window_*`` attributes written by SPECTRA regridders suffice.
    with_attrs = fake_remapped.copy()
    with_attrs.attrs = {**FAKE_WINDOW.attrs("window_"), "embedding_grid_method": "fake"}
    assert remapped_window(golden_radiance, with_attrs) == FAKE_WINDOW
    merged = merge_scene_with_column(golden_radiance, with_attrs, suffix="_microhh")
    xr.testing.assert_equal(merged["ch4_path"], reference_merge["ch4_path"])
    assert merged.attrs["window_across_start"] == 5
    assert merged.sizes == {"across": 30, "along": 140, "band": 29}

    # An explicit window takes precedence over (even unusable) attributes.
    stale = fake_remapped.copy()
    stale.attrs = {**FAKE_WINDOW.attrs("window_"), "window_along_stop": 151}
    assert remapped_window(golden_radiance, stale, FAKE_WINDOW) == FAKE_WINDOW
    merged = merge_scene_with_column(
        golden_radiance, stale, window=FAKE_WINDOW, suffix="_microhh"
    )
    xr.testing.assert_equal(merged["ch4_path"], reference_merge["ch4_path"])
    # The stale attribute is carried through as remapped provenance.
    assert merged.attrs["window_along_stop"] == 151


@pytest.mark.golden
def test_merge_requires_window(
    golden_radiance: xr.Dataset, fake_remapped: xr.Dataset
) -> None:
    # The legacy ``embedding_*`` attributes are provenance only.
    assert "embedding_across_start" in fake_remapped.attrs
    with pytest.raises(ContractError, match=NO_WINDOW):
        merge_scene_with_column(golden_radiance, fake_remapped)
    with pytest.raises(ContractError, match=NO_WINDOW):
        remapped_window(golden_radiance, fake_remapped)

    # Matching ``across``/``along`` coordinates do not identify a window.
    by_coordinates = fake_remapped.assign_coords(
        across=np.arange(5, 35), along=np.arange(10, 150)
    )
    by_coordinates.attrs = {}
    with pytest.raises(ContractError, match=NO_WINDOW):
        merge_scene_with_column(golden_radiance, by_coordinates)

    # Partial attribute sets name the missing attributes.
    partial = fake_remapped.copy()
    partial.attrs = FAKE_WINDOW.attrs("window_")
    partial.attrs.pop("window_along_stop")
    with pytest.raises(
        ContractError, match=r"missing attributes: \['window_along_stop'\]"
    ):
        merge_scene_with_column(golden_radiance, partial)


@pytest.mark.golden
def test_merge_rejects_mismatched_window(
    golden_radiance: xr.Dataset, fake_remapped: xr.Dataset
) -> None:
    too_wide = Window(across=slice(5, 36), along=slice(10, 150))
    with pytest.raises(ContractError, match="size 30, but the window \\[5, 36\\)"):
        merge_scene_with_column(golden_radiance, fake_remapped, window=too_wide)
    too_short = Window(across=slice(5, 35), along=slice(10, 149))
    with pytest.raises(ContractError, match="'along' has size 140"):
        remapped_window(golden_radiance, fake_remapped, too_short)
    beyond_scene = Window(across=slice(15, 45), along=slice(10, 150))
    with pytest.raises(ContractError, match="exceeds the scene dimension of size 40"):
        merge_scene_with_column(golden_radiance, fake_remapped, window=beyond_scene)

    # Window attributes are validated the same way.
    bad = fake_remapped.copy()
    bad.attrs = {**FAKE_WINDOW.attrs("window_"), "window_along_stop": 151}
    with pytest.raises(ContractError, match="size"):
        merge_scene_with_column(golden_radiance, bad)
    with pytest.raises(ContractError, match="missing dimensions"):
        remapped_window(golden_radiance, fake_remapped.isel(along=0), FAKE_WINDOW)


@pytest.mark.golden
def test_merge_suffixes_conflicts(
    golden_radiance: xr.Dataset, fake_remapped: xr.Dataset
) -> None:
    conflicting = fake_remapped.rename({"ch4_path": "radiance"})
    merged = merge_scene_with_column(golden_radiance, conflicting, window=FAKE_WINDOW)
    assert "radiance_plume" in merged
    assert merged["radiance"].dims == ("across", "along", "band")
    assert merged.sizes == {"across": 30, "along": 140, "band": 29}
    assert merged.attrs["embedding_grid_method"] == "fake"
    assert (
        merged.attrs["golden_across_start"]
        == golden_radiance.attrs["golden_across_start"]
    )
    blocked = conflicting.assign(radiance_plume=conflicting["radiance"])
    with pytest.raises(ContractError, match="already exist"):
        merge_scene_with_column(golden_radiance, blocked, window=FAKE_WINDOW)
    with pytest.raises(ContractError, match="missing dimensions"):
        merge_scene_with_column(
            golden_radiance.isel(across=0), fake_remapped, window=FAKE_WINDOW
        )


@pytest.mark.golden
def test_expand_window_round_trip(
    golden_column_on_grid: xr.DataArray, golden_metadata: xr.Dataset
) -> None:
    window = Window(across=slice(5, 35), along=slice(10, 150))
    cropped = golden_column_on_grid.isel(window.indexers).drop_vars(["across", "along"])
    cropped = cropped.to_dataset(name="column")
    cropped["lon"] = (("across", "along"), np.zeros(window.shape))
    expanded = expand_window(cropped, window, (40, 160))
    assert expanded.sizes == {"across": 40, "along": 160}
    assert "lon" not in expanded
    np.testing.assert_array_equal(expanded["across"].values, np.arange(40))
    inside = expanded["column"].isel(window.indexers)
    np.testing.assert_array_equal(
        inside.values, golden_column_on_grid.isel(window.indexers).values
    )
    outside = expanded["column"].values.copy()
    outside[5:35, 10:150] = np.nan
    assert np.nansum(np.abs(outside)) == 0.0
    assert np.sum(~np.isnan(outside)) == 40 * 160 - 30 * 140

    # Renamed horizontal dimensions are recovered from the ``lon`` variable.
    renamed = cropped.rename({"across": "y", "along": "x"})
    geometry = SceneGeometry.from_scene(golden_metadata)
    expanded_geometry = expand_window(renamed, window, geometry, fill_value=-1.0)
    xr.testing.assert_equal(expanded_geometry["column"].isel(window.indexers), inside)
    assert float(expanded_geometry["column"][0, 0]) == -1.0
    assert normalise_remapped_dimensions(renamed).sizes == {"across": 30, "along": 140}

    with pytest.raises(ContractError, match="match"):
        expand_window(
            cropped, Window(across=slice(5, 36), along=slice(10, 150)), (40, 160)
        )
    with pytest.raises(ContractError, match="exceeds"):
        expand_window(cropped, window, (30, 160))
    with pytest.raises(ContractError, match="two-dimensional"):
        expand_window(
            cropped.drop_vars("lon").rename({"across": "y", "along": "x"}),
            window,
            (40, 160),
        )
    with pytest.raises(ContractError, match="or neither"):
        expand_window(cropped.rename({"across": "y"}), window, (40, 160))
