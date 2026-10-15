"""Plume placement in a sensor swath against the aviris4-ch4 golden fixture."""

from __future__ import annotations

import numpy as np
import pyproj
import pytest
import xarray as xr
from shapely.geometry import Polygon

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import outline_polygon
from spectra.geometry.swath import (
    MassPoints,
    ScenePlacement,
    anchor_pixel_spacing,
    outside_fraction_bounds,
    outside_mass_fraction,
    pixel_centre,
    place_in_scene,
    random_anchor,
    swath_polygon,
    window_covering_support,
    window_nodes,
    window_polygon,
)
from spectra.types import Placement, SceneGeometry, Window

PLACEMENTS = ("rot", "flip", "partial")


def placement_from_record(record: dict) -> Placement:
    return Placement(
        anchor_along=record["along_index"],
        anchor_across=record["across_index"],
        anchor_corner=record["anchor"],
        rotation_deg=record["rotation"],
        flip=record["flip_axis"] or "none",
    )


def record_window(record: dict) -> Window:
    return Window(
        across=slice(record["across_slice"]["start"], record["across_slice"]["stop"]),
        along=slice(record["along_slice"]["start"], record["along_slice"]["stop"]),
    )


def assert_window_captures_support(
    geometry: SceneGeometry, scene_placement: ScenePlacement
) -> None:
    """The window captures the whole in-swath support area."""
    support = scene_placement.placed.support_polygon
    outline = window_polygon(geometry, scene_placement.window)
    inside = scene_placement.swath.intersection(support)
    assert inside.area > 0
    assert outline.intersection(support).area >= inside.area * (1 - 1e-9)
    if scene_placement.swath.covers(support):
        assert outline.covers(support)


@pytest.fixture(scope="module")
def geometry(golden_metadata: xr.Dataset) -> SceneGeometry:
    return SceneGeometry.from_scene(golden_metadata)


@pytest.fixture(scope="module")
def column(golden_microhh_raw: xr.Dataset) -> xr.DataArray:
    column = golden_microhh_raw["ch4_path"].copy()
    column.attrs["units"] = "kg m-2"
    return column


@pytest.mark.golden
def test_swath_polygon_matches_full_node_outline(geometry: SceneGeometry) -> None:
    n_across, n_along = geometry.shape
    full = Window(across=slice(0, n_across), along=slice(0, n_along))
    expected = outline_polygon(*window_nodes(geometry, full))
    swath = swath_polygon(geometry)
    assert np.array_equal(
        np.asarray(swath.exterior.coords), np.asarray(expected.exterior.coords)
    )
    assert swath.is_valid and swath.area > 0
    assert np.array_equal(
        np.asarray(window_polygon(geometry, full).exterior.coords),
        np.asarray(expected.exterior.coords),
    )


@pytest.mark.golden
def test_pixel_centre_and_spacing(geometry: SceneGeometry, golden_metadata) -> None:
    x, y = pixel_centre(geometry, along=15, across=2)
    assert x == float(golden_metadata["x"].values[2, 15])
    assert y == float(golden_metadata["y"].values[2, 15])
    spacing = anchor_pixel_spacing(geometry, along=15, across=2)
    assert 0 < spacing < 5
    # The last pixel uses the neighbourhood below/left of it.
    assert anchor_pixel_spacing(geometry, along=159, across=39) > 0


@pytest.mark.golden
@pytest.mark.parametrize("name", PLACEMENTS)
def test_explicit_placement_matches_record(
    golden_record: dict, geometry: SceneGeometry, column: xr.DataArray, name: str
) -> None:
    record = golden_record["placements"][name]
    placement = placement_from_record(record)
    scene_placement = place_in_scene(column, geometry, placement=placement)
    draw = scene_placement.draw
    assert scene_placement.mode == "explicit"
    assert draw.attempts is None
    assert (draw.along, draw.across) == (record["along_index"], record["across_index"])
    assert scene_placement.window == record_window(record)
    assert draw.outside_fraction == pytest.approx(record["outside_fraction"], abs=1e-12)
    assert scene_placement.placed.offset_xy == pytest.approx(
        (record["x_offset"], record["y_offset"]), abs=1e-12
    )
    assert scene_placement.placement == Placement(
        anchor_along=record["along_index"],
        anchor_across=record["across_index"],
        anchor_corner=record["anchor"],
        rotation_deg=record["rotation"],
        flip=record["flip_axis"] or "none",
    )
    attrs = scene_placement.attrs()
    assert attrs["embedding_placement"] == "explicit"
    assert attrs["embedding_outside_mass_fraction"] == draw.outside_fraction
    assert "embedding_attempts" not in attrs
    assert attrs["embedding_across_start"] == record["across_slice"]["start"]
    assert attrs["embedding_along_stop"] == record["along_slice"]["stop"]
    assert_window_captures_support(geometry, scene_placement)


@pytest.mark.golden
def test_random_placement_is_reproducible(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    kwargs = {"rotation_deg": -20, "anchor_corner": "upper left"}
    first = place_in_scene(column, geometry, rng=np.random.default_rng(1), **kwargs)
    second = place_in_scene(column, geometry, rng=np.random.default_rng(1), **kwargs)
    assert first.mode == "random"
    assert first.draw.outside_fraction == 0.0
    assert first.draw.attempts is not None and first.draw.attempts >= 1
    assert (first.draw.along, first.draw.across) == (
        second.draw.along,
        second.draw.across,
    )
    assert first.window == second.window
    assert_window_captures_support(geometry, first)
    assert first.placement.rotation_deg == -20
    assert first.placement.anchor_corner == "upper left"
    attrs = first.attrs()
    assert attrs["embedding_placement"] == "random"
    assert attrs["embedding_attempts"] == first.draw.attempts
    assert attrs["embedding_min_outside_fraction"] == 0.0
    assert attrs["embedding_max_outside_fraction"] == 0.0


@pytest.mark.golden
def test_random_placement_within_outside_bounds(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    # The golden plume holds 99.5 % of its mass in the source column, so the
    # attainable outside fractions cluster near 0 and above 0.7.
    scene_placement = place_in_scene(
        column,
        geometry,
        min_outside_fraction=0.5,
        max_outside_fraction=0.95,
        rng=np.random.default_rng(3),
    )
    fraction = scene_placement.draw.outside_fraction
    assert 0.5 <= fraction <= 0.95
    assert scene_placement.bounds == (0.5, 0.95)
    assert not scene_placement.swath.covers(scene_placement.placed.support_polygon)
    assert_window_captures_support(geometry, scene_placement)
    # The window never extends past the swath.
    n_across, n_along = geometry.shape
    assert scene_placement.window.across.start >= 0
    assert scene_placement.window.across.stop <= n_across
    assert scene_placement.window.along.stop <= n_along


@pytest.mark.golden
def test_random_placement_exhausts_attempts(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    with pytest.raises(ContractError, match="draws"):
        place_in_scene(
            column,
            geometry,
            min_outside_fraction=0.123456,
            max_outside_fraction=0.123456,
            rng=np.random.default_rng(0),
            max_attempts=1,
        )


@pytest.mark.golden
def test_place_in_scene_validation(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    placement = Placement(anchor_along=15, anchor_across=2)
    with pytest.raises(ContractError, match="random placement"):
        place_in_scene(column, geometry, placement=placement, max_outside_fraction=0.5)
    with pytest.raises(ContractError, match="along"):
        place_in_scene(column, geometry, placement=Placement(anchor_along=500))
    with pytest.raises(ContractError, match="across"):
        place_in_scene(column, geometry, placement=Placement(anchor_across=500))
    with pytest.raises(ContractError):
        place_in_scene(
            column, geometry, min_outside_fraction=0.5, max_outside_fraction=0.2
        )
    with pytest.raises(ContractError):
        place_in_scene(column, geometry, max_outside_fraction=1.0)
    with pytest.raises(ContractError):
        place_in_scene(column, geometry, max_attempts=0)
    with pytest.raises(TypeError):
        place_in_scene(column, geometry, rotation_deg=1.5)  # type: ignore[arg-type]
    unitless = column.copy()
    unitless.attrs.pop("units")
    with pytest.raises(ContractError, match="dimensions"):
        place_in_scene(column.rename({"x": "u"}), geometry, placement=placement)
    with pytest.raises(ContractError, match="units"):
        place_in_scene(unitless, geometry, placement=placement)
    assert outside_fraction_bounds(0, 0.5) == (0.0, 0.5)


def test_outside_mass_fraction_synthetic() -> None:
    values = np.zeros((3, 4))
    values[1, 1] = 1.0
    values[1, 2] = 3.0
    x = np.arange(4, dtype=float)
    y = np.arange(3, dtype=float)
    points = MassPoints.from_values(
        values, x, y, anchor_source=(0.0, 0.0), rotation_deg=0, flip="none"
    )
    assert points.total == 4.0
    support = np.asarray([(0.5, 0.5), (2.5, 0.5), (2.5, 1.5), (0.5, 1.5)])
    everything = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    assert outside_mass_fraction(everything, support, (0.0, 0.0), points) == 0.0
    nothing = Polygon([(50, 50), (60, 50), (60, 60), (50, 60)])
    assert outside_mass_fraction(nothing, support, (0.0, 0.0), points) == 1.0
    # Region containing only the centre of the first cell: 3/4 of the mass outside.
    half = Polygon([(0.75, 0.5), (1.5, 0.5), (1.5, 1.5), (0.75, 1.5)])
    assert outside_mass_fraction(half, support, (0.0, 0.0), points) == pytest.approx(
        0.75
    )
    # Translating the centre moves the plume with it.
    assert outside_mass_fraction(half, support, (1.0, 0.0), points) == pytest.approx(
        1.0
    )
    with pytest.raises(ContractError, match="support"):
        MassPoints.from_values(
            np.zeros((2, 2)),
            x[:2],
            y[:2],
            anchor_source=(0.0, 0.0),
            rotation_deg=0,
            flip="none",
        )


def test_random_anchor_on_synthetic_geometry() -> None:
    # Regular 1 m grid, 10 across x 20 along, swath in metres.
    along_x, across_y = np.meshgrid(np.arange(20.0), np.arange(10.0), indexing="xy")
    geometry = SceneGeometry(
        x=xr.DataArray(along_x, dims=("across", "along")),
        y=xr.DataArray(across_y, dims=("across", "along")),
        crs=pyproj.CRS.from_epsg(32632),
    )
    swath = swath_polygon(geometry)
    assert swath.bounds == pytest.approx((-0.5, -0.5, 19.5, 9.5))
    values = np.ones((2, 3))
    points = MassPoints.from_values(
        values,
        np.arange(3.0),
        np.arange(2.0),
        anchor_source=(0.0, 0.0),
        rotation_deg=0,
        flip="none",
    )
    support = np.asarray([(-0.5, -0.5), (2.5, -0.5), (2.5, 1.5), (-0.5, 1.5)])
    draw = random_anchor(
        geometry,
        swath,
        points,
        support,
        bounds=(0.0, 0.0),
        rng=np.random.default_rng(7),
        max_attempts=1000,
    )
    assert draw.outside_fraction == 0.0
    assert draw.along <= 17 and draw.across <= 8
    window = window_covering_support(
        geometry,
        support + np.asarray((draw.anchor_x, draw.anchor_y)),
        swath=swath,
        covered=True,
        anchor_index=(draw.along, draw.across),
    )
    assert window.along.start <= draw.along
    assert window.along.stop >= draw.along + 3
    assert window.across.stop >= draw.across + 2


def test_scene_placement_attrs_keys(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    scene_placement = place_in_scene(
        column, geometry, placement=Placement(anchor_along=15, anchor_across=2)
    )
    assert isinstance(scene_placement, ScenePlacement)
    assert set(scene_placement.attrs()) == {
        "embedding_placement",
        "embedding_outside_mass_fraction",
        "embedding_across_start",
        "embedding_across_stop",
        "embedding_along_start",
        "embedding_along_stop",
    }
