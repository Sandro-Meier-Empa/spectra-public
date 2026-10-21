"""Plume placement in a sensor swath, partly against the aviris4-ch4 golden fixture."""

from __future__ import annotations

import functools

import numpy as np
import pyproj
import pytest
import shapely
import xarray as xr
from shapely.geometry import Polygon

import spectra.geometry.swath as swath_module
from spectra.contracts.errors import ContractError
from spectra.geometry.corners import outline_polygon
from spectra.geometry.crs import to_lonlat
from spectra.geometry.feasibility import AnchorTiles, FeasibilityMap
from spectra.geometry.placement import PlacedColumn, place_column
from spectra.geometry.swath import (
    MassPoints,
    ResolvedAnchor,
    anchor_pixel_spacing,
    locate_pixel,
    outside_fraction_bounds,
    outside_mass_fraction,
    pixel_centre,
    random_anchor,
    resolve_anchor,
    swath_polygon,
    window_nodes,
    window_polygon,
)
from spectra.geometry.transform import support_vertices
from spectra.regrid.windows import anchor_pixel_centre, support_window
from spectra.types import AnchorPoint, Flip, Placement, SceneGeometry, Window

PLACEMENTS = ("rot", "flip", "partial")


def placement_from_record(record: dict) -> Placement:
    return Placement(
        anchor_along=record["along_index"],
        anchor_across=record["across_index"],
        anchor_corner=record["anchor"],
        rotation_deg=record["rotation"],
        flip=record["flip_axis"] or "none",
    )


def placed_on(
    column: xr.DataArray, geometry: SceneGeometry, anchor: ResolvedAnchor
) -> PlacedColumn:
    return place_column(
        column,
        crs=geometry.crs,
        anchor_target_xy=anchor_pixel_centre(geometry, anchor.placement),
        placement=anchor.placement,
    )


def assert_window_captures_support(
    geometry: SceneGeometry, placed: PlacedColumn
) -> None:
    """The regridding window captures the whole in-swath support area."""
    window = support_window(placed, geometry)
    swath = swath_polygon(geometry)
    support = placed.support_polygon
    inside = swath.intersection(support)
    assert inside.area > 0
    outline = window_polygon(geometry, window)
    assert outline.intersection(support).area >= inside.area * (1 - 1e-9)


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
    window = window_polygon(geometry, full)
    # The golden swath does not fold, so all outlines are single polygons.
    assert isinstance(expected, Polygon) and isinstance(swath, Polygon)
    assert isinstance(window, Polygon)
    assert np.array_equal(
        np.asarray(swath.exterior.coords), np.asarray(expected.exterior.coords)
    )
    assert swath.is_valid and swath.area > 0
    assert np.array_equal(
        np.asarray(window.exterior.coords), np.asarray(expected.exterior.coords)
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
def test_pixel_anchor_matches_record(
    golden_record: dict, geometry: SceneGeometry, column: xr.DataArray, name: str
) -> None:
    record = golden_record["placements"][name]
    expected = placement_from_record(record)
    anchor = resolve_anchor(
        column,
        geometry,
        mode="pixel",
        corner=expected.anchor_corner,
        rotation_deg=expected.rotation_deg,
        flip=expected.flip,
        along=record["along_index"],
        across=record["across_index"],
    )
    assert anchor.mode == "pixel" and anchor.attempts is None
    assert anchor.placement == expected
    assert anchor.outside_fraction == pytest.approx(
        record["outside_fraction"], abs=1e-12
    )
    placed = placed_on(column, geometry, anchor)
    assert placed.offset_xy == pytest.approx(
        (record["x_offset"], record["y_offset"]), abs=1e-12
    )
    assert anchor.anchor_xy == pixel_centre(
        geometry, record["along_index"], record["across_index"]
    )
    assert_window_captures_support(geometry, placed)


@pytest.mark.golden
def test_random_anchor_is_reproducible(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    resolve = functools.partial(
        resolve_anchor, column, geometry, rotation_deg=-20, corner="upper left"
    )
    first = resolve(mode="random", rng=np.random.default_rng(1))
    second = resolve(mode="random", rng=np.random.default_rng(1))
    assert first.mode == "random"
    assert first.outside_fraction == 0.0
    assert first.attempts is not None and first.attempts >= 1
    assert first.placement == second.placement
    assert first.placement.rotation_deg == -20
    assert first.placement.anchor_corner == "upper left"
    assert first.placement.anchor_xy is None
    assert_window_captures_support(geometry, placed_on(column, geometry, first))


@pytest.mark.golden
def test_random_anchor_within_outside_bounds(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    # The golden plume holds 99.5 % of its mass in the source column, so the
    # attainable outside fractions cluster near 0 and above 0.7.
    anchor = resolve_anchor(
        column,
        geometry,
        mode="random",
        corner="lower left",
        bounds=(0.5, 0.95),
        rng=np.random.default_rng(3),
    )
    assert 0.5 <= anchor.outside_fraction <= 0.95
    placed = placed_on(column, geometry, anchor)
    assert not swath_polygon(geometry).covers(placed.support_polygon)
    assert_window_captures_support(geometry, placed)


@pytest.mark.golden
def test_random_anchor_exhausts_attempts(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    with pytest.raises(ContractError, match="draws"):
        resolve_anchor(
            column,
            geometry,
            mode="random",
            corner="lower left",
            bounds=(0.123456, 0.123456),
            rng=np.random.default_rng(0),
            max_attempts=1,
        )


@pytest.mark.golden
def test_resolve_anchor_validation(
    geometry: SceneGeometry, column: xr.DataArray
) -> None:
    resolve = functools.partial(resolve_anchor, column, geometry, corner="lower left")
    with pytest.raises(ContractError, match="along and across"):
        resolve(mode="pixel", along=15)
    with pytest.raises(ContractError, match="along"):
        resolve(mode="pixel", along=500, across=2)
    with pytest.raises(ContractError, match="across"):
        resolve(mode="pixel", along=2, across=500)
    with pytest.raises(ContractError, match="lon and lat"):
        resolve(mode="lonlat")
    with pytest.raises(ContractError, match="Unknown anchor mode"):
        resolve(mode="nowhere")  # type: ignore[arg-type]
    with pytest.raises(ContractError):
        resolve(mode="random", bounds=(0.5, 0.2))
    with pytest.raises(ContractError):
        resolve(mode="random", bounds=(0.0, 1.0))
    with pytest.raises(ContractError):
        resolve(mode="random", max_attempts=0)
    with pytest.raises(TypeError):
        resolve(mode="random", rotation_deg=1.5)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="source_x"):
        resolve_anchor(column, geometry, mode="pixel", along=15, across=2)
    unitless = column.copy()
    unitless.attrs.pop("units")
    with pytest.raises(ContractError, match="dimensions"):
        resolve_anchor(
            column.rename({"x": "u"}), geometry, mode="pixel", along=15, across=2
        )
    with pytest.raises(ContractError, match="units"):
        resolve_anchor(
            unitless, geometry, mode="pixel", along=15, across=2, corner="lower left"
        )
    assert outside_fraction_bounds(0, 0.5) == (0.0, 0.5)


@pytest.mark.golden
def test_lonlat_anchor_on_golden_swath(
    geometry: SceneGeometry, column: xr.DataArray, golden_metadata: xr.Dataset
) -> None:
    sourced = column.assign_attrs(
        source_x=float(column["x"][5]), source_y=float(column["y"][6])
    )
    # A point between pixel centres, in the interior of the swath.
    x = 0.6 * float(golden_metadata["x"][20, 80]) + 0.4 * float(
        golden_metadata["x"][21, 81]
    )
    y = 0.6 * float(golden_metadata["y"][20, 80]) + 0.4 * float(
        golden_metadata["y"][21, 81]
    )
    lon, lat = (float(value) for value in to_lonlat(geometry.crs, x, y))
    anchor = resolve_anchor(sourced, geometry, mode="lonlat", lonlat=(lon, lat))
    assert anchor.mode == "lonlat" and anchor.attempts is None
    assert anchor.placement.anchor_xy == pytest.approx((x, y), abs=1e-6)
    assert anchor.anchor_xy == anchor.placement.anchor_xy
    assert (anchor.placement.anchor_along, anchor.placement.anchor_across) == (80, 20)
    placed = placed_on(sourced, geometry, anchor)
    # The emission source cell centre lands exactly on the geographic anchor.
    assert placed.anchor_source_xy == (
        sourced.attrs["source_x"],
        sourced.attrs["source_y"],
    )
    assert (placed.centre_x[6, 5], placed.centre_y[6, 5]) == pytest.approx(
        (x, y), abs=1e-6
    )
    assert_window_captures_support(geometry, placed)
    far_lon, far_lat = (float(v) for v in to_lonlat(geometry.crs, x + 5000.0, y))
    with pytest.raises(ContractError, match="outside the swath"):
        resolve_anchor(sourced, geometry, mode="lonlat", lonlat=(far_lon, far_lat))


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
    placement = Placement(anchor_along=draw.along, anchor_across=draw.across)
    column = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"x": np.arange(3.0), "y": np.arange(2.0)},
        attrs={"units": "kg m-2"},
    )
    placed = place_column(
        column,
        crs=geometry.crs,
        anchor_target_xy=(draw.anchor_x, draw.anchor_y),
        placement=placement,
    )
    window = support_window(placed, geometry)
    assert window.along.start <= draw.along
    assert window.along.stop >= draw.along + 3
    assert window.across.stop >= draw.across + 2


def anchor_inputs(
    column: xr.DataArray,
    geometry: SceneGeometry,
    *,
    corner: AnchorPoint,
    rotation_deg: int,
    flip: Flip,
) -> tuple[np.ndarray, MassPoints]:
    """Return the support rectangle and mass points as resolve_anchor builds them."""
    placement = Placement(anchor_corner=corner, rotation_deg=rotation_deg, flip=flip)
    probe = place_column(
        column, crs=geometry.crs, anchor_target_xy=(0.0, 0.0), placement=placement
    )
    support = support_vertices(
        probe.support_bounds, probe.anchor_source_xy, rotation_deg, flip
    )
    points = MassPoints.from_values(
        probe.values,
        column["x"].values,
        column["y"].values,
        anchor_source=probe.anchor_source_xy,
        rotation_deg=rotation_deg,
        flip=flip,
    )
    return support, points


def synthetic_grid(n_across: int = 10, n_along: int = 20) -> SceneGeometry:
    """Return a regular 1 m grid with along-track x and across-track y."""
    along_x, across_y = np.meshgrid(
        np.arange(float(n_along)), np.arange(float(n_across)), indexing="xy"
    )
    return SceneGeometry(
        x=xr.DataArray(along_x, dims=("across", "along")),
        y=xr.DataArray(across_y, dims=("across", "along")),
        crs=pyproj.CRS.from_epsg(32632),
    )


@pytest.mark.golden
@pytest.mark.parametrize(
    ("corner", "rotation_deg", "flip"),
    [("upper left", 0, "none"), ("lower left", -20, "x"), ("lower right", 135, "y")],
)
def test_feasibility_bounds_hold_for_every_pixel(
    geometry: SceneGeometry,
    column: xr.DataArray,
    corner: AnchorPoint,
    rotation_deg: int,
    flip: Flip,
) -> None:
    support, points = anchor_inputs(
        column, geometry, corner=corner, rotation_deg=rotation_deg, flip=flip
    )
    swath = swath_polygon(geometry)
    shapely.prepare(swath)
    tiles = AnchorTiles.from_geometry(geometry)
    feasibility = FeasibilityMap.build(tiles, swath, points)
    x = np.asarray(geometry.x.values)
    y = np.asarray(geometry.y.values)
    fractions = np.array(
        [
            [
                outside_mass_fraction(swath, support, (x[a, b], y[a, b]), points)
                for b in range(x.shape[1])
            ]
            for a in range(x.shape[0])
        ]
    )
    size = tiles.size
    lower = feasibility.lower.repeat(size, 0).repeat(size, 1)[
        : x.shape[0], : x.shape[1]
    ]
    upper = feasibility.upper.repeat(size, 0).repeat(size, 1)[
        : x.shape[0], : x.shape[1]
    ]
    # The bounds hold for every pixel, so the candidates hold every eligible one.
    assert np.all(fractions >= lower - 1e-9)
    assert np.all(fractions <= upper + 1e-9)
    # They are informative: most pixels are pinned down closely.
    assert np.mean(upper - lower < 0.1) > 0.5


def test_random_anchor_fails_fast_when_no_pixel_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = synthetic_grid()
    swath = swath_polygon(geometry)
    # The plume spans 12 m across track, the swath 10 m.
    points = MassPoints.from_values(
        np.ones((12, 2)),
        np.arange(2.0),
        np.arange(12.0),
        anchor_source=(0.0, 0.0),
        rotation_deg=0,
        flip="none",
    )
    support = np.asarray([(-0.5, -0.5), (1.5, -0.5), (1.5, 11.5), (-0.5, 11.5)])
    tested: list[object] = []

    def counting(*args: object) -> float:
        tested.append(args)
        return 0.0

    monkeypatch.setattr(swath_module, "outside_mass_fraction", counting)
    with pytest.raises(ContractError, match="reaches fractions between"):
        random_anchor(
            geometry,
            swath,
            points,
            support,
            bounds=(0.0, 0.0),
            rng=np.random.default_rng(0),
        )
    assert not tested


def test_random_anchor_skips_pixels_without_geolocation() -> None:
    geometry = synthetic_grid()
    x = np.asarray(geometry.x.values)
    y = np.asarray(geometry.y.values)
    x[3:7, 5:15] = np.nan
    y[3:7, 5:15] = np.nan
    swath = swath_polygon(geometry)
    points = MassPoints.from_values(
        np.ones((1, 1)),
        np.zeros(1),
        np.zeros(1),
        anchor_source=(0.0, 0.0),
        rotation_deg=0,
        flip="none",
    )
    support = np.asarray([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    tiles = AnchorTiles.from_geometry(geometry)
    for seed in range(40):
        draw = random_anchor(
            geometry,
            swath,
            points,
            support,
            bounds=(0.0, 0.0),
            rng=np.random.default_rng(seed),
            tiles=tiles,
        )
        assert np.isfinite(x[draw.across, draw.along])
        assert draw.attempts == 1


def test_locate_pixel_matches_brute_force() -> None:
    # Rotated, slightly sheared 7 m grid, 60 across x 400 along.
    along, across = np.meshgrid(np.arange(400.0), np.arange(60.0), indexing="xy")
    angle = np.deg2rad(23.0)
    x = 500000.0 + 7.0 * (np.cos(angle) * along - np.sin(angle) * across) + 0.3 * across
    y = 5200000.0 + 7.0 * (np.sin(angle) * along + np.cos(angle) * across)
    geometry = SceneGeometry(
        x=xr.DataArray(x, dims=("across", "along")).chunk({"along": 50}),
        y=xr.DataArray(y, dims=("across", "along")).chunk({"along": 50}),
        crs=pyproj.CRS.from_epsg(32632),
    )
    rng = np.random.default_rng(5)
    for _ in range(25):
        i, j = int(rng.integers(0, 60)), int(rng.integers(0, 400))
        point = (x[i, j] + rng.uniform(-3, 3), y[i, j] + rng.uniform(-3, 3))
        distance = np.hypot(x - point[0], y - point[1])
        row, column = np.unravel_index(np.argmin(distance), distance.shape)
        assert locate_pixel(geometry, point) == (int(column), int(row))


def test_source_anchor_pins_emission_point() -> None:
    values = np.zeros((4, 6))
    values[1:3, 2:5] = 1.0
    column = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"x": 10.0 + 2.0 * np.arange(6), "y": 20.0 + 2.0 * np.arange(4)},
        attrs={"units": "kg m-2", "source_x": 15.0, "source_y": 23.0},
    )
    placement = Placement(anchor_corner="source", rotation_deg=90, flip="x")
    placed = place_column(
        column,
        crs=pyproj.CRS.from_epsg(32632),
        anchor_target_xy=(1000.0, 2000.0),
        placement=placement,
    )
    assert placed.anchor_source_xy == (15.0, 23.0)
    # The source point is the pivot: it maps onto the target, the rest rotates.
    assert placed.offset_xy == (985.0, 1977.0)
    ny, nx = values.shape
    # Cell (row 1, column 2) sits at (14, 22), i.e. (-1, -1) from the source;
    # mirroring x gives (1, -1), rotating 90 deg counterclockwise (1, 1).
    centre = placed.centre_x[1, 2], placed.centre_y[1, 2]
    assert centre == pytest.approx((1001.0, 2001.0))
    assert (ny, nx) == placed.shape
    missing = column.copy()
    missing.attrs = {"units": "kg m-2"}
    with pytest.raises(ContractError, match="source_x"):
        place_column(
            missing,
            crs=pyproj.CRS.from_epsg(32632),
            anchor_target_xy=(0.0, 0.0),
            placement=placement,
        )


def test_folded_swath_edge_is_repaired() -> None:
    # Edge pixels displaced backwards along track (aircraft roll) make the
    # boundary ring cross itself; the swath is the repaired enclosed region.
    along_x, across_y = np.meshgrid(np.arange(30.0), np.arange(10.0), indexing="xy")
    along_x[0, 12:15] -= 4.5
    geometry = SceneGeometry(
        x=xr.DataArray(along_x, dims=("across", "along")),
        y=xr.DataArray(across_y, dims=("across", "along")),
        crs=pyproj.CRS.from_epsg(32632),
    )
    swath = swath_polygon(geometry)
    assert swath.is_valid and swath.area > 0
    inner = shapely.points(along_x[1:].ravel(), across_y[1:].ravel())
    assert bool(np.all(shapely.covers(swath, inner)))
    values = np.ones((2, 3))
    column = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"x": np.arange(3.0), "y": np.arange(2.0)},
        attrs={"units": "kg m-2"},
    )
    anchor = resolve_anchor(
        column, geometry, mode="pixel", along=20, across=5, corner="lower left"
    )
    assert anchor.outside_fraction == 0.0
