"""Tests of the polygon-overlap and bucket conservative regridders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
import pytest
import shapely
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.geometry.corners import (
    cell_areas,
    cell_corner_nodes,
    cell_polygons,
    outline_polygon,
)
from spectra.geometry.crs import from_lonlat, spatial_ref_coordinate, to_lonlat
from spectra.geometry.placement import place_column
from spectra.geometry.swath import anchor_pixel_spacing, swath_polygon
from spectra.protocols import Regridder
from spectra.registry import get_regridder
from spectra.regrid import (
    BucketRegridder,
    PolygonRegridder,
    RegridResult,
    regrid_polygons,
    support_window,
)
from spectra.regrid.base import check_mass
from spectra.regrid.polygon import clip_areas
from spectra.regrid.windows import anchor_pixel_centre
from spectra.types import ANCHOR_CORNERS, Placement, SceneGeometry, Window

CRS = pyproj.CRS.from_epsg(32632)
ORIGIN = (500000.0, 5200000.0)

# Relative mass tolerances. Both methods conserve mass to rounding when the
# support lies inside the window: the polygon method by construction, the
# bucket method because every sub-cell is binned (measured <= 1e-14).
POLYGON_RTOL = 1e-9
BUCKET_RTOL = 1e-12

ROTATIONS = (0, 37, 90, 270, -20)
FLIPS = ("none", "x", "y")
# LES (nx, ny, dx, dy), pixel (across, along) size and scene (across, along).
CASES = {
    "coarsen_2m_to_10m": ((100, 60, 2.0, 2.0), (10.0, 10.0), (80, 80)),
    "refine_20m_to_5m": ((20, 12, 20.0, 20.0), (5.0, 5.0), (200, 200)),
    "anisotropic_1m_to_0.6x0.35m": ((40, 24, 1.0, 1.0), (0.6, 0.35), (200, 300)),
}
REGRIDDERS = {
    "polygon": (PolygonRegridder(), POLYGON_RTOL),
    "bucket": (BucketRegridder(supersample=8), BUCKET_RTOL),
}


def make_geometry(
    n_across: int,
    n_along: int,
    pixel_across_m: float,
    pixel_along_m: float,
    origin: tuple[float, float] = ORIGIN,
    rotation_deg: float = 0.0,
    crs: pyproj.CRS = CRS,
) -> SceneGeometry:
    """Pixel centres on a regular lattice: along -> +x, across -> +y, rotated."""
    across = np.arange(n_across, dtype=float)[:, None] * pixel_across_m
    along = np.arange(n_along, dtype=float)[None, :] * pixel_along_m
    u = np.broadcast_to(along, (n_across, n_along))
    v = np.broadcast_to(across, (n_across, n_along))
    angle = np.deg2rad(rotation_deg)
    x = origin[0] + np.cos(angle) * u - np.sin(angle) * v
    y = origin[1] + np.sin(angle) * u + np.cos(angle) * v
    dims = ("across", "along")
    return SceneGeometry(
        x=xr.DataArray(x, dims=dims), y=xr.DataArray(y, dims=dims), crs=crs
    )


def scene_from_geometry(geometry: SceneGeometry) -> xr.Dataset:
    """Wrap a geometry into the minimal dataset ``SceneGeometry.from_scene`` needs."""
    n_across, n_along = geometry.shape
    lon, lat = to_lonlat(geometry.crs, geometry.x.values, geometry.y.values)
    dims = ("across", "along")
    return xr.Dataset(
        coords={
            "across": np.arange(n_across),
            "along": np.arange(n_along),
            "x": (dims, geometry.x.values, {"units": "m"}),
            "y": (dims, geometry.y.values, {"units": "m"}),
            "lon": (dims, lon, {"units": "degrees_east"}),
            "lat": (dims, lat, {"units": "degrees_north"}),
            "spatial_ref": spatial_ref_coordinate(geometry.crs),
        },
        attrs={"scene_id": "synthetic"},
    )


def make_scene(
    n_across: int,
    n_along: int,
    pixel_across_m: float,
    pixel_along_m: float,
    **kwargs: object,
) -> xr.Dataset:
    return scene_from_geometry(
        make_geometry(n_across, n_along, pixel_across_m, pixel_along_m, **kwargs)  # type: ignore[arg-type]
    )


def make_column(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    pattern: str = "gaussian",
    units: str = "kg m-2",
) -> xr.DataArray:
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    if pattern == "ones":
        values = np.ones((ny, nx))
    elif pattern == "checker":
        values = ((np.arange(ny)[:, None] + np.arange(nx)[None, :]) % 2).astype(float)
    elif pattern == "gaussian":
        xx, yy = np.meshgrid(x, y)
        cx, cy = nx * dx / 2, ny * dy / 2
        sx, sy = nx * dx / 6, ny * dy / 6
        r2 = ((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2
        values = 1e-3 * np.exp(-0.5 * r2)
        values[r2 > 9] = 0.0
    else:
        raise ValueError(pattern)
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        attrs={"units": units},
        name="column",
    )


def integrated_mass(result: RegridResult) -> float:
    return float(np.sum(result.data.values * result.target_area))


def assert_provenance(data: xr.DataArray, placement: Placement, method: str) -> None:
    window = Window.from_attrs(data.attrs, prefix="window_")
    assert window.shape == data.shape
    for key, value in placement.attrs().items():
        assert data.attrs[key] == value
    assert data.attrs["regrid_method"] == method
    assert data.attrs["units"] == "kg m-2"
    for key in (
        "regrid_source_mass",
        "regrid_source_mass_inside",
        "regrid_target_mass",
        "regrid_inside_mass_fraction",
        "regrid_max_relative_mass_error",
    ):
        assert key in data.attrs


def conservation_params():
    params = []
    index = 0
    for case_name in CASES:
        for rotation in ROTATIONS:
            for flip in FLIPS:
                corner = ANCHOR_CORNERS[index % len(ANCHOR_CORNERS)]
                index += 1
                params.append(
                    pytest.param(
                        case_name,
                        rotation,
                        flip,
                        corner,
                        id=f"{case_name}-rot{rotation}-flip{flip}-"
                        f"{corner.replace(' ', '_')}",
                    )
                )
    return params


@pytest.mark.parametrize("method", list(REGRIDDERS))
@pytest.mark.parametrize(
    ("case_name", "rotation", "flip", "corner"), conservation_params()
)
def test_scene_regridding_conserves_mass(method, case_name, rotation, flip, corner):
    regridder, rtol = REGRIDDERS[method]
    (nx, ny, dx, dy), (pixel_across, pixel_along), (n_across, n_along) = CASES[
        case_name
    ]
    scene = make_scene(n_across, n_along, pixel_across, pixel_along)
    column = make_column(nx, ny, dx, dy, "gaussian")
    placement = Placement(
        anchor_along=n_along // 2,
        anchor_across=n_across // 2,
        anchor_corner=corner,
        rotation_deg=rotation,
        flip=flip,
    )
    result = regridder.regrid_to_scene(column, scene, placement)
    # Source areas come from the transformed nodes (1e-11 relative at 5e6 m).
    assert result.source_mass == pytest.approx(column.values.sum() * dx * dy, rel=1e-9)
    assert result.source_mass_inside == pytest.approx(result.source_mass, rel=rtol)
    assert abs(result.target_mass - result.source_mass) / result.source_mass < rtol
    assert result.relative_error < rtol
    assert integrated_mass(result) == pytest.approx(result.target_mass, rel=1e-12)
    check_mass(result, rtol)
    assert result.data.dims == ("across", "along")
    assert np.all(result.data.values >= 0)
    assert result.target_area.shape == result.data.shape
    assert result.source_area.shape == (ny, nx)
    assert_provenance(result.data, placement, method)
    for key in ("x", "y", "lon", "lat"):
        assert result.data.coords[key].dims == ("across", "along")
    window = Window.from_attrs(result.data.attrs)
    xr.testing.assert_equal(
        result.data.coords["x"].reset_coords(drop=True),
        scene["x"].isel(window.indexers).reset_coords(drop=True),
    )


@pytest.mark.parametrize("method", list(REGRIDDERS))
@pytest.mark.parametrize("corner", ANCHOR_CORNERS)
@pytest.mark.parametrize("pattern", ["ones", "checker"])
def test_anchor_corners_and_patterns_conserve_mass(method, corner, pattern):
    regridder, rtol = REGRIDDERS[method]
    scene = make_scene(80, 80, 10.0, 10.0)
    column = make_column(50, 30, 2.0, 2.0, pattern)
    placement = Placement(
        anchor_along=40, anchor_across=40, anchor_corner=corner, rotation_deg=37
    )
    result = regridder.regrid_to_scene(column, scene, placement)
    assert abs(result.target_mass - result.source_mass) / result.source_mass < rtol
    # The anchor pixel centre carries the selected corner cell centre.
    placed = place_column(
        column,
        crs=CRS,
        anchor_target_xy=anchor_pixel_centre(
            SceneGeometry.from_scene(scene), placement
        ),
        placement=placement,
    )
    row = -1 if corner.startswith("upper") else 0
    col = -1 if corner.endswith("right") else 0
    assert placed.centre_x[row, col] == pytest.approx(float(scene["x"][40, 40]))
    assert placed.centre_y[row, col] == pytest.approx(float(scene["y"][40, 40]))


def test_polygon_identity_on_matching_lattice():
    scene = make_scene(30, 40, 10.0, 10.0)
    column = make_column(8, 6, 10.0, 10.0, "ones")
    column.values[:] = np.arange(1, 49, dtype=float).reshape(6, 8)
    placement = Placement(anchor_along=4, anchor_across=3)
    result = PolygonRegridder().regrid_to_scene(column, scene, placement)
    block = result.data.sel(across=slice(3, 8), along=slice(4, 11))
    assert block.shape == (6, 8)
    assert np.allclose(block.values, column.values, rtol=1e-12, atol=0)
    outside = result.data.where(
        ~((result.data.across >= 3) & (result.data.across <= 8))
        | ~((result.data.along >= 4) & (result.data.along <= 11)),
        0.0,
    )
    assert np.all(outside.values == 0)
    assert result.relative_error == 0


def test_polygon_fracarea_extrapolates_at_support_edge():
    scene = make_scene(40, 40, 10.0, 10.0)
    column = make_column(15, 9, 2.0, 2.0, "ones")
    placement = Placement(anchor_along=20, anchor_across=20, rotation_deg=37)
    destarea = PolygonRegridder(norm="destarea").regrid_to_scene(
        column, scene, placement
    )
    fracarea = PolygonRegridder(norm="fracarea").regrid_to_scene(
        column, scene, placement
    )
    assert fracarea.data.attrs["regrid_norm"] == "fracarea"
    # Slivers below the 1e-9 overlap fraction are dropped under fracarea.
    assert np.all(
        (fracarea.data.values >= destarea.data.values - 1e-9)
        | (destarea.data.values < 1e-9)
    )
    assert fracarea.target_mass > destarea.target_mass
    # Partially covered pixels take the source value of a constant field.
    covered = fracarea.data.values > 0
    assert np.allclose(fracarea.data.values[covered], 1.0, rtol=1e-12)


@pytest.mark.parametrize("method", list(REGRIDDERS))
def test_partial_support_outside_scene(method):
    regridder, rtol = REGRIDDERS[method]
    scene = make_scene(40, 40, 10.0, 10.0)
    column = make_column(100, 60, 2.0, 2.0, "gaussian")
    placement = Placement(anchor_along=30, anchor_across=35, anchor_corner="lower left")
    result = regridder.regrid_to_scene(column, scene, placement)
    assert result.source_mass_inside < result.source_mass
    assert 0.1 < result.inside_mass_fraction < 0.9
    assert (
        abs(result.target_mass - result.source_mass_inside) < rtol * result.source_mass
    )
    assert result.relative_error < rtol
    check_mass(result, rtol)
    assert result.data.attrs["window_across_stop"] == 40
    assert result.data.attrs["window_along_stop"] == 40
    reference = PolygonRegridder().regrid_to_scene(column, scene, placement)
    # The bucket classifies sub-cells of 0.25 m; the polygon overlap is exact.
    assert result.source_mass_inside == pytest.approx(
        reference.source_mass_inside, rel=5e-3
    )
    fracarea = PolygonRegridder(norm="fracarea").regrid_to_scene(
        column, scene, placement
    )
    assert fracarea.target_mass >= reference.target_mass


def test_polygon_and_bucket_agree_per_pixel():
    scene = make_scene(80, 80, 10.0, 10.0)
    column = make_column(100, 60, 2.0, 2.0, "gaussian")
    placement = Placement(anchor_along=40, anchor_across=40, rotation_deg=37, flip="y")
    polygon = PolygonRegridder().regrid_to_scene(column, scene, placement)
    window = Window.from_attrs(polygon.data.attrs)
    bucket = BucketRegridder(supersample=8).regrid_to_scene(
        column, scene, placement, window=window
    )
    assert bucket.data.shape == polygon.data.shape
    difference = np.abs(bucket.data.values - polygon.data.values).max()
    assert difference / polygon.data.values.max() < 0.02
    assert bucket.target_mass == pytest.approx(polygon.target_mass, rel=1e-12)


@pytest.mark.parametrize("method", list(REGRIDDERS))
@pytest.mark.parametrize("rotation", [0, 37])
def test_rotated_target_lattice_conserves_mass(method, rotation):
    regridder, rtol = REGRIDDERS[method]
    scene = make_scene(80, 80, 10.0, 10.0, rotation_deg=30.0)
    column = make_column(100, 60, 2.0, 2.0, "gaussian")
    placement = Placement(anchor_along=40, anchor_across=40, rotation_deg=rotation)
    result = regridder.regrid_to_scene(column, scene, placement)
    assert abs(result.target_mass - result.source_mass) / result.source_mass < rtol
    assert result.relative_error < rtol
    geometry = SceneGeometry.from_scene(scene)
    expected = outline_polygon(
        cell_corner_nodes(geometry.x.values), cell_corner_nodes(geometry.y.values)
    )
    assert swath_polygon(geometry).symmetric_difference(expected).area < 1e-6


def test_to_scene_honours_explicit_window():
    scene = make_scene(80, 80, 10.0, 10.0)
    column = make_column(100, 60, 2.0, 2.0, "gaussian")
    placement = Placement(anchor_along=40, anchor_across=40, rotation_deg=37)
    window = Window(across=slice(30, 60), along=slice(35, 70))
    data = PolygonRegridder().to_scene(column, scene, placement, window=window)
    assert data.shape == window.shape
    assert Window.from_attrs(data.attrs) == window
    assert np.array_equal(data["along"].values, np.arange(35, 70))
    # A window that leaves part of the support out still conserves what it covers.
    narrow = Window(across=slice(40, 50), along=slice(40, 50))
    result = PolygonRegridder().regrid_to_scene(column, scene, placement, window=narrow)
    assert result.source_mass_inside < result.source_mass
    assert result.relative_error < POLYGON_RTOL
    with pytest.raises(ContractError):
        PolygonRegridder().to_scene(
            column, scene, placement, window=Window(slice(0, 90), slice(0, 10))
        )


def test_registry_exposes_protocol_compliant_regridders():
    for name in ("polygon", "bucket"):
        regridder = get_regridder(name)
        assert isinstance(regridder, Regridder)
        assert regridder.name == name
    assert get_regridder("bucket", supersample=2).supersample == 2  # type: ignore[attr-defined]
    assert get_regridder("polygon", norm="fracarea").norm == "fracarea"  # type: ignore[attr-defined]
    assert isinstance(PolygonRegridder(), Regridder)
    assert isinstance(BucketRegridder(), Regridder)


def test_invalid_options_and_placements_raise():
    with pytest.raises(ContractError):
        PolygonRegridder(norm="sum")  # type: ignore[arg-type]
    with pytest.raises(ContractError):
        PolygonRegridder(engine="gpu")  # type: ignore[arg-type]
    with pytest.raises(ContractError):
        PolygonRegridder(rtol=-1.0)
    with pytest.raises(ContractError):
        BucketRegridder(supersample=0)
    scene = make_scene(20, 20, 10.0, 10.0)
    column = make_column(10, 6, 2.0, 2.0, "ones")
    with pytest.raises(ContractError, match="outside"):
        PolygonRegridder().to_scene(column, scene, Placement(anchor_along=20))
    with pytest.raises(ContractError, match="outside"):
        BucketRegridder().to_scene(column, scene, Placement(anchor_across=25))
    nan_scene = scene.copy(deep=True)
    nan_scene["x"][3, 3] = np.nan
    with pytest.raises(ContractError, match="finite"):
        PolygonRegridder().to_scene(
            column, nan_scene, Placement(anchor_across=3, anchor_along=3)
        )


def brute_force_window(placed, geometry: SceneGeometry, margin: int = 1) -> Window:
    """Reference window from fully loaded centres (same rule as support_window)."""
    spacing = anchor_pixel_spacing(
        geometry,
        along=placed.placement.anchor_along,
        across=placed.placement.anchor_across,
    )
    xmin, ymin, xmax, ymax = placed.support_polygon.bounds
    x = np.asarray(geometry.x.values)
    y = np.asarray(geometry.y.values)
    inside = (
        (x >= xmin - spacing)
        & (x <= xmax + spacing)
        & (y >= ymin - spacing)
        & (y <= ymax + spacing)
    )
    rows, cols = np.nonzero(inside)
    n_across, n_along = geometry.shape
    return Window(
        across=slice(
            max(0, rows.min() - margin), min(n_across, rows.max() + 1 + margin)
        ),
        along=slice(max(0, cols.min() - margin), min(n_along, cols.max() + 1 + margin)),
    )


def test_support_window_matches_brute_force_on_large_lazy_swath():
    dask = pytest.importorskip("dask.array")
    n_across, n_along = 600, 3000
    i = dask.arange(n_across, chunks=128, dtype=float)[:, None]
    j = dask.arange(n_along, chunks=512, dtype=float)[None, :]
    angle = np.deg2rad(25.0)
    u = 0.35 * j + 0.0 * i
    v = 0.6 * i + 0.0 * j
    x = ORIGIN[0] + np.cos(angle) * u - np.sin(angle) * v
    y = ORIGIN[1] + np.sin(angle) * u + np.cos(angle) * v
    dims = ("across", "along")
    lazy = SceneGeometry(
        x=xr.DataArray(x, dims=dims), y=xr.DataArray(y, dims=dims), crs=CRS
    )
    loaded = SceneGeometry(
        x=xr.DataArray(np.asarray(x), dims=dims),
        y=xr.DataArray(np.asarray(y), dims=dims),
        crs=CRS,
    )
    column = make_column(30, 12, 1.0, 1.0, "ones")
    for placement in (
        Placement(anchor_along=1500, anchor_across=300, rotation_deg=37),
        Placement(anchor_along=20, anchor_across=10, rotation_deg=270, flip="x"),
        Placement(anchor_along=2990, anchor_across=590, anchor_corner="upper right"),
    ):
        placed = place_column(
            column,
            crs=CRS,
            anchor_target_xy=anchor_pixel_centre(lazy, placement),
            placement=placement,
        )
        window = support_window(placed, lazy)
        assert window == brute_force_window(placed, loaded)
        assert support_window(placed, loaded) == window
    far = Placement(anchor_along=2999, anchor_across=599, anchor_corner="lower left")
    placed = place_column(
        column, crs=CRS, anchor_target_xy=anchor_pixel_centre(lazy, far), placement=far
    )
    window = support_window(placed, lazy)
    assert window.across.stop == n_across and window.along.stop == n_along


def test_support_window_rejects_support_outside_scene():
    geometry = make_geometry(20, 20, 10.0, 10.0)
    column = make_column(10, 6, 2.0, 2.0, "ones")
    placement = Placement(anchor_along=0, anchor_across=0, rotation_deg=180)
    placed = place_column(
        column,
        crs=CRS,
        anchor_target_xy=(ORIGIN[0] - 500.0, ORIGIN[1] - 500.0),
        placement=placement,
    )
    with pytest.raises(ContractError, match="overlap"):
        support_window(placed, geometry)


def random_quads(rng: np.random.Generator, count: int, offset: float) -> np.ndarray:
    centre = rng.uniform(-1.0, 1.0, (count, 2)) + offset
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    size = rng.uniform(0.2, 1.5, (count, 2))
    base = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    corners = base[None] * size[:, None, :]
    rotation = np.stack(
        (np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)), axis=1
    ).reshape(count, 2, 2)
    quads = np.einsum("nij,nkj->nki", rotation, corners) + centre[:, None, :]
    clockwise = rng.random(count) < 0.5
    quads[clockwise] = quads[clockwise][:, ::-1]
    return quads


@pytest.mark.parametrize("offset", [0.0, 5.2e6])
def test_clip_areas_match_shapely(offset):
    rng = np.random.default_rng(7)
    clip = shapely.polygons(random_quads(rng, 5000, offset))
    subject = shapely.polygons(random_quads(rng, 5000, offset))
    arrow = np.array([[0, 0], [2, 0], [1, 0.5], [2, 1], [0, 1]], dtype=float)
    concave = shapely.polygons(
        arrow[None] + rng.uniform(-1.0, 1.0, (5000, 1, 2)) + offset
    )
    # Intersection points inherit the coordinate precision (1e-9 m at 5e6 m).
    atol = 1e-12 + 1e-15 * offset
    for polygons in (subject, concave, clip):
        expected = shapely.area(shapely.intersection(polygons, clip))
        rings = shapely.get_coordinates(polygons).reshape(5000, -1, 2)[:, :-1]
        clips = shapely.get_coordinates(clip).reshape(5000, -1, 2)[:, :-1]
        got = clip_areas(rings, clips)
        assert np.abs(got - expected).max() < atol
        assert np.count_nonzero(expected) > 1000


def test_polygon_engines_agree():
    scene = make_scene(60, 60, 10.0, 10.0, rotation_deg=30.0)
    column = make_column(50, 30, 2.0, 2.0, "gaussian")
    placement = Placement(anchor_along=30, anchor_across=30, rotation_deg=37, flip="x")
    clip = PolygonRegridder(engine="clip").regrid_to_scene(column, scene, placement)
    overlay = PolygonRegridder(engine="shapely").regrid_to_scene(
        column, scene, placement
    )
    assert clip.data.attrs["regrid_engine"] == "clip"
    scale = overlay.data.values.max()
    assert np.abs(clip.data.values - overlay.data.values).max() / scale < 1e-9
    assert clip.source_mass_inside == pytest.approx(
        overlay.source_mass_inside, rel=1e-9
    )
    with pytest.raises(ContractError):
        regrid_polygons(
            np.asarray([shapely.Polygon([(0, 0), (2, 0), (1, 0.5), (2, 1), (0, 1)])]),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1, 4)),
            np.zeros((1, 1, 4)),
            engine="clip",
        )


def test_regrid_polygons_bookkeeping():
    nodes = np.arange(5.0)
    node_x, node_y = np.meshgrid(nodes, nodes, indexing="xy")
    vx = np.stack(
        (node_x[:-1, :-1], node_x[:-1, 1:], node_x[1:, 1:], node_x[1:, :-1]), axis=-1
    )
    vy = np.stack(
        (node_y[:-1, :-1], node_y[:-1, 1:], node_y[1:, 1:], node_y[1:, :-1]), axis=-1
    )
    polygons = cell_polygons(vx, vy)
    values = np.zeros((4, 4))
    values[1, 1] = 2.0
    values[2, 3] = 3.0
    areas = cell_areas(vx, vy)
    # Shifted target grid: every target cell overlaps up to four source cells.
    target_vx = vx[:3, :3] + 0.5
    target_vy = vy[:3, :3] + 0.5
    for engine in ("clip", "shapely"):
        out, source_mass, inside, target_mass = regrid_polygons(
            polygons, values, areas, target_vx, target_vy, engine=engine
        )
        assert source_mass == 5.0
        assert inside == pytest.approx(2.0 + 3.0 * 0.5)
        assert target_mass == pytest.approx(inside)
        assert out[0, 0] == pytest.approx(0.5)
        assert out[1, 1] == pytest.approx(0.5)
        assert out[2, 2] == pytest.approx(0.75)
        assert np.count_nonzero(out) == 6
    out, *_ = regrid_polygons(
        polygons, values, areas, target_vx, target_vy, norm="fracarea"
    )
    assert out[0, 0] == pytest.approx(0.5)
    empty, source_mass, inside, target_mass = regrid_polygons(
        polygons, np.zeros((4, 4)), areas, target_vx, target_vy
    )
    assert np.all(empty == 0) and source_mass == inside == target_mass == 0.0
    with pytest.raises(ContractError):
        regrid_polygons(polygons, values[:2], areas, target_vx, target_vy)


@pytest.mark.data
def test_polygon_regridder_matches_cdo_remapcon(data_root: Path):
    """Compare ``fracarea`` overlap against ``cdo remapcon`` on real geometry.

    Measured on the reference files: the bulk of the 450 x 1050 window agrees
    to float32 precision; the worst pixels differ by 6e-4 of the maximum where
    CDO/YAC evaluates spherical areas of ~1e-15 sr cells (planar and
    ellipsoidal weights agree to 2e-5 there), and 42 edge pixels overlapped by
    at most 1 % of their area are left missing by CDO.
    """
    names = ("microhh_cdo.nc", "microhh_grid_aviris.nc", "microhh_remapped.nc")
    paths = [data_root / name for name in names]
    if not all(path.exists() for path in paths):
        pytest.skip("CDO comparison files are missing")
    source = xr.open_dataset(paths[0], decode_times=False)
    target = xr.open_dataset(paths[1], decode_times=False)
    remapped = xr.open_dataset(paths[2], decode_times=False)
    crs = pyproj.CRS.from_user_input(source.attrs["embedding_crs"])
    source_vx, source_vy = from_lonlat(
        crs, source["lonc"].values, source["latc"].values
    )
    target_vx, target_vy = from_lonlat(
        crs, target["lonc"].values, target["latc"].values
    )
    values = np.nan_to_num(source["ch4_path"].values.astype(np.float64))
    areas = cell_areas(source_vx, source_vy)
    polygons = cell_polygons(source_vx, source_vy)
    ours, source_mass, inside, target_mass = regrid_polygons(
        polygons, values, areas, target_vx, target_vy, norm="fracarea"
    )
    exact, _, _, exact_mass = regrid_polygons(
        polygons, values, areas, target_vx, target_vy, norm="destarea"
    )
    assert inside == pytest.approx(source_mass, rel=1e-9)
    assert exact_mass == pytest.approx(source_mass, rel=1e-9)
    cdo = remapped["ch4_path"].values.astype(np.float64)
    valid = np.isfinite(cdo)
    scale = float(np.nanmax(cdo))
    target_areas = cell_areas(target_vx, target_vy)
    relative = np.abs(ours - np.nan_to_num(cdo)) / scale
    assert relative[valid].max() < 1e-3
    assert np.percentile(relative[valid], 99.9) < 1e-4
    assert np.percentile(relative[valid], 99) < 1e-5
    cdo_mass = float(np.sum(cdo[valid] * target_areas[valid]))
    our_mass = float(np.sum(ours[valid] * target_areas[valid]))
    assert abs(our_mass - cdo_mass) / cdo_mass < 2e-6
    assert abs(target_mass - cdo_mass) / cdo_mass < 1e-4
    # Pixels CDO leaves missing are barely overlapped by the source domain.
    missing = ~valid & (ours != 0)
    fraction = exact[missing] / ours[missing]
    assert fraction.max() < 0.02
    assert np.sum(ours[missing] * target_areas[missing]) / target_mass < 1e-4
    assert not np.any(valid & (ours == 0))
