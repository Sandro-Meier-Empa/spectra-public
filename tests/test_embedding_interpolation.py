"""Interpolation of the optical depth between absorption-table nodes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from spectra.spectral.embedding import EMBEDDING_VERSION, embed_column_in_radiance
from spectra.spectral.toy import ToyForwardModel
from spectra.types import SpectralResponse, ViewingGeometry

GEOMETRY = ViewingGeometry(
    sza=40.0, vza=0.0, surface_elevation=0.0, sensor_altitude=3000.0
)
RESPONSE = SpectralResponse(
    wavelength=np.linspace(2280.0, 2320.0, 9), fwhm=np.full(9, 5.0)
)
# Strong saturation, so that the optical depth is clearly convex in the amount.
MODEL = ToyForwardModel(saturation_ppm_m=2.0e4)


def toy_table(version: str = EMBEDDING_VERSION) -> xr.Dataset:
    """Return the toy absorption table, labelled with an embedding version."""
    table = MODEL.lut(RESPONSE, GEOMETRY, gas="ch4")
    table.attrs["embedding_version"] = version
    return table


def embedded_transmittance(table: xr.Dataset, amount: np.ndarray) -> np.ndarray:
    """Embed vertical path amounts [ppm m] into unit radiance, (amount, band)."""
    radiance = xr.DataArray(
        np.ones((1, amount.size, RESPONSE.wavelength.size)),
        dims=("across", "along", "band"),
        coords={"wavelength": ("band", RESPONSE.wavelength)},
    )
    column = xr.DataArray(
        amount[np.newaxis, :], dims=("across", "along"), attrs={"units": "ppm-m"}
    )
    return embed_column_in_radiance(radiance, column, table).values[0]


def exact_transmittance(table: xr.Dataset, amount: np.ndarray) -> np.ndarray:
    """Return the toy model transmittance for vertical path amounts [ppm m]."""
    shape = MODEL.line_shape(RESPONSE.wavelength)
    slant = amount * table.attrs["geometric_air_mass_factor"]
    depth = -shape * amount[:, np.newaxis] / (1.0 + slant[:, np.newaxis] / 2.0e4)
    return np.exp(depth)


def midpoints(table: xr.Dataset) -> np.ndarray:
    """Return the geometric midpoints [ppm m] between the table nodes."""
    nodes = table["vertical_amount"].values
    return np.sqrt(nodes[:-1] * nodes[1:])


def test_new_tables_use_pchip_version() -> None:
    """Tables built now carry the version that interpolates with PCHIP."""
    assert MODEL.lut(RESPONSE, GEOMETRY, gas="ch4").attrs["embedding_version"] == (
        "lambert-beer-v2"
    )


def test_tables_are_exact_at_nodes() -> None:
    """Both interpolations reproduce the forward model at the table nodes."""
    nodes = toy_table()["vertical_amount"].values
    for version in ("lambert-beer-v1", "lambert-beer-v2"):
        table = toy_table(version)
        np.testing.assert_allclose(
            embedded_transmittance(table, nodes),
            exact_transmittance(table, nodes),
            rtol=1e-6,
        )


def test_pchip_follows_saturation_between_nodes() -> None:
    """PCHIP is far closer to the forward model between nodes than linear."""
    table = toy_table()
    amount = midpoints(table)
    exact = exact_transmittance(table, amount)

    linear_error = np.abs(
        embedded_transmittance(toy_table("lambert-beer-v1"), amount) / exact - 1
    ).max()
    pchip_error = np.abs(embedded_transmittance(table, amount) / exact - 1).max()

    assert linear_error > 5e-3  # v1 under-attenuates the saturated lines
    assert pchip_error < 0.15 * linear_error


def test_v1_tables_keep_linear_interpolation() -> None:
    """Tables of version 1 interpolate the optical depth linearly in the amount."""
    table = toy_table("lambert-beer-v1")
    nodes = table["vertical_amount"].values
    amount = midpoints(table)[5:6]
    lower, upper = nodes[5], nodes[6]
    depth = table["optical_depth"].values
    weight = (amount[0] - lower) / (upper - lower)
    expected = np.exp((1 - weight) * depth[5] + weight * depth[6])

    np.testing.assert_allclose(
        embedded_transmittance(table, amount)[0], expected, rtol=1e-6
    )


def test_unknown_embedding_version_raises() -> None:
    """A table of an unknown version cannot be embedded."""
    with pytest.raises(ValueError, match="unknown embedding_version"):
        embedded_transmittance(toy_table("lambert-beer-v0"), np.array([1.0]))
