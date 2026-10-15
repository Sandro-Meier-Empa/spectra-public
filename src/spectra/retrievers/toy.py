"""Synthetic retriever: a per-pixel log-radiance regression on a target spectrum.

The retriever is a deterministic test double, not a scientific algorithm.
For every pixel it fits ``ln(L / L_bg) = a * k(lambda) + c`` by least
squares, where ``L_bg`` is the mean spectrum of the pixel's across-track
column and ``k`` the target unit absorption spectrum, and reports ``a`` in
kg m-2 together with a per-column residual uncertainty.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import spectra
from spectra.contracts.column_product import (
    CANONICAL_COLUMN_UNITS,
    COLUMN_PRODUCT_CONTRACT,
    COLUMN_PRODUCT_VERSION,
    PPM_METRE_UNITS,
    convert_column_units,
)
from spectra.contracts.errors import ContractError
from spectra.contracts.lut import validate_unit_absorption_spectrum
from spectra.contracts.sensor_scene import RADIANCE_UNITS, SPATIAL_DIMS
from spectra.testing.synthetic import scene_spatial_coords
from spectra.types import RetrievalSettings, SpectralResponse

GAS = "CH4"
_MIN_BANDS = 3
_WAVELENGTH_ATOL_NM = 1e-6


def _radiance(scene: xr.Dataset) -> np.ndarray:
    """Return the radiance as a float64 ``(across, along, band)`` array."""
    radiance = scene["radiance"].transpose("across", "along", "band")
    return np.asarray(radiance.values, dtype=np.float64)


def _valid(scene: xr.Dataset) -> np.ndarray:
    """Return the ``valid`` mask as a boolean ``(across, along)`` array."""
    return np.asarray(scene["valid"].transpose(*SPATIAL_DIMS).values, dtype=bool)


def _select_target(
    scene: xr.Dataset, target: xr.Dataset, settings: RetrievalSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scene band indices and the target spectrum sampled on them.

    Bands inside ``settings.band_windows_nm`` whose wavelengths occur in the
    target are used directly. When the target grid differs and
    ``settings.options["interpolate_target"]`` is true (the default), the
    target is interpolated linearly onto the scene bands inside its range;
    otherwise only the exactly matching bands are kept.
    """
    try:
        validate_unit_absorption_spectrum(target)
    except ContractError as error:
        raise ContractError(f"Invalid retrieval target: {error}") from error
    response = SpectralResponse.from_scene(scene)
    candidates = response.select(settings.band_windows_nm)
    if candidates.size == 0:
        raise ContractError("No scene band lies inside settings.band_windows_nm.")
    scene_wavelength = response.wavelength[candidates]
    target_wavelength = np.asarray(target["wavelength"].values, dtype=float)
    target_values = np.asarray(target["unit_abs_spect"].values, dtype=float)
    matches = np.isclose(
        scene_wavelength[:, None],
        target_wavelength[None, :],
        rtol=0.0,
        atol=_WAVELENGTH_ATOL_NM,
    )
    matched = matches.any(axis=1)
    interpolate = bool(settings.options.get("interpolate_target", True))
    if matched.all():
        bands = candidates
        values = target_values[matches.argmax(axis=1)]
    elif interpolate:
        inside = (scene_wavelength >= target_wavelength[0]) & (
            scene_wavelength <= target_wavelength[-1]
        )
        bands = candidates[inside]
        values = np.interp(scene_wavelength[inside], target_wavelength, target_values)
    elif matched.any():
        bands = candidates[matched]
        values = target_values[matches.argmax(axis=1)[matched]]
    else:
        raise ContractError(
            "The target wavelengths match no scene band and interpolation is off."
        )
    if bands.size < _MIN_BANDS:
        raise ContractError(
            f"The toy retriever needs at least {_MIN_BANDS} bands covered by the "
            f"target; got {bands.size}."
        )
    return bands, values


class ToyRetriever:
    """Deterministic log-radiance regression retriever (``Retriever`` protocol).

    Args:
        uncertainty_floor_kg_m2: Smallest reported uncertainty, so noise-free
            scenes still yield a positive uncertainty.
    """

    name: str = "toy"
    version: str = spectra.__version__

    def __init__(self, *, uncertainty_floor_kg_m2: float = 1e-9) -> None:
        if not uncertainty_floor_kg_m2 > 0:
            raise ValueError("uncertainty_floor_kg_m2 must be positive.")
        self.uncertainty_floor_kg_m2 = float(uncertainty_floor_kg_m2)

    def background_statistics(
        self, scene: xr.Dataset, *, settings: RetrievalSettings
    ) -> xr.Dataset:
        """Return the mean valid spectrum of every across-track column.

        Args:
            scene: ``SensorScene`` dataset assumed free of plumes.
            settings: Retrieval settings; recorded in the attributes.

        Returns:
            Dataset with ``background_mean(across, band)`` on all scene bands
            and ``n_samples(across)``.
        """
        radiance = _radiance(scene)
        usable = _valid(scene)[:, :, None] & np.isfinite(radiance) & (radiance > 0)
        total = np.where(usable, radiance, 0.0).sum(axis=1)
        count = usable.sum(axis=1)
        mean = np.divide(
            total, count, out=np.full(total.shape, np.nan), where=count > 0
        )
        return xr.Dataset(
            {
                "background_mean": (
                    ("across", "band"),
                    mean,
                    {"units": RADIANCE_UNITS, "long_name": "mean column spectrum"},
                ),
                "n_samples": (
                    ("across",),
                    count.min(axis=1),
                    {"long_name": "valid pixels per across-track column"},
                ),
            },
            coords={"wavelength": scene["wavelength"], "fwhm": scene["fwhm"]},
            attrs={
                "retriever": self.name,
                "retriever_version": self.version,
                "sensor": str(scene.attrs.get("sensor", "")),
                "scene_id": str(scene.attrs.get("scene_id", "")),
                "statistics_scope": settings.statistics_scope,
            },
        )

    def retrieve(
        self,
        scene: xr.Dataset,
        target: xr.Dataset,
        *,
        settings: RetrievalSettings,
        background_stats: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Retrieve the column enhancement of every pixel.

        Args:
            scene: ``SensorScene`` dataset.
            target: Unit absorption spectrum with ``unit_abs_spect(wavelength)``
                in ppm-1 m-1 (negative values).
            settings: Retrieval settings; ``band_windows_nm`` selects the bands
                and ``options["interpolate_target"]`` allows resampling the
                target onto the scene bands.
            background_stats: Output of ``background_statistics``; computed
                from the scene itself when omitted.

        Returns:
            Sensor-space ``ColumnProduct`` with ``ch4_mass``,
            ``ch4_mass_uncertainty`` and ``retrieval_valid``.

        Raises:
            ContractError: If the target or the statistics do not fit the scene.
        """
        bands, spectrum = _select_target(scene, target, settings)
        stats = (
            self.background_statistics(scene, settings=settings)
            if background_stats is None
            else background_stats
        )
        if "background_mean" not in stats:
            raise ContractError("background_stats must contain 'background_mean'.")
        mean = stats["background_mean"]
        if set(mean.dims) != {"across", "band"}:
            raise ContractError("'background_mean' must have dims (across, band).")
        mean_values = np.asarray(mean.transpose("across", "band").values, dtype=float)
        if mean_values.shape != (scene.sizes["across"], scene.sizes["band"]):
            raise ContractError("'background_mean' does not match the scene shape.")

        radiance = _radiance(scene)[:, :, bands]
        background = mean_values[:, bands]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(radiance / background[:, None, :])
        # Centring both sides over the bands fits an intercept, which absorbs
        # spectrally flat albedo differences between the pixel and its column.
        kernel = spectrum - spectrum.mean()
        kernel_norm = float(np.sum(kernel * kernel))
        if kernel_norm <= 0:
            raise ContractError("The target spectrum is constant on the used bands.")
        centred = log_ratio - log_ratio.mean(axis=-1, keepdims=True)
        enhancement = centred @ kernel / kernel_norm
        residual = centred - enhancement[:, :, None] * kernel[None, None, :]

        valid = _valid(scene) & np.isfinite(enhancement)
        n_bands = int(bands.size)
        dof = max(n_bands - 2, 1)
        usable = valid[:, :, None] & np.isfinite(residual)
        n_across = usable.shape[0]
        squares = np.where(usable, residual**2, 0.0).reshape(n_across, -1).sum(axis=1)
        counts = usable.reshape(n_across, -1).sum(axis=1)
        column_variance = np.divide(
            squares, counts, out=np.zeros(n_across), where=counts > 0
        ) * (n_bands / dof)
        sigma_ppm_m = np.sqrt(column_variance) / np.sqrt(kernel_norm)

        enhancement_ppm_m = xr.DataArray(
            np.where(valid, enhancement, np.nan),
            dims=SPATIAL_DIMS,
            attrs={"units": PPM_METRE_UNITS},
        )
        uncertainty_ppm_m = xr.DataArray(
            np.broadcast_to(sigma_ppm_m[:, None], enhancement.shape).copy(),
            dims=SPATIAL_DIMS,
            attrs={"units": PPM_METRE_UNITS},
        )
        mass = convert_column_units(enhancement_ppm_m, CANONICAL_COLUMN_UNITS, gas=GAS)
        uncertainty = convert_column_units(
            uncertainty_ppm_m, CANONICAL_COLUMN_UNITS, gas=GAS
        )
        factor = float(
            convert_column_units(
                xr.DataArray(1.0, attrs={"units": PPM_METRE_UNITS}),
                CANONICAL_COLUMN_UNITS,
                gas=GAS,
            ).values
        )
        uncertainty_values = np.maximum(
            np.asarray(uncertainty.values, dtype=float), self.uncertainty_floor_kg_m2
        )
        valid &= np.isfinite(uncertainty_values)
        conversion = f"ppm_m_to_kg_m2={factor:.6e}; p=101325 Pa; T=288.15 K"

        product = xr.Dataset(
            {
                "ch4_mass": (
                    SPATIAL_DIMS,
                    np.asarray(mass.values, dtype=np.float32),
                    {
                        "units": CANONICAL_COLUMN_UNITS,
                        "long_name": f"{GAS} column enhancement",
                        "native_units": PPM_METRE_UNITS,
                        "native_variable": "enhancement",
                        "column_conversion": conversion,
                    },
                ),
                "ch4_mass_uncertainty": (
                    SPATIAL_DIMS,
                    uncertainty_values.astype(np.float32),
                    {
                        "units": CANONICAL_COLUMN_UNITS,
                        "long_name": f"{GAS} column uncertainty (1 sigma)",
                        "native_units": PPM_METRE_UNITS,
                        "column_conversion": conversion,
                    },
                ),
                "retrieval_valid": (
                    SPATIAL_DIMS,
                    valid,
                    {"long_name": "retrieval succeeded"},
                ),
                "pixel_area": scene["pixel_area"].transpose(*SPATIAL_DIMS),
            },
            coords=scene_spatial_coords(scene),
            attrs={
                "contract": COLUMN_PRODUCT_CONTRACT,
                "contract_version": COLUMN_PRODUCT_VERSION,
                "sensor": str(scene.attrs["sensor"]),
                "platform": str(scene.attrs.get("platform", "airborne")),
                "scene_id": str(scene.attrs["scene_id"]),
                "acquisition_time": str(scene.attrs.get("acquisition_time", "")),
                "background_group": str(scene.attrs["background_group"]),
                "geometry_space": "sensor",
                "gas": GAS,
                "embedding_method": "none",
                "retriever": self.name,
                "retriever_version": self.version,
                "retrieval_statistics_scope": settings.statistics_scope,
                "retrieval_n_bands": n_bands,
                "retrieval_wavelength_min_nm": float(
                    scene["wavelength"].values[bands].min()
                ),
                "retrieval_wavelength_max_nm": float(
                    scene["wavelength"].values[bands].max()
                ),
                "uncertainty_source": "toy per-column residual",
                "column_conversion": conversion,
            },
        )
        return product


__all__ = ["GAS", "ToyRetriever"]
