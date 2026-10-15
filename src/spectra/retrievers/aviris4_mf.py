"""Retriever wrapping the ``aviris4_ch4`` matched filter."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import ModuleType

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError, MissingExtraError
from spectra.contracts.lut import validate_unit_absorption_spectrum
from spectra.contracts.sensor_scene import select_bands
from spectra.sensors.aviris4 import mf_product_to_column_product
from spectra.types import RetrievalSettings

DEFAULT_CHUNKS: dict[str, int] = {"across": 8, "along": -1, "band": -1}


def _aviris4_mf() -> ModuleType:
    """Import ``aviris4_ch4.mf`` lazily."""
    try:
        import aviris4_ch4.mf as mf
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise MissingExtraError("aviris4", "aviris4_ch4", error) from error
    return mf


def _target_for_bands(target: xr.Dataset, wavelength: np.ndarray) -> xr.Dataset:
    """Return the target spectrum on exactly the scene's selected wavelengths."""
    validate_unit_absorption_spectrum(target)
    available = np.asarray(target["wavelength"].values, dtype=float)
    missing = wavelength[~np.isin(wavelength, available)]
    if missing.size:
        raise ContractError(
            "Target spectrum lacks the selected scene wavelengths "
            f"(first missing: {missing[:3]}); compute the target for this scene."
        )
    selected = target.sel(wavelength=wavelength)
    if "gas" not in selected:
        selected["gas"] = xr.DataArray(
            np.array(str(target.attrs.get("gas", "CH4")), dtype=object)
        )
    return selected


class Aviris4MatchedFilter:
    """Matched-filter retrieval of ``aviris4_ch4`` exposed as a SPECTRA retriever.

    Args:
        noise_model_file: Parametric noise-model file for the uncertainty
            estimate (``aviris4_ch4`` ships EMIT's model).
        mean_function: Background statistic, ``"median"`` or ``"mean"``.
        albedo_correction: Apply the Foote et al. albedo correction.
        sensitivity_correction: Apply the Fahlen et al. sensitivity correction.
        lognormal: Filter log-transformed radiances.
        num_iter: Iterative refinement passes.
        mask_shadows: Exclude shadowed pixels (scene ``shadow`` mask).
        diffmf: Also retrieve DiffMF derivative maps.
        parallel: Use Dask-vectorised row processing.
        chunks: Chunks applied to the radiance before filtering.
        **options: Further ``aviris4_ch4.mf.MatchedFilter`` keyword arguments.
    """

    name = "aviris4_mf"

    def __init__(
        self,
        *,
        noise_model_file: str | None = None,
        mean_function: str = "median",
        albedo_correction: bool = True,
        sensitivity_correction: bool = False,
        lognormal: bool = False,
        num_iter: int = 0,
        mask_shadows: bool = True,
        diffmf: bool = False,
        parallel: bool = True,
        chunks: Mapping[str, int] | None = None,
        **options: object,
    ) -> None:
        """Store the matched-filter configuration."""
        self.noise_model_file = noise_model_file
        self.mean_function = mean_function
        self.albedo_correction = albedo_correction
        self.sensitivity_correction = sensitivity_correction
        self.lognormal = lognormal
        self.num_iter = num_iter
        self.mask_shadows = mask_shadows
        self.diffmf = diffmf
        self.parallel = parallel
        self.chunks = dict(chunks or DEFAULT_CHUNKS)
        self.options = dict(options)

    def _filter_input(
        self, scene: xr.Dataset, settings: RetrievalSettings
    ) -> xr.Dataset:
        """Return the dataset handed to ``MatchedFilter.apply_matched_filter``."""
        subset = select_bands(scene, settings.band_windows_nm)
        data = subset[["radiance"]]
        if self.mask_shadows and "shadow" in scene:
            data["shadow_mask"] = scene["shadow"].astype(bool)
            data["shadow_mask"].attrs = dict(scene["shadow"].attrs)
        if settings.exclude_injected_from_background and "plume_mask" in scene:
            data["plume_mask"] = scene["plume_mask"].astype(bool)
        if data["radiance"].chunks is not None:
            data = data.chunk(self.chunks)
        return data

    def retrieve(
        self,
        scene: xr.Dataset,
        target: xr.Dataset,
        *,
        settings: RetrievalSettings,
        background_stats: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Run the matched filter on a scene and return a ``ColumnProduct``.

        Background statistics are computed from the data handed to the filter
        (``statistics_scope="window"``). Cached full-scene statistics require
        an entry point in ``aviris4_ch4.mf.MatchedFilter`` that does not exist
        yet; other scopes fall back to the window with a warning.
        """
        if background_stats is not None or settings.statistics_scope != "window":
            warnings.warn(
                "aviris4_mf computes background statistics from the filtered "
                "window; cached/full-scene statistics are not supported yet.",
                stacklevel=2,
            )
        mf = _aviris4_mf()
        data = self._filter_input(scene, settings)
        target_ds = _target_for_bands(
            target, np.asarray(data["wavelength"].values, dtype=float)
        )
        mask_shadows = self.mask_shadows and "shadow_mask" in data
        matched_filter = mf.MatchedFilter(
            unit_abs_spect_dataset=target_ds,
            min_n_samples=settings.min_n_samples,
            noise_model_file=self.noise_model_file,
            mask_shadows=mask_shadows,
            mask_plumes="plume_mask" in data,
            lognormal=self.lognormal,
            mean_function=self.mean_function,
            albedo_correction=self.albedo_correction,
            sensitivity_correction=self.sensitivity_correction,
            num_iter=self.num_iter,
            diffmf=self.diffmf,
            **self.options,
        )
        retrieved = matched_filter.apply_matched_filter(
            data=data,
            enhancement=settings.target_enhancement_ppm,
            parallel=self.parallel,
            save=False,
        )
        attrs = {
            key: scene.attrs[key]
            for key in ("sensor", "platform", "scene_id", "background_group")
            if key in scene.attrs
        }
        gas = str(target_ds["gas"].values).lower() if "gas" in target_ds else "ch4"
        product = mf_product_to_column_product(
            retrieved, attrs=attrs, geometry=scene, gas=gas
        )
        product.attrs.update(
            {
                "retriever": self.name,
                "retriever_version": getattr(mf, "__version__", "unknown"),
                "retrieval_statistics_scope": "window",
                "retrieval_target_enhancement_ppm": float(
                    settings.target_enhancement_ppm
                ),
                "retrieval_band_windows_nm": str(settings.band_windows_nm),
                "retrieval_min_n_samples": int(settings.min_n_samples),
            }
        )
        return product

    def background_statistics(
        self, scene: xr.Dataset, *, settings: RetrievalSettings
    ) -> xr.Dataset:
        """Cached statistics are not available for this retriever yet."""
        raise NotImplementedError(
            "aviris4_mf cannot export cached background statistics until "
            "aviris4_ch4.mf.MatchedFilter accepts precomputed per-column statistics."
        )


__all__ = ["Aviris4MatchedFilter", "select_bands"]
