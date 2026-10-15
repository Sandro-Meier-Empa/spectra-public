"""Background quality assessment of sensor scenes and column products.

The metrics are deliberately simple and sensor independent: mask fractions
from the ``SensorScene`` masks, background statistics of a ``ColumnProduct``
and a plume-like detection count that flags scenes which are not plume-free
backgrounds. Large scenes are subsampled with strides; dask-backed masks are
reduced lazily.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy import ndimage

from spectra.contracts.column_product import MAP_SPACE_DIMS, SENSOR_SPACE_DIMS
from spectra.contracts.errors import ContractError, MissingExtraError
from spectra.contracts.sensor_scene import OPTIONAL_MASKS
from spectra.types import Window

DetectionMethod = Literal["snr", "ddeq", "none"]
MASK_NAMES: tuple[str, ...] = ("valid", *OPTIONAL_MASKS)


@dataclass(frozen=True)
class QaSettings:
    """Settings of :func:`compute_scene_qa`.

    Attributes:
        detection: Plume detection method: ``snr`` (local enhancement over a
            median-filtered background against the retrieval uncertainty),
            ``ddeq`` (experimental; the z-test building blocks of
            ``ddeq.dplume``) or ``none``.
        detection_snr: Signal-to-noise ratio above which a pixel is a
            candidate plume pixel.
        detection_min_pixels: Minimum connected candidate pixels per detection.
        local_filter_px: Standard deviation in pixels of the Gaussian kernel
            that averages the local enhancement (and its noise) before the
            threshold; ``0`` disables the smoothing.
        background_filter_px: Side length in pixels of the median filter that
            estimates the background column.
        noise: Per-pixel noise of the SNR: the product ``uncertainty``
            (default) or ``mad``, the robust scatter (1.4826 MAD) of the local
            enhancement over valid pixels, for products whose uncertainty is
            not a per-pixel one-sigma column noise.
        use_product: Compute background and detection metrics from the column
            product when one is given.
        subsample: Stride applied to both spatial dimensions before any
            computation, for large scenes. Filter sizes refer to the
            subsampled grid.
        gas: Lower-case gas name of the product variables.
    """

    detection: DetectionMethod = "snr"
    detection_snr: float = 3.0
    detection_min_pixels: int = 9
    local_filter_px: float = 1.0
    background_filter_px: int = 51
    noise: Literal["uncertainty", "mad"] = "uncertainty"
    use_product: bool = True
    subsample: int | None = None
    gas: str = "ch4"

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.detection not in ("snr", "ddeq", "none"):
            raise ContractError("detection must be 'snr', 'ddeq' or 'none'.")
        if self.detection_snr <= 0:
            raise ContractError("detection_snr must be positive.")
        if self.detection_min_pixels < 1:
            raise ContractError("detection_min_pixels must be at least 1.")
        if self.local_filter_px < 0:
            raise ContractError("local_filter_px must be non-negative.")
        if self.background_filter_px < 1:
            raise ContractError("background_filter_px must be at least 1.")
        if self.noise not in ("uncertainty", "mad"):
            raise ContractError("noise must be 'uncertainty' or 'mad'.")
        if self.subsample is not None and self.subsample < 1:
            raise ContractError("subsample must be a positive stride or None.")


@dataclass(frozen=True)
class QaResult:
    """Metrics of one scene or window; NaN/``None`` where unavailable.

    Attributes:
        valid_fraction: Fraction of valid pixels.
        cloud_fraction: Fraction of cloudy pixels.
        shadow_fraction: Fraction of shadowed pixels.
        water_fraction: Fraction of water pixels.
        saturation_fraction: Fraction of saturated pixels.
        bg_median_kgm2: Median background column over valid pixels.
        bg_std_kgm2: Standard deviation of the background column.
        bg_mad_kgm2: Median absolute deviation of the background column.
        uncertainty_median_kgm2: Median retrieval uncertainty.
        n_detections: Number of plume-like detections, ``None`` when no
            detection was attempted.
        max_detection_snr: Largest pixel SNR inside the detections.
        detection_method: Method that produced the detections.
        subsample: Stride the metrics were computed with.
        detections: Boolean detection mask on the (subsampled) product grid;
            not stored in the catalogue table.
    """

    valid_fraction: float = math.nan
    cloud_fraction: float = math.nan
    shadow_fraction: float = math.nan
    water_fraction: float = math.nan
    saturation_fraction: float = math.nan
    bg_median_kgm2: float = math.nan
    bg_std_kgm2: float = math.nan
    bg_mad_kgm2: float = math.nan
    uncertainty_median_kgm2: float = math.nan
    n_detections: int | None = None
    max_detection_snr: float = math.nan
    detection_method: str = "none"
    subsample: int | None = None
    detections: xr.DataArray | None = field(default=None, repr=False, compare=False)

    def metrics(self) -> dict[str, float | int | None]:
        """Return the metric values keyed by catalogue column name."""
        from spectra.catalog.schema import QA_METRIC_COLUMNS

        return {name: getattr(self, name) for name in QA_METRIC_COLUMNS}


def _strides(dims: tuple[str, ...], subsample: int | None) -> dict[str, slice]:
    if subsample is None or subsample == 1:
        return {}
    return {dim: slice(None, None, int(subsample)) for dim in dims}


def _spatial_dims(product: xr.Dataset, gas: str) -> tuple[str, str]:
    space = product.attrs.get("geometry_space")
    if space == "sensor":
        return SENSOR_SPACE_DIMS
    if space == "map":
        return MAP_SPACE_DIMS
    dims = tuple(str(dim) for dim in product[f"{gas}_mass"].dims)
    if len(dims) != 2:
        raise ContractError(f"{gas}_mass must be two-dimensional; got {dims}.")
    return (dims[0], dims[1])


def _mask_fraction(scene: xr.Dataset, name: str, subsample: int | None) -> float:
    """Return the fraction of set pixels of a scene mask (dask-aware)."""
    if name not in scene:
        return math.nan
    mask = scene[name]
    mask = mask.isel(_strides(tuple(str(d) for d in mask.dims), subsample))
    if mask.size == 0:
        return math.nan
    return float(mask.astype("float64").mean().values)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, math.ceil(3.0 * sigma))
    axis = np.arange(-radius, radius + 1, dtype=float)
    one_d = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = np.outer(one_d, one_d)
    return kernel / kernel.sum()


def _label_components(candidate: np.ndarray, min_pixels: int) -> tuple[np.ndarray, int]:
    """Keep 8-connected candidate components with at least ``min_pixels``."""
    structure = np.ones((3, 3), dtype=bool)
    labels = np.zeros(candidate.shape, dtype=np.int32)
    ndimage.label(candidate, structure=structure, output=labels)
    n_labels = int(labels.max())
    if n_labels == 0:
        return np.zeros(candidate.shape, dtype=bool), 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_pixels)
    keep[0] = False
    return keep[labels], int(keep.sum())


def _robust_std(sample: np.ndarray) -> float:
    """Return 1.4826 times the median absolute deviation."""
    sample = np.asarray(sample, dtype=float)
    if sample.size == 0:
        return math.nan
    return 1.4826 * float(np.median(np.abs(sample - np.median(sample))))


def local_snr(
    mass: np.ndarray, uncertainty: np.ndarray, valid: np.ndarray, settings: QaSettings
) -> np.ndarray:
    """Return the signal-to-noise ratio of the local column enhancement.

    The background is a median filter of the column (invalid pixels filled
    with the valid median), the enhancement is averaged with a Gaussian
    kernel and divided by the correspondingly averaged random noise. Invalid
    pixels are NaN.

    Args:
        mass: Column in kg m-2.
        uncertainty: Per-pixel random uncertainty in kg m-2.
        valid: Pixels contributing to the statistics.
        settings: Filter sizes.

    Returns:
        SNR array with the shape of ``mass``.
    """
    mass = np.asarray(mass, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(mass)
    if not valid.any():
        return np.full(mass.shape, np.nan)
    fill = float(np.median(mass[valid]))
    filled = np.where(valid, mass, fill)
    size = int(settings.background_filter_px)
    if size > 1:
        background = ndimage.median_filter(filled, size=size, mode="nearest")
    else:
        background = np.full(mass.shape, fill)
    enhancement = np.where(valid, filled - background, 0.0)
    pixel_noise = (
        np.full(mass.shape, _robust_std(enhancement[valid]))
        if settings.noise == "mad"
        else uncertainty
    )
    variance = np.where(valid & np.isfinite(pixel_noise), pixel_noise**2, 0.0)
    if settings.local_filter_px > 0:
        kernel = _gaussian_kernel(settings.local_filter_px)
        # Normalised convolution: the weight sum cancels in the ratio, and the
        # noise of a weighted mean of independent pixels is sqrt(sum w^2 u^2).
        signal = ndimage.convolve(enhancement, kernel, mode="constant", cval=0.0)
        noise = np.sqrt(
            ndimage.convolve(variance, kernel**2, mode="constant", cval=0.0)
        )
    else:
        signal = enhancement
        noise = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise > 0, signal / noise, np.nan)
    snr[~valid] = np.nan
    return snr


def _detect_snr(
    mass: np.ndarray, uncertainty: np.ndarray, valid: np.ndarray, settings: QaSettings
) -> tuple[np.ndarray, int, float]:
    snr = local_snr(mass, uncertainty, valid, settings)
    candidate = np.isfinite(snr) & (snr >= settings.detection_snr)
    detections, n_detections = _label_components(
        candidate, settings.detection_min_pixels
    )
    max_snr = float(np.nanmax(snr[detections])) if n_detections else math.nan
    return detections, n_detections, max_snr


def _detect_ddeq(
    mass: np.ndarray, uncertainty: np.ndarray, valid: np.ndarray, settings: QaSettings
) -> tuple[np.ndarray, int, float]:
    """Experimental: z-test of ``ddeq.dplume`` without point sources.

    ``ddeq.dplume.detect_plumes`` assigns plumes to known point sources, so
    only its building blocks (local median background, Gaussian local mean
    and z-test, labelling) are used here.
    """
    try:
        import ddeq.dplume as dplume
    except ImportError as error:
        raise MissingExtraError("emit", "ddeq", error) from error
    from scipy.stats import norm

    mass = np.asarray(mass, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(mass)
    if not valid.any():
        return np.zeros(mass.shape, dtype=bool), 0, math.nan
    fill = float(np.median(mass[valid]))
    filled = np.where(valid, mass, fill)
    background = np.asarray(
        dplume.local_median(
            xr.DataArray(filled), size=int(settings.background_filter_px)
        )
    )
    if settings.noise == "mad":
        var_rand = _robust_std((mass - background)[valid]) ** 2
    else:
        var_rand = float(np.nanmean(np.asarray(uncertainty, dtype=float)[valid]) ** 2)
    mean_s, variance, n_s = dplume.local_mean(
        xr.DataArray(np.where(valid, mass, np.nan)),
        size=max(1, round(settings.local_filter_px)),
        kernel_type="gaussian",
        var_rand=var_rand,
    )
    q = float(norm.cdf(settings.detection_snr))
    z_values, is_hit = dplume.do_test(
        np.asarray(mean_s),
        background,
        None,
        None,
        np.asarray(n_s, dtype="f4"),
        q,
        variance=np.asarray(variance),
    )
    z_values = np.asarray(z_values, dtype=float)
    candidate = np.asarray(is_hit, dtype=bool) & valid & np.isfinite(z_values)
    labels = np.asarray(dplume.label_plumes(candidate, settings.detection_min_pixels))
    detections = labels > 0
    n_detections = int(labels.max()) if detections.any() else 0
    max_snr = float(np.nanmax(z_values[detections])) if n_detections else math.nan
    return detections, n_detections, max_snr


def _product_metrics(
    product: xr.Dataset,
    scene_valid: np.ndarray | None,
    settings: QaSettings,
) -> dict[str, Any]:
    """Background statistics and detections of a (subsampled) product."""
    gas = settings.gas
    mass_name, unc_name = f"{gas}_mass", f"{gas}_mass_uncertainty"
    for name in (mass_name, unc_name, "retrieval_valid"):
        if name not in product:
            raise ContractError(f"ColumnProduct must contain '{name}'.")
    dims = _spatial_dims(product, gas)
    strides = _strides(dims, settings.subsample)
    mass_array = product[mass_name].isel(strides).transpose(*dims)
    mass = np.asarray(mass_array.values, dtype=float)
    uncertainty = np.asarray(
        product[unc_name].isel(strides).transpose(*dims).values, dtype=float
    )
    valid = np.asarray(
        product["retrieval_valid"].isel(strides).transpose(*dims).values, dtype=bool
    )
    valid &= np.isfinite(mass)
    if scene_valid is not None and scene_valid.shape == valid.shape:
        valid &= scene_valid
    metrics: dict[str, Any] = {}
    if valid.any():
        sample = mass[valid]
        median = float(np.median(sample))
        metrics["bg_median_kgm2"] = median
        metrics["bg_std_kgm2"] = float(np.std(sample))
        metrics["bg_mad_kgm2"] = float(np.median(np.abs(sample - median)))
        metrics["uncertainty_median_kgm2"] = float(np.nanmedian(uncertainty[valid]))
    if settings.detection == "none":
        return metrics
    detect = _detect_snr if settings.detection == "snr" else _detect_ddeq
    detections, n_detections, max_snr = detect(mass, uncertainty, valid, settings)
    coords = {dim: mass_array[dim].variable for dim in dims if dim in mass_array.coords}
    metrics["n_detections"] = n_detections
    metrics["max_detection_snr"] = max_snr
    metrics["detection_method"] = settings.detection
    metrics["detections"] = xr.DataArray(
        detections,
        dims=dims,
        coords=coords,
        name="detections",
        attrs={
            "long_name": "plume-like detections",
            "method": settings.detection,
            "detection_snr": settings.detection_snr,
            "detection_min_pixels": settings.detection_min_pixels,
            "local_filter_px": settings.local_filter_px,
            "background_filter_px": settings.background_filter_px,
            "subsample": settings.subsample or 1,
            "scene_id": str(product.attrs.get("scene_id", "")),
            "sensor": str(product.attrs.get("sensor", "")),
        },
    )
    return metrics


def _scene_valid(
    scene: xr.Dataset, product: xr.Dataset | None, settings: QaSettings
) -> np.ndarray | None:
    """Return the subsampled scene ``valid`` mask when it shares the product grid."""
    if product is None or "valid" not in scene:
        return None
    dims = _spatial_dims(product, settings.gas)
    valid = scene["valid"]
    if tuple(str(d) for d in valid.dims) != dims:
        return None
    valid = valid.isel(_strides(dims, settings.subsample))
    return np.asarray(valid.values, dtype=bool)


def _compute(
    scene: xr.Dataset | None, product: xr.Dataset | None, settings: QaSettings
) -> QaResult:
    values: dict[str, Any] = {"subsample": settings.subsample}
    scene_valid: np.ndarray | None = None
    if scene is not None:
        # Read the (possibly radiance-derived, dask-backed) valid mask once.
        scene_valid = _scene_valid(scene, product, settings)
        for name in MASK_NAMES:
            if name == "valid" and scene_valid is not None:
                values["valid_fraction"] = (
                    float(scene_valid.mean()) if scene_valid.size else math.nan
                )
            else:
                values[f"{name}_fraction"] = _mask_fraction(
                    scene, name, settings.subsample
                )
    elif product is not None and "retrieval_valid" in product:
        values["valid_fraction"] = _mask_fraction(
            product, "retrieval_valid", settings.subsample
        )
    if product is not None and settings.use_product:
        values.update(_product_metrics(product, scene_valid, settings))
    return QaResult(**values)


def compute_scene_qa(
    scene: xr.Dataset,
    product: xr.Dataset | None = None,
    settings: QaSettings | None = None,
) -> QaResult:
    """Compute background QA metrics of a whole scene.

    Args:
        scene: ``SensorScene`` dataset (masks may be dask-backed).
        product: ``ColumnProduct`` of the scene on the same grid, if any.
        settings: QA settings; defaults to :class:`QaSettings`.

    Returns:
        Mask fractions, background statistics and detections.
    """
    return _compute(scene, product, settings or QaSettings())


def compute_window_qa(
    product: xr.Dataset,
    window: Window,
    settings: QaSettings | None = None,
    *,
    scene: xr.Dataset | None = None,
) -> QaResult:
    """Compute QA metrics restricted to an index window on the sensor grid.

    Args:
        product: Sensor-space ``ColumnProduct``.
        window: Across/along index window.
        settings: QA settings; ``subsample`` is normally ``None`` for windows.
        scene: Optional ``SensorScene`` supplying the mask fractions; without
            it ``valid_fraction`` comes from ``retrieval_valid``.

    Returns:
        Metrics of the window.
    """
    settings = settings or QaSettings()
    if _spatial_dims(product, settings.gas) != SENSOR_SPACE_DIMS:
        raise ContractError("compute_window_qa needs a sensor-space product.")
    indexers: Mapping[str, slice] = window.indexers
    sub_scene = None if scene is None else scene.isel(indexers)
    return _compute(sub_scene, product.isel(indexers), settings)


__all__ = [
    "DetectionMethod",
    "QaResult",
    "QaSettings",
    "compute_scene_qa",
    "compute_window_qa",
    "local_snr",
]
