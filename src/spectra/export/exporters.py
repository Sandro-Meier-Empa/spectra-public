"""First-party exporters turning embedded samples into downstream datasets.

An exporter is the last plugin of the chain: it receives the embedded sample
(a retrieved ``ColumnProduct`` or, for ``needs_retrieval = False``, the
embedded ``SensorScene``), shapes it for its consumer and writes it. prism is
one consumer among others; a different machine-learning framework registers
its own exporter under the ``spectra.exporters`` entry-point group.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import xarray as xr

import spectra
from spectra.contracts.column_product import validate_column_product
from spectra.contracts.errors import ContractError
from spectra.contracts.sensor_scene import validate_sensor_scene
from spectra.export.io import write_dataset
from spectra.export.prism import to_prism_product


def _with_attrs(
    dataset: xr.Dataset, extra_attrs: Mapping[str, Any] | None
) -> xr.Dataset:
    """Return a copy with merged attributes; NetCDF/Zarr cannot hold ``None``."""
    output = dataset.copy()
    output.attrs.update(dict(extra_attrs or {}))
    output.attrs["spectra_version"] = spectra.__version__
    output.attrs = {
        key: value for key, value in output.attrs.items() if value is not None
    }
    return output


def _require_injected(dataset: xr.Dataset, gas: str) -> None:
    name = f"injected_{gas}_mass"
    if name not in dataset:
        raise ContractError(f"Sample must contain '{name}'; it is the label source.")


class PrismExporter:
    """prism sample contract: ``(y, x)`` product validated with prism's validator.

    Args:
        validate: Validate with prism's ``validate_sample`` when prism is
            installed.
    """

    name = "prism"
    needs_retrieval = True

    def __init__(self, *, validate: bool = True) -> None:
        """Store the validation switch."""
        self.validate = bool(validate)

    def prepare(
        self,
        sample: xr.Dataset,
        *,
        gas: str = "ch4",
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        """Return the sample as a prism product on ``(y, x)``."""
        return to_prism_product(
            sample, gas=gas, extra_attrs=extra_attrs, validate=self.validate
        )

    def write(
        self, dataset: xr.Dataset, path: str | Path, *, overwrite: bool = False
    ) -> Path:
        """Write the prism product."""
        return write_dataset(dataset, path, overwrite=overwrite)


class ColumnProductExporter:
    """Retrieved ``ColumnProduct`` kept in its geometry space (no prism rules).

    Args:
        require_injected: Require ``injected_{gas}_mass`` (the label source).
    """

    name = "column_product"
    needs_retrieval = True

    def __init__(self, *, require_injected: bool = True) -> None:
        """Store the label requirement."""
        self.require_injected = bool(require_injected)

    def prepare(
        self,
        sample: xr.Dataset,
        *,
        gas: str = "ch4",
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        """Validate the column product and merge the provenance attributes."""
        validate_column_product(sample, gas=gas)
        if self.require_injected:
            _require_injected(sample, gas)
        return _with_attrs(sample, extra_attrs)

    def write(
        self, dataset: xr.Dataset, path: str | Path, *, overwrite: bool = False
    ) -> Path:
        """Write the column product."""
        return write_dataset(dataset, path, overwrite=overwrite)


class RadianceExporter:
    """Embedded ``SensorScene`` (window or full scene) without retrieval.

    The sample keeps every band and pixel it was embedded with; use it to feed
    an external retrieval (e.g. emit-ghg) or a radiance-based model.
    """

    name = "radiance"
    needs_retrieval = False

    def prepare(
        self,
        sample: xr.Dataset,
        *,
        gas: str = "ch4",
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        """Validate the embedded scene and merge the provenance attributes."""
        validate_sensor_scene(sample)
        _require_injected(sample, gas)
        return _with_attrs(sample, extra_attrs)

    def write(
        self, dataset: xr.Dataset, path: str | Path, *, overwrite: bool = False
    ) -> Path:
        """Stream the embedded radiance to disk (scenes may be large)."""
        return write_dataset(dataset, path, overwrite=overwrite, load=False)


__all__ = ["ColumnProductExporter", "PrismExporter", "RadianceExporter"]
