"""AVIRIS-3 sensor adapter slot (ORNL DAAC AV3_L1B_RDN NetCDF).

The adapter is registered so ``sensor: aviris3`` resolves, but it is not
implemented until a sample granule has been inspected: the variable paths,
dimension order, fill value, scaling, geometry files (OBS/ORT) and whether
the radiance is in sensor geometry are unknown. ``docs/sensors.md`` lists what
has to be checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import xarray as xr

from spectra.types import NativeLayout, Platform

ROLES: dict[str, bool] = {"radiance": True, "obs": True}
# Provisional; to be confirmed against a sample granule.
LAYOUT = NativeLayout(
    "radiance", "radiance", {"lines": "along", "samples": "across", "bands": "band"}
)


class Aviris3Adapter:
    """Placeholder adapter for AVIRIS-3 granules."""

    name = "aviris3"
    platform: Platform = "airborne"
    ROLES: Mapping[str, bool] = ROLES
    LAYOUT: NativeLayout = LAYOUT

    def open_scene(
        self,
        files: Mapping[str, str | Path],
        *,
        chunks: Mapping[str, int] | None = None,
    ) -> xr.Dataset:
        """Raise until the adapter is implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The AVIRIS-3 adapter needs a sample AV3_L1B_RDN granule to be "
            "implemented; see docs/sensors.md."
        )


__all__ = ["LAYOUT", "ROLES", "Aviris3Adapter"]
