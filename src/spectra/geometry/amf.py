"""Air-mass factors and effective layer heights.

Ported from ``aviris4_ch4.misc`` and ``aviris4_ch4.mf`` so the spectral code
no longer depends on the sensor package.
"""

from __future__ import annotations

import numpy as np

from spectra.types import RadianceLeg


def amf(angle_deg: float) -> float:
    """Return the air-mass factor of a zenith angle in degrees (secant)."""
    return float(1.0 / np.cos(np.deg2rad(angle_deg)))


def geometric_amf(sza_deg: float, vza_deg: float) -> float:
    """Return the sum of the solar and viewing air-mass factors."""
    return amf(sza_deg) + amf(vza_deg)


def path_amf(sza_deg: float, vza_deg: float, radiance_leg: RadianceLeg) -> float:
    """Return the geometric air-mass factor of the selected photon path.

    Args:
        sza_deg: Solar zenith angle in degrees.
        vza_deg: Viewing zenith angle in degrees.
        radiance_leg: Photon-path segment containing the plume.

    Returns:
        Air-mass factor of the solar leg, the view leg, or both.

    Raises:
        ValueError: If ``radiance_leg`` is invalid.
    """
    if radiance_leg == "both":
        return geometric_amf(sza_deg, vza_deg)
    if radiance_leg == "solar":
        return amf(sza_deg)
    if radiance_leg == "view":
        return amf(vza_deg)
    raise ValueError("Invalid radiance_leg. Choose 'solar', 'view' or 'both'.")


def effective_layer_height(elevation: float, layer_height: float) -> float:
    """Return the portion of the lowest model layer above the surface."""
    height = (layer_height - elevation) % layer_height
    return layer_height if np.isclose(height, 0.0) else float(height)


__all__ = ["amf", "effective_layer_height", "geometric_amf", "path_amf"]
