"""Synthetic data, synthetic native files and adapter conformance checks."""

from spectra.testing.conformance import run_adapter_conformance
from spectra.testing.synthetic import (
    make_plume_column,
    make_sensor_scene,
    make_unit_absorption_spectrum,
    scene_pixel_centres,
    scene_seed,
    scene_spatial_coords,
)

__all__ = [
    "make_plume_column",
    "make_sensor_scene",
    "make_unit_absorption_spectrum",
    "run_adapter_conformance",
    "scene_pixel_centres",
    "scene_seed",
    "scene_spatial_coords",
]
