"""Synthetic sensors, providers, retrievers and conformance tests."""

from spectra.testing.conformance import (
    conformance_parametrize,
    run_retriever_conformance,
    run_sensor_conformance,
)
from spectra.testing.synthetic import (
    make_column_product,
    make_plume_column,
    make_sensor_scene,
    make_unit_absorption_spectrum,
    scene_pixel_centres,
    scene_seed,
    scene_spatial_coords,
)

__all__ = [
    "conformance_parametrize",
    "make_column_product",
    "make_plume_column",
    "make_sensor_scene",
    "make_unit_absorption_spectrum",
    "run_retriever_conformance",
    "run_sensor_conformance",
    "scene_pixel_centres",
    "scene_seed",
    "scene_spatial_coords",
]
