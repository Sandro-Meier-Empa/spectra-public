"""What SPECTRA adds to a native radiance file, and the re-embedding guard.

An embedded file is the native radiance file with the plume pixels patched
and, in its root group:

- a ``source`` dimension, one entry per embedded plume;
- ``injected_<gas>_mass(source, <native y>, <native x>)`` in kg m-2 on the
  native spatial dimensions, zero outside each plume;
- one-dimensional ``source_*`` variables with per-plume provenance;
- ``spectra_*`` global attributes with run-level provenance, including the
  fully resolved run file in ``spectra_run_config``.

A file carrying any of these is refused as input, so plumes are never
embedded twice.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import netCDF4

from spectra.contracts.errors import ContractError

SOURCE_DIM = "source"
ATTR_PREFIX = "spectra_"
RUN_CONFIG_ATTR = "spectra_run_config"

# Per-plume variables on (source,): name -> (dtype, units, long name). The
# anchor-pixel variables are named after the native spatial dimensions.
SOURCE_VARIABLES: dict[str, tuple[str | type, str | None, str]] = {
    "source_plume_file": (str, None, "plume-column file"),
    "source_case_id": (str, None, "LES case identifier"),
    "source_anchor_mode": (str, None, "anchor given as lonlat, pixel or random"),
    "source_time_index": ("i4", None, "LES time index of the snapshot"),
    "source_emission_rate": ("f8", "kg s-1", "emission rate after scaling"),
    "source_emission_scale": ("f8", "1", "emission scale factor"),
    "source_lon": ("f8", "degrees_east", "longitude of the emission source"),
    "source_lat": ("f8", "degrees_north", "latitude of the emission source"),
    "source_rotation": (
        "f8",
        "degree",
        "plume rotation, counterclockwise positive",
    ),
    "source_flip": ("i1", None, "reflected LES axis"),
    "source_injected_mass": ("f8", "kg", "plume mass on the sensor grid"),
    "source_outside_fraction": ("f8", "1", "plume mass fraction outside the swath"),
    "source_cluster": ("i4", None, "index of the plume cluster"),
    "source_lut_sza": ("f8", "degree", "solar zenith angle of the cluster table"),
    "source_lut_vza": ("f8", "degree", "viewing zenith angle of the cluster table"),
    "source_lut_amf": ("f8", "1", "geometric air-mass factor of the cluster table"),
    "source_lut_surface_elevation": (
        "f8",
        "m",
        "surface elevation of the cluster forward model",
    ),
}
FLIP_CODES = {"none": 0, "x": 1, "y": 2}


def injected_name(gas: str) -> str:
    """Return the name of the injected-column variable of ``gas``."""
    return f"injected_{gas.lower()}_mass"


def anchor_variable(native_dim: str) -> str:
    """Return the anchor-pixel variable named after a native dimension."""
    return f"source_{native_dim}"


def _groups(dataset: netCDF4.Dataset) -> Iterator[netCDF4.Dataset]:
    yield dataset
    for group in dataset.groups.values():
        yield from _groups(group)


def check_not_embedded(path: str | Path) -> None:
    """Refuse a file that SPECTRA already embedded into, or that would collide.

    Every group is inspected: a ``spectra_*`` attribute or a ``source``
    dimension means the file is already embedded; a variable named
    ``injected_*_mass`` or ``source_*`` would collide with the output schema.

    Raises:
        ContractError: Naming what was found.
    """
    with netCDF4.Dataset(path) as dataset:
        for group in _groups(dataset):
            where = group.path
            attrs = [name for name in group.ncattrs() if name.startswith(ATTR_PREFIX)]
            if attrs or SOURCE_DIM in group.dimensions:
                raise ContractError(
                    f"{path} is already embedded by SPECTRA (group {where} carries "
                    f"{attrs or [SOURCE_DIM]}); embed into the original file instead."
                )
            clashes = [
                name
                for name in group.variables
                if name.startswith("source_")
                or (name.startswith("injected_") and name.endswith("_mass"))
            ]
            if clashes:
                raise ContractError(
                    f"{path} has variables {clashes} in group {where} that would "
                    "collide with the SPECTRA output schema."
                )


__all__ = [
    "ATTR_PREFIX",
    "FLIP_CODES",
    "RUN_CONFIG_ATTR",
    "SOURCE_DIM",
    "SOURCE_VARIABLES",
    "anchor_variable",
    "check_not_embedded",
    "injected_name",
]
