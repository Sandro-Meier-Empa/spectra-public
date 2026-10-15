"""Entry-point registry of sensor adapters, retrievers, providers and regridders."""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Literal, cast

from spectra.contracts.errors import MissingExtraError, PluginError
from spectra.protocols import (
    Regridder,
    Retriever,
    SensorAdapter,
    UnitAbsorptionProvider,
)

PluginGroup = Literal["sensors", "retrievers", "unit_absorption", "regridders"]

GROUPS: dict[PluginGroup, str] = {
    "sensors": "spectra.sensors",
    "retrievers": "spectra.retrievers",
    "unit_absorption": "spectra.unit_absorption",
    "regridders": "spectra.regridders",
}
_PROTOCOLS: dict[PluginGroup, type] = {
    "sensors": SensorAdapter,
    "retrievers": Retriever,
    "unit_absorption": UnitAbsorptionProvider,
    "regridders": Regridder,
}
# Import names of optional dependencies mapped to the pip extra providing them.
_EXTRAS: dict[str, str] = {
    "aviris4_ch4": "aviris4",
    "ddeq": "emit",
    "rasterio": "emit",
    "earthaccess": "emit",
    "microhh_processing_chain": "microhh",
    "prism": "prism",
    "cscs": "slurm",
}


def list_plugins(group: PluginGroup) -> dict[str, EntryPoint]:
    """Return the registered entry points of one plugin group by name."""
    if group not in GROUPS:
        raise PluginError(
            f"Unknown plugin group {group!r}; choose from {list(GROUPS)}."
        )
    return {point.name: point for point in entry_points(group=GROUPS[group])}


def load_plugin(group: PluginGroup, name: str, **options: object) -> object:
    """Load, instantiate and protocol-check one registered plugin.

    Args:
        group: Plugin group.
        name: Entry-point name.
        **options: Keyword arguments passed to the plugin class or factory.

    Returns:
        The plugin instance.

    Raises:
        PluginError: If the plugin is unknown, fails to load, or does not
            implement the group's protocol.
        MissingExtraError: If an optional dependency is missing.
    """
    points = list_plugins(group)
    if name not in points:
        raise PluginError(
            f"No {group} plugin named {name!r}; registered: {sorted(points)}."
        )
    try:
        factory = points[name].load()
    except ImportError as error:
        package = (error.name or "").split(".")[0]
        if package in _EXTRAS:
            raise MissingExtraError(_EXTRAS[package], package, error) from error
        raise PluginError(f"Loading {group} plugin {name!r} failed: {error}") from error
    try:
        plugin = factory(**options)
    except MissingExtraError:
        raise
    except ImportError as error:
        package = (error.name or "").split(".")[0]
        if package in _EXTRAS:
            raise MissingExtraError(_EXTRAS[package], package, error) from error
        raise PluginError(f"Instantiating {group} plugin {name!r} failed.") from error
    protocol = _PROTOCOLS[group]
    if not isinstance(plugin, protocol):
        raise PluginError(
            f"{group} plugin {name!r} ({points[name].value}) does not implement "
            f"{protocol.__name__}."
        )
    return plugin


def get_sensor(name: str, **options: object) -> SensorAdapter:
    """Return the sensor adapter registered under ``name``."""
    return cast(SensorAdapter, load_plugin("sensors", name, **options))


def get_retriever(name: str, **options: object) -> Retriever:
    """Return the retriever registered under ``name``."""
    return cast(Retriever, load_plugin("retrievers", name, **options))


def get_unit_absorption_provider(
    name: str, **options: object
) -> UnitAbsorptionProvider:
    """Return the unit-absorption provider registered under ``name``."""
    return cast(UnitAbsorptionProvider, load_plugin("unit_absorption", name, **options))


def get_regridder(name: str, **options: object) -> Regridder:
    """Return the regridder registered under ``name``."""
    return cast(Regridder, load_plugin("regridders", name, **options))


__all__ = [
    "GROUPS",
    "PluginGroup",
    "get_regridder",
    "get_retriever",
    "get_sensor",
    "get_unit_absorption_provider",
    "list_plugins",
    "load_plugin",
]
