"""Sensor adapters: native files in, ``SensorScene`` out.

``SENSORS`` maps the names accepted in run files to the built-in adapters;
:func:`get_sensor` also accepts any object satisfying
:class:`spectra.protocols.SensorAdapter`.
"""

from __future__ import annotations

from spectra.contracts.errors import ContractError
from spectra.protocols import SensorAdapter
from spectra.sensors.aviris3 import Aviris3Adapter
from spectra.sensors.aviris4 import Aviris4Adapter
from spectra.sensors.emit import EmitAdapter
from spectra.sensors.native import resolve_files

SENSORS: dict[str, type[SensorAdapter]] = {
    "aviris3": Aviris3Adapter,
    "aviris4": Aviris4Adapter,
    "emit": EmitAdapter,
}


def get_sensor(sensor: str | SensorAdapter) -> SensorAdapter:
    """Return the adapter named ``sensor`` or the given adapter object.

    Raises:
        ContractError: If the name is unknown or the object is no adapter.
    """
    if isinstance(sensor, str):
        if sensor not in SENSORS:
            raise ContractError(
                f"Unknown sensor {sensor!r}; choose from {sorted(SENSORS)}."
            )
        return SENSORS[sensor]()
    if not isinstance(sensor, SensorAdapter):
        raise ContractError(
            f"{type(sensor).__name__} does not implement the SensorAdapter protocol."
        )
    return sensor


__all__ = [
    "SENSORS",
    "Aviris3Adapter",
    "Aviris4Adapter",
    "EmitAdapter",
    "get_sensor",
    "resolve_files",
]
