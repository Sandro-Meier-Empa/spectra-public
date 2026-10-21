"""Run files: one scene, its files, the plumes to embed and the forward model.

A run file is YAML (or the same structure as a mapping)::

    sensor: emit
    files: {radiance: /path/to/RAD.nc, obs: /path/to/OBS.nc}
    output: /path/to/embedded_RAD.nc
    overwrite: false
    seed: 1
    forward_model:
      solar_reference: /path/to/hybrid_reference_spectrum.nc
      tau: {all: /path/to/tau_all.nc, CH4: /path/to/tau_CH4.nc}
      profiles: {z: /path/to/z.xml, t: /path/to/t.xml, CH4: /path/to/CH4.xml}
      backgrounds: {CH4: 1.93}
    plumes:
      - file: /path/to/plume.nc
        time_index: 120
        anchor: {lon: -102.9, lat: 32.1}
        rotation: 30

Relative paths resolve against the run file's directory. Every file is named
explicitly; nothing is discovered. :func:`dump_run` writes a resolved run
(seed and randomly drawn anchors filled in) back to YAML, which is stored in
the output file and reproduces it.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from spectra.contracts.errors import ContractError
from spectra.geometry.swath import DEFAULT_MAX_ATTEMPTS, AnchorMode
from spectra.protocols import ForwardModel, SensorAdapter
from spectra.spectral.libradtran import LibRadtranArtsConfig, LibRadtranArtsFiles
from spectra.spectral.toy import ToyForwardModel
from spectra.types import ANCHOR_POINTS, FLIPS, AnchorPoint, Flip

RUN_KEYS = (
    "sensor",
    "files",
    "output",
    "overwrite",
    "seed",
    "wavelength_ranges_nm",
    "forward_model",
    "plumes",
)
PLUME_KEYS = ("file", "time_index", "anchor", "rotation", "flip", "emission_scale")
ANCHOR_KEYS = (
    "lon",
    "lat",
    "along",
    "across",
    "random",
    "corner",
    "min_outside_fraction",
    "max_outside_fraction",
    "max_attempts",
)
LIBRADTRAN_KEYS = (
    "model",
    "solar_reference",
    "tau",
    "profiles",
    "backgrounds",
    "scenario",
    "rho",
    "wmin",
    "wmax",
    "enhancement_min_ppm",
    "enhancement_max_ppm",
    "n_enhancements",
    "layer_height_m",
)
TOY_KEYS = (
    "model",
    "lines_nm",
    "strength",
    "width_nm",
    "saturation_ppm_m",
    "spectral_range_nm",
    "enhancements_ppm",
    "layer_height_m",
)


@dataclass(frozen=True)
class AnchorSpec:
    """Where a plume's anchor point goes.

    Exactly one of a geographic position (``lon``/``lat``), a pixel
    (``along``/``across``) or ``random`` is given. A random anchor that was
    already drawn keeps ``random`` and carries the drawn pixel, which then
    acts as a pixel anchor.

    Attributes:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        along: Along-track pixel index.
        across: Across-track pixel index.
        random: Draw a uniformly random pixel.
        corner: LES point pinned to the anchor: the emission ``source`` or a
            grid corner cell.
        min_outside_fraction: Smallest accepted plume mass fraction outside the
            swath (random anchors).
        max_outside_fraction: Largest accepted fraction (random anchors).
        max_attempts: Maximum number of random draws.
    """

    lon: float | None = None
    lat: float | None = None
    along: int | None = None
    across: int | None = None
    random: bool = False
    corner: AnchorPoint = "source"
    min_outside_fraction: float = 0.0
    max_outside_fraction: float = 0.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        """Require exactly one way of giving the anchor."""
        geographic = self.lon is not None or self.lat is not None
        pixel = self.along is not None or self.across is not None
        if geographic and (self.lon is None or self.lat is None):
            raise ContractError("A geographic anchor needs both lon and lat.")
        if pixel and (self.along is None or self.across is None):
            raise ContractError("A pixel anchor needs both along and across.")
        if geographic and (pixel or self.random):
            raise ContractError("Give either lon/lat, along/across or random.")
        if not (geographic or pixel or self.random):
            raise ContractError("An anchor needs lon/lat, along/across or random.")
        if self.corner not in ANCHOR_POINTS:
            raise ContractError(f"corner must be one of {ANCHOR_POINTS}.")
        if self.max_attempts < 1:
            raise ContractError("max_attempts must be at least 1.")

    @property
    def mode(self) -> AnchorMode:
        """Return ``lonlat``, ``pixel`` or ``random``."""
        if self.lon is not None:
            return "lonlat"
        if self.along is not None:
            return "pixel"
        return "random"


@dataclass(frozen=True)
class PlumeSpec:
    """One plume to embed.

    Attributes:
        file: Plume-column file (``PlumeColumn`` contract).
        time_index: LES snapshot index.
        anchor: Where the plume goes.
        rotation: Whole-degree rotation about the anchor, counterclockwise.
        flip: LES axis reflected before the rotation.
        emission_scale: Factor applied to the column and emission rate.
    """

    file: Path
    time_index: int
    anchor: AnchorSpec
    rotation: int = 0
    flip: Flip = "none"
    emission_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the snapshot index, rotation, flip and scale."""
        object.__setattr__(self, "file", Path(self.file))
        if self.time_index < 0:
            raise ContractError("time_index must be non-negative.")
        if not -359 <= self.rotation <= 359:
            raise ContractError("rotation must lie between -359 and 359 degrees.")
        if self.flip not in FLIPS:
            raise ContractError(f"flip must be one of {FLIPS}.")
        if not np.isfinite(self.emission_scale) or self.emission_scale <= 0:
            raise ContractError("emission_scale must be positive.")


ForwardModelSpec = LibRadtranArtsConfig | ForwardModel


def wavelength_ranges(
    value: object, where: str = "wavelength_ranges_nm"
) -> tuple[tuple[float, float], ...]:
    """Return wavelength ranges in nm, sorted and validated.

    Args:
        value: One ``[lower, upper]`` pair or a sequence of such pairs.
        where: Name used in error messages.

    Returns:
        The ranges as ``(lower, upper)`` float pairs in increasing order.

    Raises:
        :exc:`ContractError`: If a range is not an increasing pair of finite
            numbers or two ranges overlap.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ContractError(f"{where} must be a [lower, upper] pair or a list of them.")
    pairs = list(value) if isinstance(value[0], (list, tuple)) else [value]
    ranges = []
    for index, pair in enumerate(pairs):
        label = f"{where}[{index}]"
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ContractError(f"{label} must be a [lower, upper] pair, got {pair!r}.")
        lower, upper = (_float(item, label) for item in pair)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ContractError(f"{label} must be finite with lower < upper.")
        ranges.append((lower, upper))
    ranges.sort()
    for first, second in itertools.pairwise(ranges):
        if second[0] <= first[1]:
            raise ContractError(f"{where}: the ranges {first} and {second} overlap.")
    return tuple(ranges)


@dataclass(frozen=True)
class RunConfig:
    """One embedding run.

    Attributes:
        sensor: Sensor name or adapter object.
        files: Scene files by role.
        output: Output radiance file.
        plumes: Plumes to embed.
        forward_model: libRadtran/ARTS settings or a forward-model object.
        overwrite: Replace an existing output.
        seed: Seed of the random anchors; ``None`` draws one at run time.
        wavelength_ranges_nm: Disjoint ranges whose bands are attenuated;
            ``None`` attenuates every band in the forward model's range.
    """

    sensor: str | SensorAdapter
    files: dict[str, Path]
    output: Path
    plumes: tuple[PlumeSpec, ...]
    forward_model: ForwardModelSpec
    overwrite: bool = False
    seed: int | None = None
    wavelength_ranges_nm: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Validate the plume list, the output path and the wavelength ranges."""
        if self.wavelength_ranges_nm is not None:
            object.__setattr__(
                self,
                "wavelength_ranges_nm",
                wavelength_ranges(self.wavelength_ranges_nm),
            )
        if not self.plumes:
            raise ContractError("A run needs at least one plume.")
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(
            self, "files", {role: Path(path) for role, path in self.files.items()}
        )
        for role, path in self.files.items():
            if Path(path) == self.output:
                raise ContractError(
                    f"output is the input {role!r} file; SPECTRA never modifies inputs."
                )
        if self.seed is not None and not 0 <= int(self.seed) < 2**63:
            raise ContractError("seed must be a non-negative 63-bit integer.")

    def replace(self, **changes: object) -> RunConfig:
        """Return a copy with ``changes`` applied."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]


# -- loading --------------------------------------------------------------------


def _require_mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be a mapping.")
    return cast(Mapping[str, Any], value)


def _check_keys(
    mapping: Mapping[str, Any], allowed: tuple[str, ...], where: str
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ContractError(
            f"Unknown keys {unknown} in {where}; allowed: {list(allowed)}."
        )


def _path(value: object, where: str, base: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ContractError(f"{where} must be a path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be a number, got {value!r}.")
    return float(value)


def _int(value: object, where: str) -> int:
    number = _float(value, where)
    if number != int(number):
        raise ContractError(f"{where} must be an integer, got {value!r}.")
    return int(number)


def _anchor(value: object, where: str) -> AnchorSpec:
    if value == "random":
        return AnchorSpec(random=True)
    mapping = _require_mapping(value, where)
    _check_keys(mapping, ANCHOR_KEYS, where)
    kwargs: dict[str, Any] = {}
    for key in ("lon", "lat", "min_outside_fraction", "max_outside_fraction"):
        if mapping.get(key) is not None:
            kwargs[key] = _float(mapping[key], f"{where}.{key}")
    for key in ("along", "across", "max_attempts"):
        if mapping.get(key) is not None:
            kwargs[key] = _int(mapping[key], f"{where}.{key}")
    if "random" in mapping:
        if not isinstance(mapping["random"], bool):
            raise ContractError(f"{where}.random must be true or false.")
        kwargs["random"] = mapping["random"]
    if "corner" in mapping:
        kwargs["corner"] = str(mapping["corner"])
    try:
        return AnchorSpec(**kwargs)
    except ContractError as error:
        raise ContractError(f"{where}: {error}") from error


def _plume(value: object, where: str, base: Path) -> PlumeSpec:
    mapping = _require_mapping(value, where)
    _check_keys(mapping, PLUME_KEYS, where)
    for key in ("file", "time_index", "anchor"):
        if key not in mapping:
            raise ContractError(f"{where} needs '{key}'.")
    try:
        return PlumeSpec(
            file=_path(mapping["file"], f"{where}.file", base),
            time_index=_int(mapping["time_index"], f"{where}.time_index"),
            anchor=_anchor(mapping["anchor"], f"{where}.anchor"),
            rotation=_int(mapping.get("rotation", 0), f"{where}.rotation"),
            flip=cast(Flip, str(mapping.get("flip", "none"))),
            emission_scale=_float(
                mapping.get("emission_scale", 1.0), f"{where}.emission_scale"
            ),
        )
    except ContractError as error:
        if str(error).startswith(where):
            raise
        raise ContractError(f"{where}: {error}") from error


def _paths(value: object, where: str, base: Path) -> dict[str, Path]:
    mapping = _require_mapping(value, where)
    return {
        str(key): _path(path, f"{where}.{key}", base) for key, path in mapping.items()
    }


def _forward_model(value: object, base: Path) -> ForwardModelSpec:
    where = "forward_model"
    mapping = _require_mapping(value, where)
    model = mapping.get("model", "libradtran_arts")
    if model == "toy":
        _check_keys(mapping, TOY_KEYS, where)
        kwargs = {key: item for key, item in mapping.items() if key != "model"}
        return ToyForwardModel(**kwargs)
    if model != "libradtran_arts":
        raise ContractError(f"{where}.model must be 'libradtran_arts' or 'toy'.")
    _check_keys(mapping, LIBRADTRAN_KEYS, where)
    for key in ("solar_reference", "tau", "profiles", "backgrounds"):
        if key not in mapping:
            raise ContractError(f"{where} needs '{key}'.")
    files = LibRadtranArtsFiles(
        solar_reference=_path(
            mapping["solar_reference"], f"{where}.solar_reference", base
        ),
        tau=_paths(mapping["tau"], f"{where}.tau", base),
        profiles=_paths(mapping["profiles"], f"{where}.profiles", base),
    )
    backgrounds = {
        str(gas): _float(item, f"{where}.backgrounds.{gas}")
        for gas, item in _require_mapping(
            mapping["backgrounds"], f"{where}.backgrounds"
        ).items()
    }
    options: dict[str, Any] = {}
    for key in (
        "rho",
        "wmin",
        "wmax",
        "enhancement_min_ppm",
        "enhancement_max_ppm",
        "layer_height_m",
    ):
        if mapping.get(key) is not None:
            options[key] = _float(mapping[key], f"{where}.{key}")
    if mapping.get("n_enhancements") is not None:
        options["n_enhancements"] = _int(
            mapping["n_enhancements"], f"{where}.n_enhancements"
        )
    if mapping.get("scenario") is not None:
        options["scenario"] = str(mapping["scenario"])
    return LibRadtranArtsConfig(files=files, backgrounds=backgrounds, **options)


def load_run(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    base: str | os.PathLike[str] | None = None,
) -> RunConfig:
    """Read a run file (YAML path) or mapping into a :class:`RunConfig`.

    Args:
        source: Path of a YAML run file, or the run as a mapping.
        base: Directory relative paths resolve against; defaults to the run
            file's directory, or the working directory for a mapping.

    Returns:
        The validated run.

    Raises:
        :exc:`ContractError`: Naming the first invalid or unknown entry.
    """
    if isinstance(source, Mapping):
        mapping: Mapping[str, Any] = source
        root = Path(base) if base is not None else Path.cwd()
    else:
        path = Path(source)
        if not path.is_file():
            raise ContractError(f"Run file {path} does not exist.")
        loaded = yaml.safe_load(path.read_text())
        mapping = _require_mapping(loaded, str(path))
        root = Path(base) if base is not None else path.parent
    root = root.expanduser().resolve()
    _check_keys(mapping, RUN_KEYS, "the run file")
    for key in ("sensor", "files", "output", "forward_model", "plumes"):
        if key not in mapping:
            raise ContractError(f"The run file needs '{key}'.")
    plumes = mapping["plumes"]
    if not isinstance(plumes, list) or not plumes:
        raise ContractError("plumes must be a non-empty list.")
    sensor = mapping["sensor"]
    if not isinstance(sensor, str):
        raise ContractError("sensor must be a sensor name.")
    overwrite = mapping.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ContractError("overwrite must be true or false.")
    seed = mapping.get("seed")
    return RunConfig(
        sensor=sensor,
        files=_paths(mapping["files"], "files", root),
        output=_path(mapping["output"], "output", root),
        plumes=tuple(
            _plume(item, f"plumes[{index}]", root) for index, item in enumerate(plumes)
        ),
        forward_model=_forward_model(mapping["forward_model"], root),
        overwrite=overwrite,
        seed=None if seed is None else _int(seed, "seed"),
        wavelength_ranges_nm=mapping.get("wavelength_ranges_nm"),
    )


# -- writing --------------------------------------------------------------------


def _plain(value: object) -> Any:  # noqa: ANN401
    """Return YAML-friendly builtins for numpy scalars, paths and sequences."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _anchor_mapping(anchor: AnchorSpec) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for item in dataclasses.fields(anchor):
        value = getattr(anchor, item.name)
        if value is None or (item.name != "corner" and value == item.default):
            continue
        mapping[item.name] = value
    return _plain(mapping)


def _forward_model_mapping(model: ForwardModelSpec) -> dict[str, Any]:
    if isinstance(model, LibRadtranArtsConfig):
        mapping: dict[str, Any] = {
            "model": "libradtran_arts",
            "solar_reference": model.files.solar_reference,
            "tau": dict(model.files.tau),
            "profiles": dict(model.files.profiles),
            "backgrounds": dict(model.backgrounds),
        }
        for item in dataclasses.fields(model):
            if item.name in ("files", "backgrounds"):
                continue
            value = getattr(model, item.name)
            if value is not None:
                mapping[item.name] = value
        return _plain(mapping)
    if isinstance(model, ToyForwardModel):
        return _plain(
            {
                "model": "toy",
                "lines_nm": model.lines_nm,
                "strength": model.strength,
                "width_nm": model.width_nm,
                "saturation_ppm_m": model.saturation_ppm_m,
                "spectral_range_nm": model.spectral_range_nm,
                "enhancements_ppm": model.enhancements_ppm,
                "layer_height_m": model.layer_height_m,
            }
        )
    return _plain({"model": model.name, **model.provenance()})


def run_to_mapping(config: RunConfig) -> dict[str, Any]:
    """Return a run as plain YAML-friendly data."""
    sensor = config.sensor if isinstance(config.sensor, str) else config.sensor.name
    mapping: dict[str, Any] = {
        "sensor": sensor,
        "files": _plain(config.files),
        "output": str(config.output),
        "overwrite": config.overwrite,
        "seed": config.seed,
    }
    if config.wavelength_ranges_nm is not None:
        mapping["wavelength_ranges_nm"] = _plain(config.wavelength_ranges_nm)
    return mapping | {
        "forward_model": _forward_model_mapping(config.forward_model),
        "plumes": [
            {
                "file": str(plume.file),
                "time_index": plume.time_index,
                "anchor": _anchor_mapping(plume.anchor),
                "rotation": plume.rotation,
                "flip": plume.flip,
                "emission_scale": plume.emission_scale,
            }
            for plume in config.plumes
        ],
    }


def dump_run(config: RunConfig) -> str:
    """Return a run as YAML."""
    return yaml.safe_dump(run_to_mapping(config), sort_keys=False)


__all__ = [
    "AnchorSpec",
    "ForwardModelSpec",
    "PlumeSpec",
    "RunConfig",
    "dump_run",
    "load_run",
    "run_to_mapping",
    "wavelength_ranges",
]
