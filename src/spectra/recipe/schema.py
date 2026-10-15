"""Recipe schema: frozen dataclasses mirroring the YAML dataset recipe.

A recipe fully determines a dataset: together with the scene catalogue and
the plume files it names, the seed fixes every sample. ``load_recipe``
validates the YAML explicitly and names the offending key in every error;
``recipe_hash`` is the provenance identity written to every product.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast, get_type_hints

import yaml

from spectra.contracts.errors import ContractError
from spectra.types import ANCHOR_CORNERS, FLIPS

RECIPE_VERSION = 1
SPLITS: tuple[str, ...] = ("train", "val", "test")
METHODS: tuple[str, ...] = ("radiance", "product")
TIME_MODES: tuple[str, ...] = ("uniform", "fixed", "list")
EMISSION_DISTRIBUTIONS: tuple[str, ...] = ("fixed", "uniform", "loguniform", "list")
EMISSION_UNITS: dict[str, float] = {"kg h-1": 1.0 / 3600.0, "kg s-1": 1.0}
ROTATION_MODES: tuple[str, ...] = ("uniform_int", "fixed", "list")
RADIANCE_LEGS: tuple[str, ...] = ("solar", "view", "both")
STATISTICS_SCOPES: tuple[str, ...] = ("window", "full_scene", "cached_scene")
PRODUCT_FORMATS: tuple[str, ...] = ("netcdf", "zarr")
GROUP_COLUMNS: tuple[str, ...] = ("background_group", "les_case")
# Recipe keys holding paths that are resolved relative to the recipe file.
_PATH_KEYS: tuple[tuple[str, ...], ...] = (
    ("sensor", "scenes"),
    ("catalog", "path"),
    ("output", "root"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _positive(value: float, key: str) -> None:
    _require(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0,
        f"{key} must be a positive finite number; got {value!r}.",
    )


@dataclass(frozen=True)
class SensorSpec:
    """Sensor adapter selection.

    Attributes:
        name: Registered sensor adapter.
        scenes: Scene table (CSV/YAML with explicit file paths) passed to
            ``list_scenes`` when no catalogue is used.
        options: Adapter constructor options.
    """

    name: str
    scenes: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the adapter name."""
        _require(bool(self.name), "sensor.name must be a registered adapter name.")


@dataclass(frozen=True)
class CatalogSpec:
    """Scene catalogue selection.

    Attributes:
        path: Catalogue directory; ``None`` lists scenes with the adapter.
        status: Catalogue statuses accepted as backgrounds.
        filters: Extra ``SceneCatalog.filter`` criteria (or ``list_scenes``
            filters without a catalogue).
    """

    path: str | None = None
    status: tuple[str, ...] = ("eligible",)
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the status list."""
        _require(len(self.status) > 0, "catalog.status must list at least one status.")


@dataclass(frozen=True)
class TimePolicy:
    """How the LES snapshot of a sample is chosen.

    Attributes:
        mode: ``uniform`` over all snapshots at or after
            ``min_simulation_time_s``; ``fixed`` uses the single index in
            ``indices``; ``list`` draws uniformly from ``indices``.
        min_simulation_time_s: Spin-up time excluded by ``uniform``.
        indices: Snapshot indices of the ``fixed`` and ``list`` modes.
    """

    mode: str = "uniform"
    min_simulation_time_s: float = 0.0
    indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the mode and its indices."""
        _require(
            self.mode in TIME_MODES,
            f"plumes.time_policy.mode must be one of {TIME_MODES}; got {self.mode!r}.",
        )
        _require(
            self.min_simulation_time_s >= 0,
            "plumes.time_policy.min_simulation_time_s must be non-negative.",
        )
        if self.mode == "uniform":
            return
        _require(
            self.indices is not None and len(self.indices) > 0,
            f"plumes.time_policy.indices is required for mode {self.mode!r}.",
        )
        assert self.indices is not None
        if self.mode == "fixed":
            _require(
                len(self.indices) == 1,
                "plumes.time_policy.indices must hold exactly one index for 'fixed'.",
            )
        _require(
            all(isinstance(i, int) and i >= 0 for i in self.indices),
            "plumes.time_policy.indices must be non-negative integers.",
        )


@dataclass(frozen=True)
class EmissionSpec:
    """Emission-rate distribution of the embedded plumes.

    Attributes:
        distribution: ``fixed`` (one value in ``values``), ``uniform`` or
            ``loguniform`` between ``low`` and ``high``, or ``list`` (uniform
            choice among ``values``).
        low: Lower bound of the continuous distributions.
        high: Upper bound of the continuous distributions.
        values: Values of ``fixed`` and ``list``.
        units: ``kg h-1`` or ``kg s-1``.
    """

    distribution: str = "loguniform"
    low: float | None = None
    high: float | None = None
    values: tuple[float, ...] | None = None
    units: str = "kg h-1"

    def __post_init__(self) -> None:
        """Validate the distribution parameters."""
        _require(
            self.distribution in EMISSION_DISTRIBUTIONS,
            "plumes.emission.distribution must be one of "
            f"{EMISSION_DISTRIBUTIONS}; got {self.distribution!r}.",
        )
        _require(
            self.units in EMISSION_UNITS,
            f"plumes.emission.units must be one of {tuple(EMISSION_UNITS)}.",
        )
        if self.distribution in ("uniform", "loguniform"):
            _require(
                self.low is not None and self.high is not None,
                f"plumes.emission.low and high are required for {self.distribution!r}.",
            )
            assert self.low is not None and self.high is not None
            _positive(self.low, "plumes.emission.low")
            _positive(self.high, "plumes.emission.high")
            _require(self.low <= self.high, "plumes.emission.low must not exceed high.")
        else:
            _require(
                self.values is not None and len(self.values) > 0,
                f"plumes.emission.values is required for {self.distribution!r}.",
            )
            assert self.values is not None
            if self.distribution == "fixed":
                _require(
                    len(self.values) == 1,
                    "plumes.emission.values must hold one value for 'fixed'.",
                )
            for value in self.values:
                _positive(value, "plumes.emission.values")

    @property
    def to_kg_s(self) -> float:
        """Return the factor converting ``units`` to kg s-1."""
        return EMISSION_UNITS[self.units]

    @property
    def maximum_kg_s(self) -> float:
        """Return the largest emission rate the distribution can draw, in kg s-1."""
        if self.distribution in ("uniform", "loguniform"):
            assert self.high is not None
            return float(self.high) * self.to_kg_s
        assert self.values is not None
        return float(max(self.values)) * self.to_kg_s


@dataclass(frozen=True)
class PlumeSpec:
    """Plume-column files and how snapshots are drawn from them.

    Attributes:
        files: Plume column files; shell globs are allowed.
        time_policy: Snapshot selection.
        emission: Emission-rate distribution.
        column_convention: Convention of the injected column.
    """

    files: tuple[str, ...]
    time_policy: TimePolicy = field(default_factory=TimePolicy)
    emission: EmissionSpec = field(default_factory=EmissionSpec)
    column_convention: str = "vertical"

    def __post_init__(self) -> None:
        """Validate the file list and the convention."""
        _require(len(self.files) > 0, "plumes.files must list at least one file.")
        _require(
            self.column_convention in ("vertical", "slant"),
            "plumes.column_convention must be 'vertical' or 'slant'.",
        )


@dataclass(frozen=True)
class RotationSpec:
    """Rotation distribution in whole degrees.

    Attributes:
        mode: ``uniform_int`` in ``[low, high]``, ``fixed`` (``value``) or
            ``list`` (uniform choice among ``values``).
        low: Lower bound of ``uniform_int``.
        high: Upper bound of ``uniform_int`` (inclusive).
        value: Rotation of ``fixed``.
        values: Rotations of ``list``.
    """

    mode: str = "uniform_int"
    low: int = 0
    high: int = 359
    value: int = 0
    values: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the mode and its bounds."""
        _require(
            self.mode in ROTATION_MODES,
            f"placement.rotations_deg.mode must be one of {ROTATION_MODES}.",
        )
        for name in ("low", "high", "value"):
            number = getattr(self, name)
            _require(
                isinstance(number, int) and -359 <= number <= 359,
                f"placement.rotations_deg.{name} must be a whole number in "
                "[-359, 359].",
            )
        _require(self.low <= self.high, "placement.rotations_deg.low must be <= high.")
        if self.mode == "list":
            _require(len(self.values) > 0, "placement.rotations_deg.values is empty.")
            _require(
                all(isinstance(v, int) and -359 <= v <= 359 for v in self.values),
                "placement.rotations_deg.values must be whole degrees in [-359, 359].",
            )


@dataclass(frozen=True)
class PlacementSpec:
    """Placement distribution of the plumes in the swath.

    Attributes:
        mode: Only ``anchor`` is supported.
        rotations_deg: Rotation distribution.
        flips: Reflection axis to weight.
        anchor_corners: LES corner cells drawn uniformly.
        min_outside_fraction: Smallest accepted plume mass fraction outside
            the swath.
        max_outside_fraction: Largest accepted outside mass fraction.
        max_attempts: Anchor draws before giving up.
        window_shape: Minimum ``(n_across, n_along)`` of the sample window.
    """

    mode: str = "anchor"
    rotations_deg: RotationSpec = field(default_factory=RotationSpec)
    flips: dict[str, float] = field(default_factory=lambda: {"none": 1.0})
    anchor_corners: tuple[str, ...] = ANCHOR_CORNERS
    min_outside_fraction: float = 0.0
    max_outside_fraction: float = 0.0
    max_attempts: int = 10_000
    window_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Validate the placement distribution."""
        _require(self.mode == "anchor", "placement.mode must be 'anchor'.")
        _require(len(self.flips) > 0, "placement.flips must weight at least one axis.")
        unknown = set(self.flips) - set(FLIPS)
        _require(not unknown, f"placement.flips has unknown axes {sorted(unknown)}.")
        _require(
            all(w >= 0 for w in self.flips.values()) and sum(self.flips.values()) > 0,
            "placement.flips weights must be non-negative with a positive sum.",
        )
        unknown = set(self.anchor_corners) - set(ANCHOR_CORNERS)
        _require(
            len(self.anchor_corners) > 0 and not unknown,
            f"placement.anchor_corners must be a subset of {ANCHOR_CORNERS}.",
        )
        for name in ("min_outside_fraction", "max_outside_fraction"):
            value = getattr(self, name)
            _require(0 <= value < 1, f"placement.{name} must lie in [0, 1).")
        _require(
            self.min_outside_fraction <= self.max_outside_fraction,
            "placement.min_outside_fraction must not exceed max_outside_fraction.",
        )
        _require(self.max_attempts >= 1, "placement.max_attempts must be >= 1.")
        if self.window_shape is not None:
            _require(
                len(self.window_shape) == 2
                and all(isinstance(n, int) and n >= 1 for n in self.window_shape),
                "placement.window_shape must be two positive integers.",
            )


@dataclass(frozen=True)
class LutSpec:
    """Absorption-table nodes of the radiance embedding.

    Attributes:
        enhancement_min_ppm: Smallest node in ppm.
        enhancement_max_ppm: Largest node in ppm.
        n_nodes: Number of logarithmically spaced nodes.
        layer_height_m: Nominal lowest-layer height of the forward model.
        radiance_leg: Photon-path segment containing the plume.
    """

    enhancement_min_ppm: float = 0.01
    enhancement_max_ppm: float = 200.0
    n_nodes: int = 12
    layer_height_m: float = 1000.0
    radiance_leg: str = "both"

    def __post_init__(self) -> None:
        """Validate the node range."""
        _positive(self.enhancement_min_ppm, "embedding.lut.enhancement_min_ppm")
        _positive(self.enhancement_max_ppm, "embedding.lut.enhancement_max_ppm")
        _require(
            self.enhancement_min_ppm < self.enhancement_max_ppm,
            "embedding.lut.enhancement_min_ppm must be below enhancement_max_ppm.",
        )
        _require(self.n_nodes >= 2, "embedding.lut.n_nodes must be at least 2.")
        _positive(self.layer_height_m, "embedding.lut.layer_height_m")
        _require(
            self.radiance_leg in RADIANCE_LEGS,
            f"embedding.lut.radiance_leg must be one of {RADIANCE_LEGS}.",
        )


@dataclass(frozen=True)
class ProviderSpec:
    """Unit-absorption provider selection.

    Attributes:
        provider: Registered provider name.
        options: Provider constructor options.
    """

    provider: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the provider name."""
        _require(bool(self.provider), "embedding.unit_absorption.provider is empty.")


@dataclass(frozen=True)
class EmbeddingSettings:
    """Regridder and radiance-embedding settings.

    Attributes:
        regridder: Registered regridder.
        regridder_options: Regridder constructor options.
        lut: Absorption-table nodes.
        unit_absorption: Provider of the radiance method.
    """

    regridder: str = "polygon"
    regridder_options: dict[str, Any] = field(default_factory=dict)
    lut: LutSpec = field(default_factory=LutSpec)
    unit_absorption: ProviderSpec | None = None


@dataclass(frozen=True)
class RetrievalSpec:
    """Retriever and retrieval settings of the radiance method.

    Attributes:
        retriever: Registered retriever.
        options: Retriever constructor options.
        statistics_scope: Background statistics scope.
        target_enhancement_ppm: Enhancement of the target spectrum.
        band_windows_nm: Wavelength windows of the retrieval.
        min_n_samples: Minimum background spectra per statistic.
        exclude_injected_from_background: Mask the plume from the statistics.
    """

    retriever: str
    options: dict[str, Any] = field(default_factory=dict)
    statistics_scope: str = "window"
    target_enhancement_ppm: float = 10.0
    band_windows_nm: tuple[tuple[float, float], ...] = (
        (1480.0, 1800.0),
        (2080.0, 2500.0),
    )
    min_n_samples: int = 7000
    exclude_injected_from_background: bool = False

    def __post_init__(self) -> None:
        """Validate the retrieval settings."""
        _require(bool(self.retriever), "retrieval.retriever is empty.")
        _require(
            self.statistics_scope in STATISTICS_SCOPES,
            f"retrieval.statistics_scope must be one of {STATISTICS_SCOPES}.",
        )
        _positive(self.target_enhancement_ppm, "retrieval.target_enhancement_ppm")
        _require(self.min_n_samples >= 1, "retrieval.min_n_samples must be >= 1.")
        for window in self.band_windows_nm:
            _require(
                len(window) == 2 and window[0] < window[1],
                "retrieval.band_windows_nm entries must be [lower, upper] pairs.",
            )


@dataclass(frozen=True)
class NegativesSpec:
    """Plume-free samples.

    Attributes:
        fraction: Probability that a sample carries no plume.
    """

    fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fraction."""
        _require(0 <= self.fraction < 1, "negatives.fraction must lie in [0, 1).")


@dataclass(frozen=True)
class LabelSpec:
    """Label thresholds forwarded to prism.

    Attributes:
        detection_snr: Injected-column SNR of a target pixel.
        ambiguous_snr: SNR above which non-target pixels are ignored.
    """

    detection_snr: float = 2.0
    ambiguous_snr: float = 1.0

    def __post_init__(self) -> None:
        """Validate the thresholds like prism's ``CompositeSettings``."""
        _positive(self.detection_snr, "labels.detection_snr")
        _require(
            0 < self.ambiguous_snr <= self.detection_snr,
            "labels.ambiguous_snr must lie in (0, labels.detection_snr].",
        )


@dataclass(frozen=True)
class SplitSpec:
    """Leakage-safe split assignment.

    Attributes:
        ratios: Fraction of groups (and samples) per split.
        group_by: Group columns assigned independently.
        seed: Seed of the group assignment; defaults to the recipe seed.
    """

    ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    group_by: tuple[str, ...] = GROUP_COLUMNS
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the ratios and the group columns."""
        _require(
            set(self.ratios) == set(SPLITS),
            f"split.ratios must define exactly {SPLITS}.",
        )
        _require(
            all(0 <= value <= 1 for value in self.ratios.values())
            and math.isclose(sum(self.ratios.values()), 1.0, abs_tol=1e-9),
            "split.ratios must be non-negative and sum to 1.",
        )
        _require(
            set(self.group_by) == set(GROUP_COLUMNS),
            f"split.group_by must be {GROUP_COLUMNS}.",
        )


@dataclass(frozen=True)
class OutputSpec:
    """Where and how products are written.

    Attributes:
        root: Dataset root; samples go to ``<root>/samples``.
        product_format: ``netcdf`` or ``zarr``.
        dataset_version: Immutable version label passed to prism.
    """

    root: str
    dataset_version: str
    product_format: str = "netcdf"

    def __post_init__(self) -> None:
        """Validate the format and the version label."""
        _require(bool(self.root), "output.root is empty.")
        _require(bool(self.dataset_version), "output.dataset_version is empty.")
        _require(
            self.product_format in PRODUCT_FORMATS,
            f"output.product_format must be one of {PRODUCT_FORMATS}.",
        )

    @property
    def suffix(self) -> str:
        """Return the file suffix of the product format."""
        return ".zarr" if self.product_format == "zarr" else ".nc"


@dataclass(frozen=True)
class SlurmSpec:
    """Slurm job array on CSCS Alps.

    Attributes:
        account: Slurm account.
        run_time: Wall time per array job (``HH:MM:SS``).
        array_size: Number of array jobs.
        tasks_per_job: Shards generated concurrently by one array job.
        uenv: User environment image, if any.
        partition: Slurm partition; defaults to ``normal``.
        cluster: Alps cluster.
        processor: Processor architecture.
    """

    account: str
    run_time: str
    array_size: int
    tasks_per_job: int = 1
    uenv: str | None = None
    partition: str | None = None
    cluster: str = "eiger"
    processor: str = "cpu"

    def __post_init__(self) -> None:
        """Validate the array layout."""
        _require(bool(self.account), "compute.slurm.account is empty.")
        _require(self.array_size >= 1, "compute.slurm.array_size must be >= 1.")
        _require(self.tasks_per_job >= 1, "compute.slurm.tasks_per_job must be >= 1.")


@dataclass(frozen=True)
class ComputeSpec:
    """Execution settings.

    Attributes:
        chunks: Dask chunks passed to ``open_scene``.
        slurm: Slurm array settings, if generation runs on a cluster.
    """

    chunks: dict[str, int] = field(default_factory=dict)
    slurm: SlurmSpec | None = None


@dataclass(frozen=True)
class Recipe:
    """A complete, validated dataset recipe.

    Attributes:
        name: Dataset name; prefixes every sample id.
        seed: Master seed of all draws.
        n_samples: Number of samples.
        sensor: Sensor adapter.
        catalog: Background scene selection.
        plumes: Plume files and draws.
        placement: Placement draws.
        method: ``radiance`` or ``product`` embedding.
        embedding: Regridder and LUT settings.
        retrieval: Retrieval settings (radiance method only).
        negatives: Plume-free sample fraction.
        labels: Label thresholds.
        split: Split assignment.
        output: Output location and format.
        compute: Execution settings.
        recipe_version: Schema version of the recipe.
    """

    name: str
    seed: int
    n_samples: int
    sensor: SensorSpec
    plumes: PlumeSpec
    output: OutputSpec
    method: str = "radiance"
    catalog: CatalogSpec = field(default_factory=CatalogSpec)
    placement: PlacementSpec = field(default_factory=PlacementSpec)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSpec | None = None
    negatives: NegativesSpec = field(default_factory=NegativesSpec)
    labels: LabelSpec = field(default_factory=LabelSpec)
    split: SplitSpec = field(default_factory=SplitSpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    recipe_version: int = RECIPE_VERSION

    def __post_init__(self) -> None:
        """Validate cross-section consistency."""
        _require(
            self.recipe_version == RECIPE_VERSION,
            f"recipe_version must be {RECIPE_VERSION}; got {self.recipe_version!r}.",
        )
        _require(
            bool(self.name) and "/" not in self.name,
            "name must be a non-empty label without '/'.",
        )
        _require(
            isinstance(self.seed, int)
            and not isinstance(self.seed, bool)
            and self.seed >= 0,
            "seed must be a non-negative integer.",
        )
        _require(
            isinstance(self.n_samples, int) and self.n_samples >= 1,
            "n_samples must be at least 1.",
        )
        _require(self.method in METHODS, f"method must be one of {METHODS}.")
        if self.method == "product":
            _require(
                self.retrieval is None,
                "retrieval must be omitted for method 'product'; the adapter's "
                "retrieved product is used.",
            )
        else:
            _require(
                self.retrieval is not None,
                "retrieval is required for method 'radiance'.",
            )
            _require(
                self.embedding.unit_absorption is not None,
                "embedding.unit_absorption is required for method 'radiance'.",
            )
        _require(
            self.catalog.path is not None or self.sensor.scenes is not None,
            "sensor.scenes (a scene table) is required when catalog.path is null.",
        )

    @property
    def split_seed(self) -> int:
        """Return the seed of the group assignment."""
        return self.seed if self.split.seed is None else self.split.seed


# -- loading ---------------------------------------------------------------
def _is_optional(hint: object) -> tuple[bool, object]:
    """Return whether ``hint`` is ``X | None`` and the inner type."""
    origin = typing.get_origin(hint)
    if origin is types.UnionType or origin is typing.Union:
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, hint


def _coerce(value: object, hint: object, key: str) -> object:
    """Convert a YAML value to the annotated field type."""
    optional, inner = _is_optional(hint)
    if value is None:
        if optional:
            return None
        raise ContractError(f"{key} must not be null.")
    hint = inner
    if is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, Mapping):
            raise ContractError(f"{key} must be a mapping.")
        return _build(hint, value, key)
    origin = typing.get_origin(hint)
    if origin is tuple:
        if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
            raise ContractError(f"{key} must be a list.")
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _coerce(item, args[0], f"{key}[{i}]") for i, item in enumerate(value)
            )
        if len(value) != len(args):
            raise ContractError(f"{key} must have {len(args)} entries.")
        return tuple(
            _coerce(item, arg, f"{key}[{i}]")
            for i, (item, arg) in enumerate(zip(value, args, strict=True))
        )
    if origin is dict or hint is dict:
        if not isinstance(value, Mapping):
            raise ContractError(f"{key} must be a mapping.")
        return {str(k): v for k, v in value.items()}
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ContractError(f"{key} must be a number; got {value!r}.")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{key} must be an integer; got {value!r}.")
        return int(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise ContractError(f"{key} must be true or false; got {value!r}.")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ContractError(f"{key} must be a string; got {value!r}.")
        return value
    return value


def _build(cls: type, mapping: Mapping[str, Any], prefix: str) -> object:
    """Instantiate a recipe dataclass from a mapping, naming bad keys."""
    hints = get_type_hints(cls)
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(mapping) - names)
    if unknown:
        raise ContractError(f"{prefix or 'recipe'} has unknown keys {unknown}.")
    values: dict[str, Any] = {}
    for item in fields(cls):
        key = f"{prefix}.{item.name}" if prefix else item.name
        if item.name in mapping:
            values[item.name] = _coerce(mapping[item.name], hints[item.name], key)
        elif (
            item.default is dataclasses.MISSING
            and item.default_factory is dataclasses.MISSING
        ):
            raise ContractError(f"{key} is required.")
    try:
        return cls(**values)
    except ContractError as error:
        raise ContractError(f"{prefix or 'recipe'}: {error}") from error


def _resolve_paths(mapping: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve relative recipe paths against the recipe directory."""
    resolved = json.loads(json.dumps(mapping))
    for keys in _PATH_KEYS:
        section = resolved.get(keys[0])
        if isinstance(section, dict) and isinstance(section.get(keys[1]), str):
            section[keys[1]] = str((base / section[keys[1]]).resolve())
    plumes = resolved.get("plumes")
    if isinstance(plumes, dict) and isinstance(plumes.get("files"), list):
        plumes["files"] = [
            str((base / item).resolve()) if isinstance(item, str) else item
            for item in plumes["files"]
        ]
    return resolved


def recipe_from_mapping(
    mapping: Mapping[str, Any], *, base: Path | None = None
) -> Recipe:
    """Build and validate a recipe from a YAML mapping.

    Args:
        mapping: Parsed recipe.
        base: Directory relative paths are resolved against.

    Returns:
        The validated recipe.

    Raises:
        ContractError: On any schema violation, naming the key.
    """
    if not isinstance(mapping, Mapping):
        raise ContractError("A recipe must be a YAML mapping.")
    data = dict(mapping)
    if base is not None:
        data = _resolve_paths(data, base)
    return cast(Recipe, _build(Recipe, data, ""))


def load_recipe(path: str | Path) -> Recipe:
    """Load and validate a YAML recipe; relative paths resolve against the file."""
    recipe_path = Path(path)
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe {recipe_path} does not exist.")
    with recipe_path.open() as stream:
        mapping = yaml.safe_load(stream)
    return recipe_from_mapping(mapping, base=recipe_path.resolve().parent)


def _plain(value: object) -> object:
    """Return ``value`` with tuples as lists so YAML and JSON can hold it."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def recipe_to_mapping(recipe: Recipe) -> dict[str, Any]:
    """Return the recipe as a plain mapping (tuples become lists)."""
    return cast(dict[str, Any], _plain(dataclasses.asdict(recipe)))


def recipe_hash(recipe: Recipe) -> str:
    """Return the SHA-256 of the canonical JSON of the recipe."""
    payload = json.dumps(recipe_to_mapping(recipe), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_resolved(recipe: Recipe, path: str | Path) -> Path:
    """Write the fully resolved recipe (all defaults filled in) as YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as stream:
        yaml.safe_dump(recipe_to_mapping(recipe), stream, sort_keys=False)
    return target


__all__ = [
    "EMISSION_DISTRIBUTIONS",
    "EMISSION_UNITS",
    "GROUP_COLUMNS",
    "METHODS",
    "PRODUCT_FORMATS",
    "RECIPE_VERSION",
    "ROTATION_MODES",
    "SPLITS",
    "STATISTICS_SCOPES",
    "TIME_MODES",
    "CatalogSpec",
    "ComputeSpec",
    "EmbeddingSettings",
    "EmissionSpec",
    "LabelSpec",
    "LutSpec",
    "NegativesSpec",
    "OutputSpec",
    "PlacementSpec",
    "PlumeSpec",
    "ProviderSpec",
    "Recipe",
    "RetrievalSpec",
    "RotationSpec",
    "SensorSpec",
    "SlurmSpec",
    "SplitSpec",
    "TimePolicy",
    "dump_resolved",
    "load_recipe",
    "recipe_from_mapping",
    "recipe_hash",
    "recipe_to_mapping",
]
