"""One call from a run file to the embedded native file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectra.contracts.embedded import RUN_CONFIG_ATTR
from spectra.contracts.sensor_scene import validate_sensor_scene
from spectra.export.native import check_output_target, write_native
from spectra.pipeline import embed_scene
from spectra.protocols import ForwardModel
from spectra.result import EmbedResult
from spectra.run import RunConfig, dump_run, load_run
from spectra.sensors import get_sensor
from spectra.sensors.native import resolve_files
from spectra.spectral.libradtran import LibRadtranArts, LibRadtranArtsConfig


@dataclass(frozen=True, eq=False)
class EmbedRun:
    """Outcome of :func:`embed`.

    Attributes:
        output: The written native file.
        config: The run with the seed and drawn anchors filled in, as stored
            in the file's ``spectra_run_config`` attribute.
        result: The in-memory embedding result.
    """

    output: Path
    config: RunConfig
    result: EmbedResult


def forward_model_from(spec: LibRadtranArtsConfig | ForwardModel) -> ForwardModel:
    """Return a forward-model object for a run's ``forward_model`` entry."""
    if isinstance(spec, LibRadtranArtsConfig):
        return LibRadtranArts(spec)
    return spec


def resolve_run(
    run: RunConfig | Mapping[str, Any] | str | os.PathLike[str],
    *,
    output: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    overwrite: bool | None = None,
) -> RunConfig:
    """Load a run and apply overrides; draw a seed when none is set."""
    config = run if isinstance(run, RunConfig) else load_run(run)
    changes: dict[str, Any] = {}
    if output is not None:
        changes["output"] = Path(output).expanduser().resolve()
    if seed is not None:
        changes["seed"] = int(seed)
    if overwrite is not None:
        changes["overwrite"] = bool(overwrite)
    config = config.replace(**changes) if changes else config
    if config.seed is None:
        drawn = int(np.random.SeedSequence().generate_state(1, np.uint64)[0] >> 1)
        config = config.replace(seed=drawn)
    return config


def embed(
    run: RunConfig | Mapping[str, Any] | str | os.PathLike[str],
    *,
    output: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    overwrite: bool | None = None,
) -> EmbedRun:
    """Embed the plumes of a run file into its scene and write the native file.

    Args:
        run: Run file path, mapping or :class:`RunConfig`.
        output: Override of the output path.
        seed: Override of the random-anchor seed.
        overwrite: Override of the overwrite flag.

    Returns:
        The output path, the resolved run and the embedding result.
    """
    config = resolve_run(run, output=output, seed=seed, overwrite=overwrite)
    adapter = get_sensor(config.sensor)
    files = resolve_files(adapter.ROLES, config.files)
    check_output_target(files, config.output, overwrite=config.overwrite)
    scene = adapter.open_scene(files)
    validate_sensor_scene(scene, require_lazy=True)
    model = forward_model_from(config.forward_model)
    result, plumes = embed_scene(
        scene, config.plumes, model, seed=int(config.seed or 0)
    )
    resolved = config.replace(plumes=plumes)
    result.attrs[RUN_CONFIG_ATTR] = dump_run(resolved)
    path = write_native(
        adapter.LAYOUT, files, config.output, result, overwrite=config.overwrite
    )
    return EmbedRun(output=path, config=resolved, result=result)


__all__ = ["EmbedRun", "embed", "forward_model_from", "resolve_run"]
