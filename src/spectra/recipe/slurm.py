"""Slurm job array for sharded generation on CSCS Alps (``cscs`` extra)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from spectra.contracts.errors import ContractError, MissingExtraError
from spectra.recipe.generate import RESOLVED_RECIPE_FILE, dump_resolved
from spectra.recipe.schema import Recipe

SBATCH_FILE = "generate.sbatch"
LOGS_DIR = "logs"


def shard_layout(recipe: Recipe) -> tuple[int, int]:
    """Return ``(n_shards, samples_per_shard)`` of the recipe's Slurm array."""
    slurm = recipe.compute.slurm
    if slurm is None:
        raise ContractError("compute.slurm is required to write a Slurm array.")
    n_shards = slurm.array_size * slurm.tasks_per_job
    return n_shards, math.ceil(recipe.n_samples / n_shards)


def array_commands(
    recipe: Recipe, recipe_path: Path, out_dir: Path, *, python: str
) -> list[str]:
    """Return the shell lines of one array job.

    Array job ``j`` runs ``tasks_per_job`` shards ``k = j * tasks_per_job + t``
    concurrently; shard ``k`` generates ``[k * per_shard, (k + 1) * per_shard)``.
    """
    slurm = recipe.compute.slurm
    assert slurm is not None
    _, per_shard = shard_layout(recipe)
    return [
        "set -euo pipefail",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
        f"N_SAMPLES={recipe.n_samples}",
        f"PER_SHARD={per_shard}",
        f"TASKS={slurm.tasks_per_job}",
        "PIDS=()",
        "for TASK in $(seq 0 $((TASKS - 1))); do",
        "    SHARD=$((SLURM_ARRAY_TASK_ID * TASKS + TASK))",
        "    START=$((SHARD * PER_SHARD))",
        "    STOP=$((START + PER_SHARD))",
        '    if [ "$STOP" -gt "$N_SAMPLES" ]; then STOP=$N_SAMPLES; fi',
        '    if [ "$START" -ge "$N_SAMPLES" ]; then continue; fi',
        f"    {python} -m spectra.cli generate {recipe_path} --out {out_dir} "
        '--index-range "$START:$STOP" --shard "$SHARD" &',
        "    PIDS+=($!)",
        "done",
        "FAILED=0",
        'for PID in ${PIDS[@]+"${PIDS[@]}"}; do wait "$PID" || FAILED=1; done',
        'exit "$FAILED"',
    ]


def write_slurm_array(
    recipe: Recipe, out_dir: str | Path, *, python: str = sys.executable
) -> Path:
    """Write the Slurm array script generating a recipe in shards.

    The resolved recipe is written next to the script so the jobs load a
    recipe with absolute paths. Nothing is submitted.

    Args:
        recipe: Recipe with ``compute.slurm``.
        out_dir: Run directory.
        python: Interpreter running ``spectra.cli`` on the nodes.

    Returns:
        Path of the batch script.

    Raises:
        MissingExtraError: If ``cscs`` is not installed.
        ContractError: If the recipe has no Slurm settings.
    """
    slurm = recipe.compute.slurm
    if slurm is None:
        raise ContractError("compute.slurm is required to write a Slurm array.")
    try:
        from cscs import AlpsSetup
    except ImportError as error:
        raise MissingExtraError("slurm", "cscs", error) from error
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / LOGS_DIR).mkdir(exist_ok=True)
    recipe_path = dump_resolved(recipe, out / RESOLVED_RECIPE_FILE)
    setup = AlpsSetup(
        cluster=slurm.cluster,
        processor=slurm.processor,
        account=slurm.account,
        run_time=slurm.run_time,
        nodes=1,
        ntasks=1,
        partition=slurm.partition or "normal",
        uenv=slurm.uenv or "",
    )
    setup.create_sbatch_script(
        array_commands(recipe, recipe_path, out, python=python),
        job_name=recipe.name,
        file_path=out / SBATCH_FILE,
        scratch_path=out,
        output_pattern=f"{LOGS_DIR}/{{job_name}}_%A_%a.out",
        extra_sbatch={"array": f"0-{slurm.array_size - 1}"},
    )
    return out / SBATCH_FILE


def usage_hint(recipe: Recipe, script: Path, out_dir: Path) -> str:
    """Return the submission and finalisation commands for the user."""
    n_shards, per_shard = shard_layout(recipe)
    return (
        f"Wrote {script} ({n_shards} shards of up to {per_shard} samples).\n"
        f"Run the dry run first if not done yet: spectra generate "
        f"{out_dir / RESOLVED_RECIPE_FILE} --out {out_dir} --dry-run\n"
        f"Submit with: sbatch {script}\n"
        f"When every shard finished: spectra finalize "
        f"{out_dir / RESOLVED_RECIPE_FILE} --out {out_dir}"
    )


__all__ = [
    "LOGS_DIR",
    "SBATCH_FILE",
    "array_commands",
    "shard_layout",
    "usage_hint",
    "write_slurm_array",
]
