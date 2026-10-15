"""Recipe-driven, seeded dataset generation."""

from spectra.recipe.finalize import finalize, merge_shards
from spectra.recipe.generate import (
    GeneratedSample,
    Plugins,
    RunCache,
    build_plugins,
    dry_run,
    generate,
    generate_sample,
    lut_coverage,
)
from spectra.recipe.params import (
    ContextResources,
    GenerationContext,
    PlumeEntry,
    SampleParams,
    assign_splits,
    build_context,
    build_parameter_table,
    derive_sample_params,
    sample_rng,
)
from spectra.recipe.schema import (
    Recipe,
    dump_resolved,
    load_recipe,
    recipe_from_mapping,
    recipe_hash,
)
from spectra.recipe.slurm import write_slurm_array

__all__ = [
    "ContextResources",
    "GeneratedSample",
    "GenerationContext",
    "Plugins",
    "PlumeEntry",
    "Recipe",
    "RunCache",
    "SampleParams",
    "assign_splits",
    "build_context",
    "build_parameter_table",
    "build_plugins",
    "derive_sample_params",
    "dry_run",
    "dump_resolved",
    "finalize",
    "generate",
    "generate_sample",
    "load_recipe",
    "lut_coverage",
    "merge_shards",
    "recipe_from_mapping",
    "recipe_hash",
    "sample_rng",
    "write_slurm_array",
]
