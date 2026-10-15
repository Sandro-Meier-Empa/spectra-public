"""Scene catalogue, background QA and window selection."""

from spectra.catalog.qa import (
    DetectionMethod,
    QaResult,
    QaSettings,
    compute_scene_qa,
    compute_window_qa,
    local_snr,
)
from spectra.catalog.schema import (
    ALLOWED_TRANSITIONS,
    CATALOG_VERSION,
    COLUMNS,
    QA_METRIC_COLUMNS,
    STATUSES,
    CatalogError,
    QaThresholds,
    SceneRow,
    Status,
    record_to_row,
    row_to_record,
)
from spectra.catalog.store import SceneCatalog

__all__ = [
    "ALLOWED_TRANSITIONS",
    "CATALOG_VERSION",
    "COLUMNS",
    "QA_METRIC_COLUMNS",
    "STATUSES",
    "CatalogError",
    "DetectionMethod",
    "QaResult",
    "QaSettings",
    "QaThresholds",
    "SceneCatalog",
    "SceneRow",
    "Status",
    "compute_scene_qa",
    "compute_window_qa",
    "local_snr",
    "record_to_row",
    "row_to_record",
]
