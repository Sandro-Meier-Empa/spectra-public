"""Conservative regridding of placed plume columns onto sensor grids."""

from spectra.regrid.base import RegridResult
from spectra.regrid.polygon import PolygonRegridder, regrid_polygons
from spectra.regrid.windows import support_window

__all__ = ["PolygonRegridder", "RegridResult", "regrid_polygons", "support_window"]
