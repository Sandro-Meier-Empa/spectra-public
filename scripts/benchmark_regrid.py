#!/usr/bin/env python3
"""Time the polygon regridder on a real scene with a large plume.

Places a synthetic LES plume of ``--cells`` x ``--cells`` cells (spacing
``--dx`` metres) at the scene centre and regrids it with the exact polygon
overlap, printing wall time, peak memory, window size and the mass error.
The supersampled bucket regridder was removed after this benchmark showed
the exact method within a factor of two of it (AVIRIS-4, 512 x 512 cells of
2 m onto a 356 x 992 pixel window: 9.2 s against 5.0 s). Every scene file is
given explicitly by role (``--file ROLE=PATH``).

Example::

    python scripts/benchmark_regrid.py --sensor emit \
        --file radiance=/path/to/EMIT_L1B_RAD.nc --file obs=/path/to/EMIT_L1B_OBS.nc \
        --cells 512 --dx 10
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

from spectra.regrid.polygon import PolygonRegridder
from spectra.sensors import get_sensor
from spectra.testing import make_plume_column
from spectra.types import Placement


def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--file", action="append", required=True, metavar="ROLE=PATH")
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--rotation", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    files = dict(item.split("=", 1) for item in args.file)
    scene = get_sensor(args.sensor).open_scene({k: Path(v) for k, v in files.items()})
    plume = make_plume_column(
        nx=args.cells, ny=args.cells, dx=args.dx, n_time=1, seed=1, sigma0_m=4 * args.dx
    )
    column = plume["column"].isel(time=0)
    column.attrs.update(plume.attrs)
    placement = Placement(
        anchor_along=scene.sizes["along"] // 2,
        anchor_across=scene.sizes["across"] // 2,
        anchor_corner="source",
        rotation_deg=args.rotation,
    )
    print(
        f"{args.sensor}: scene {dict(scene.sizes)}, plume {args.cells}x{args.cells} "
        f"cells of {args.dx} m"
    )
    regridder = PolygonRegridder()
    times = []
    for _ in range(max(1, args.repeat)):
        start = time.perf_counter()
        result = regridder.regrid_to_scene(column, scene, placement)
        times.append(time.perf_counter() - start)
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(
            f"  {regridder.name} {times[-1]:7.2f} s  window {result.data.shape}  "
            f"mass error {result.relative_error:.1e}  peak RSS {peak_gb:.2f} GB"
        )
    print(f"  best {min(times):.2f} s")


if __name__ == "__main__":
    main()
