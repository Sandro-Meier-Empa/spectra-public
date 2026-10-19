#!/usr/bin/env python3
# Frozen provenance script for the pre-port code; not type-checked.
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false, reportArgumentType=false
"""Build the golden fixture with the pre-migration ``aviris4_ch4`` code on real data.

Kept for provenance only: ``aviris4_ch4.embedding`` and the MicroHH/CDO half
of ``aviris4_ch4.regridding`` were removed after the port (aviris4-ch4
changelog 2026-10-14). To rebuild the fixture, check out aviris4-ch4 at commit
2b1d8c5 (the last commit carrying that code) and run this script from there.

The fixture freezes small crops of one AVIRIS-4 flight line, one MicroHH plume
snapshot, a real absorption look-up table and the outputs of the
``aviris4_ch4`` embedding, placement and merging functions. The SPECTRA port is
tested against these outputs. Every input file is given on the command line
(``--radiance``, ``--metadata``, ``--microhh-case``, ``--resources``,
``--root``); the fixture is versioned, so this script only runs when the
reference has to be rebuilt.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import aviris4_ch4.embedding as embedding
import aviris4_ch4.mf as mf
import aviris4_ch4.misc as misc
import aviris4_ch4.regridding as regridding
import numpy as np
import xarray as xr

ACROSS = slice(20, 60)
ALONG = slice(6980, 7140)
BANDS = (2080, 2500)
BAND_STEP = 2
TIME_INDEX = 120
PLUME_X = slice(512, 542)
PLUME_Y = slice(505, 517)
XCH4_BG = 1.93191
ELEVATION = 95.0
LAYER_HEIGHT = 1000.0
NODES = np.logspace(np.log10(0.01), np.log10(200.0), 8)
PLACEMENTS = {
    "rot": {"anchor": "upper left", "rotation": -20, "flip_axis": None},
    "flip": {"anchor": "lower right", "rotation": 10, "flip_axis": "x"},
}


def sha256(values: np.ndarray) -> str:
    """Return the digest of an array's bytes, shape and dtype."""
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def jsonable(value: object) -> object:
    """Convert numpy scalars, arrays and slices for JSON."""
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def write(dataset: xr.Dataset | xr.DataArray, path: Path) -> None:
    """Write a NetCDF file, replacing any existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    dataset.to_netcdf(path)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def build(
    root: Path,
    *,
    radiance_path: Path,
    metadata_path: Path,
    microhh_case: Path,
    resources: Path,
) -> None:
    """Write the fixture below ``root`` from the explicitly given input files."""
    inputs = root / "inputs"
    expected = root / "expected"
    record: dict[str, object] = {
        "built_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "aviris4_ch4_version": getattr(
            __import__("aviris4_ch4"), "__version__", "unknown"
        ),
        "source_files": {
            "metadata": str(metadata_path),
            "radiance": str(radiance_path),
            "microhh_case": str(microhh_case),
            "resources": str(resources),
        },
        "window": {"across": jsonable(ACROSS), "along": jsonable(ALONG)},
        "bands": {"range_nm": BANDS, "step": BAND_STEP},
        "plume": {
            "time_index": TIME_INDEX,
            "x": jsonable(PLUME_X),
            "y": jsonable(PLUME_Y),
        },
    }

    # 1. Metadata window, re-indexed so it acts as a complete small swath.
    metadata = xr.open_dataset(metadata_path)
    md = metadata.isel(across=ACROSS, along=ALONG).load()
    md = md.assign_coords(
        across=np.arange(md.sizes["across"]), along=np.arange(md.sizes["along"])
    )
    md.attrs.update(
        {
            "golden_source_file": metadata_path.name,
            "golden_across_start": int(ACROSS.start),
            "golden_along_start": int(ALONG.start),
        }
    )
    write(md, inputs / "metadata_window.nc")

    # 2. Radiance window with a thinned SWIR band subset.
    radiance = xr.open_dataset(
        radiance_path, chunks={"across": 8, "along": 512, "band": -1}
    )
    rw = radiance.isel(across=ACROSS, along=ALONG)
    rw = misc.select_wavelengths(rw, wavelengths=[BANDS])
    rw = rw.isel(band=slice(0, None, BAND_STEP)).load()
    # The converter stores wavelength/fwhm as data variables; the embedding
    # kernel needs them as coordinates of the radiance.
    rw = rw.set_coords(
        [name for name in ("wavelength", "fwhm") if name in rw.data_vars]
    )
    rw = rw.assign_coords(
        across=np.arange(rw.sizes["across"]), along=np.arange(rw.sizes["along"])
    )
    rw.attrs.update(
        {
            "golden_source_file": radiance_path.name,
            "golden_across_start": int(ACROSS.start),
            "golden_along_start": int(ALONG.start),
        }
    )
    write(rw, inputs / "radiance_window.nc")

    # 3. MicroHH snapshot crop around the source, native layout (no attrs).
    import microhh_processing_chain.initialisation as init
    import microhh_processing_chain.io as mio

    case = init.read_case(case_directory=microhh_case, scratch_path=microhh_case.parent)
    column_series = mio.open_microhh_nc(case, variable="ch4_path.xy")
    snapshot = column_series.isel(time=TIME_INDEX).sel(x=PLUME_X, y=PLUME_Y)
    simulation_time = float(snapshot["time"].values)
    snapshot = snapshot.drop_vars("time")
    snapshot = snapshot.assign_coords(
        x=snapshot.x - snapshot.x[0], y=snapshot.y - snapshot.y[0]
    )
    raw = snapshot[["ch4_path"]].load()
    raw.attrs = {
        "golden_case": microhh_case.name,
        "golden_time_index": TIME_INDEX,
        "simulation_time_s": simulation_time,
    }
    write(raw, inputs / "microhh_raw.nc")
    record["plume"]["simulation_time_s"] = simulation_time
    record["plume"]["total_column_sum"] = float(raw["ch4_path"].sum())

    # 4. Real absorption table and retrieval target.
    lat = float(md["lat"].mean())
    lon = float(md["lon"].mean())
    altitude = float(md["aircraft_altitude"].mean())
    unit_abs = mf.UnitAbsorption(
        gas="CH4",
        scenario="midlatitude-summer",
        lat=lat,
        lon=lon,
        time_utc=rw["time"].values,
        wmin=1000,
        wmax=2500,
        vza=0.0,
        cw=rw["wavelength"].values,
        fwhm=rw["fwhm"].values,
        rho=1.0,
        elevation=ELEVATION,
        altitude=altitude,
        xch4_bg=XCH4_BG,
        uvspec_path=resources / "spectrum",
        sol_reference_path=resources / "spectrum",
        arts_path=resources / "arts",
    )
    lut = embedding.build_slant_absorption_lut(
        unit_abs, NODES, layer_height=LAYER_HEIGHT
    )
    write(lut, inputs / "lut.nc")
    target = unit_abs.compute_unit_absorption_spectrum(enhancement=10.0)
    write(target, inputs / "target_10ppm.nc")
    record["geometry"] = {
        "lat": lat,
        "lon": lon,
        "time_utc": str(rw["time"].values),
        "sza": float(unit_abs.sza),
        "vza": 0.0,
        "elevation": ELEVATION,
        "altitude": altitude,
        "xch4_bg": XCH4_BG,
        "layer_height": LAYER_HEIGHT,
        "nodes_ppm": NODES.tolist(),
    }

    # 5. Placements with the current aviris4_ch4 code.
    record["placements"] = {}
    window_metadata_path = inputs / "metadata_window.nc"
    n_across, n_along = md.sizes["across"], md.sizes["along"]
    for name, options in PLACEMENTS.items():
        chosen = None
        for across_index in range(2, n_across - 2, 3):
            for along_index in range(5, n_along - 5, 5):
                try:
                    result = regridding.prepare_microhh_for_aviris_cdo(
                        raw,
                        window_metadata_path,
                        expected / f"placement_{name}_cdo_source.nc",
                        expected / f"placement_{name}_cdo_target.nc",
                        "ch4_path",
                        x_index=along_index,
                        y_index=across_index,
                        overwrite=True,
                        **options,
                    )
                except ValueError:
                    continue
                if result.outside_fraction == 0.0:
                    chosen = result
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"No fully contained placement found for {name}.")
        record["placements"][name] = {
            **options,
            **{
                key: jsonable(getattr(chosen, key))
                for key in (
                    "along_index",
                    "across_index",
                    "along_slice",
                    "across_slice",
                    "longitude",
                    "latitude",
                    "x_offset",
                    "y_offset",
                    "rotation",
                    "flip_axis",
                    "outside_fraction",
                )
            },
        }
        print(f"placement {name}: {record['placements'][name]}")

    # A partial placement near the swath edge for the outside-mass path.
    partial = regridding.prepare_microhh_for_aviris_cdo(
        raw,
        window_metadata_path,
        expected / "placement_partial_cdo_source.nc",
        expected / "placement_partial_cdo_target.nc",
        "ch4_path",
        x_index=n_along - 12,
        y_index=n_across - 4,
        overwrite=True,
        anchor="lower left",
        rotation=0,
    )
    record["placements"]["partial"] = {
        "anchor": "lower left",
        "rotation": 0,
        "flip_axis": None,
        **{
            key: jsonable(getattr(partial, key))
            for key in (
                "along_index",
                "across_index",
                "along_slice",
                "across_slice",
                "longitude",
                "latitude",
                "x_offset",
                "y_offset",
                "outside_fraction",
            )
        },
    }
    print(f"placement partial: {record['placements']['partial']}")

    # 6. Synthetic on-grid column and the embedding oracles.
    across = np.arange(rw.sizes["across"])[:, None]
    along = np.arange(rw.sizes["along"])[None, :]
    blob_values = 0.02 * np.exp(
        -0.5 * (((across - 20) / 6.0) ** 2 + ((along - 80) / 20.0) ** 2)
    )
    blob = xr.DataArray(
        blob_values.astype("float32"),
        dims=("across", "along"),
        coords={"across": rw["across"], "along": rw["along"]},
        name="column",
        attrs={"units": "kg m-2", "long_name": "synthetic plume column"},
    )
    write(blob.to_dataset(), inputs / "column_on_grid.nc")

    embedded_v = embedding.embed_column_in_radiance(
        rw["radiance"], blob, lut, column_convention="vertical"
    ).compute()
    embedded_s = embedding.embed_column_in_radiance(
        rw["radiance"], blob, lut, column_convention="slant"
    ).compute()
    injected = embedding.injected_column(blob, convention="vertical")
    write(
        embedded_v.to_dataset(name="radiance"),
        expected / "embedded_radiance_vertical.nc",
    )
    write(
        embedded_s.to_dataset(name="radiance"), expected / "embedded_radiance_slant.nc"
    )
    write(
        injected.to_dataset(name="injected_ch4_mass"), expected / "injected_ch4_mass.nc"
    )

    rng = np.random.default_rng(20261014)
    observed = xr.DataArray(
        rng.normal(0.0, 100.0, size=blob.shape).astype("float32"),
        dims=blob.dims,
        coords=blob.coords,
        name="CH4",
        attrs={"units": "ppm-m", "long_name": "synthetic matched-filter column"},
    )
    write(observed.to_dataset(), inputs / "observed_product.nc")
    product_ppm = embedding.embed_column_in_product(observed, blob)
    product_kg = embedding.embed_column_in_product(observed, blob, output_units="kg/m2")
    write(product_ppm.to_dataset(), expected / "product_embedded_ppm.nc")
    write(product_kg.to_dataset(), expected / "product_embedded_kg.nc")
    record["provenance_radiance"] = jsonable(
        embedding.embedding_provenance(
            method="radiance",
            geometric_amf=lut.attrs["geometric_air_mass_factor"],
            lut=lut,
            column_convention="vertical",
        )
    )
    record["provenance_product"] = jsonable(
        embedding.embedding_provenance(method="product")
    )

    # 7. Merge oracle with a fake remapped window.
    fake = blob.isel(across=slice(5, 35), along=slice(10, 150)).to_dataset(
        name="ch4_path"
    )
    fake = fake.drop_vars(["across", "along"])
    fake.attrs.update(
        {
            "embedding_across_start": 5,
            "embedding_across_stop": 35,
            "embedding_along_start": 10,
            "embedding_along_stop": 150,
            "embedding_grid_method": "fake",
        }
    )
    write(fake, inputs / "fake_remapped.nc")
    merged = regridding.merge_aviris_with_microhh(rw, fake).drop_vars("radiance")
    write(merged, expected / "merged.nc")

    # 8. Hashes and provenance record.
    record["hashes"] = {
        "embedded_radiance_vertical": sha256(embedded_v.values),
        "embedded_radiance_slant": sha256(embedded_s.values),
        "injected_ch4_mass": sha256(injected.values),
        "product_embedded_ppm": sha256(product_ppm.values),
        "product_embedded_kg": sha256(product_kg.values),
        "lut_optical_depth": sha256(lut["optical_depth"].values),
    }
    for name in ("rot", "flip", "partial"):
        with xr.open_dataset(expected / f"placement_{name}_cdo_source.nc") as source:
            record["hashes"][f"placement_{name}_source_ch4_path"] = sha256(
                source["ch4_path"].values
            )
            record["hashes"][f"placement_{name}_source_lonc"] = sha256(
                source["lonc"].values
            )
        with xr.open_dataset(expected / f"placement_{name}_cdo_target.nc") as target:
            record["hashes"][f"placement_{name}_target_lonc"] = sha256(
                target["lonc"].values
            )
    (root / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print("record written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radiance",
        type=Path,
        required=True,
        help="Full path of the L1 radiance cube of the flight line.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Full path of the PARGE metadata file of the flight line.",
    )
    parser.add_argument(
        "--microhh-case",
        type=Path,
        required=True,
        help="MicroHH case directory of the plume snapshot.",
    )
    parser.add_argument(
        "--resources",
        type=Path,
        required=True,
        help="libRadtran/ARTS resource directory (spectrum/, arts/).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Fixture directory to write (tests/golden of the checkout).",
    )
    args = parser.parse_args()
    build(
        args.root,
        radiance_path=args.radiance,
        metadata_path=args.metadata,
        microhh_case=args.microhh_case,
        resources=args.resources,
    )


if __name__ == "__main__":
    main()
