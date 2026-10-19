#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Record ``aviris4_ch4.mf.UnitAbsorption`` references before the port.

SPECTRA ports the libRadtran/ARTS forward model out of ``aviris4_ch4``; this
script freezes what the original code produces so the port can be tested for
bit-identical output on two sensors. It runs once, in an environment where
``aviris4_ch4`` (``mf.py`` unchanged since commit 6d4ed8b) and SPECTRA v0.1.0
are installed, and writes below ``--out``:

- ``{case}_lut.nc``: the absorption table built by
  :func:`spectra.spectral.lut.build_slant_absorption_lut` on the model, exactly
  as the ``libradtran_arts`` provider of v0.1.0 did;
- ``{case}_target_10ppm.nc``: the unit absorption spectrum at 10 ppm plus the
  ``fwhm`` of the bands;
- ``profiles.json``: the ARTS profiles parsed with ``lxml`` and the reference
  column averages, so the stdlib XML parser of the port can be compared;
- ``record.json``: geometry (including the ephemeris solar zenith angle the
  model derived), backgrounds, versions and digests.

The band sets are read from one EMIT L1B RAD file and one AVIRIS-4 radiance
file given on the command line; every input path is explicit.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import aviris4_ch4.mf as mf
import numpy as np
import xarray as xr

from spectra.spectral.lut import build_slant_absorption_lut

SCENARIO = "midlatitude-summer"
WMIN, WMAX = 1000, 2500
NODES = np.logspace(np.log10(0.01), np.log10(200.0), 12)
LAYER_HEIGHT = 1000.0
PROFILE_GASES = ("CH4", "CO2", "H2O")
ARTS_FILES = ("z", "t", *PROFILE_GASES)

# Fixed observation cases. The EMIT case sets all three backgrounds to cover
# the CO2/H2O rescaling branch that the original golden table does not reach;
# ``aviris4_high`` puts the surface in the second model layer.
CASES = {
    "emit": {
        "sensor": "emit",
        "lat": 32.1,
        "lon": -102.9,
        "time_utc": "2026-08-19T18:02:47",
        "vza": 8.0,
        "elevation": 150.0,
        "altitude": 400_000.0,
        "backgrounds": {"CH4": 1.95, "CO2": 420.0, "H2O": 5000.0},
    },
    "aviris4": {
        "sensor": "aviris4",
        "lat": 50.1,
        "lon": 19.0,
        "time_utc": "2024-09-28T07:29:27",
        "vza": 12.0,
        "elevation": 280.0,
        "altitude": 4280.0,
        "backgrounds": {"CH4": 1.93191, "CO2": None, "H2O": None},
    },
    "aviris4_high": {
        "sensor": "aviris4",
        "lat": 46.5,
        "lon": 8.0,
        "time_utc": "2025-07-01T10:00:00",
        "vza": 3.0,
        "elevation": 1350.0,
        "altitude": 5200.0,
        "backgrounds": {"CH4": 1.93191, "CO2": None, "H2O": None},
    },
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repository: Path) -> dict[str, object]:
    """Return the commit and dirty flag of a git checkout, if available."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def emit_bands(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return EMIT band centres and widths inside the modelled window."""
    with xr.open_dataset(path, group="sensor_band_parameters") as bands:
        wavelength = bands["wavelengths"].values
        fwhm = bands["fwhm"].values
    keep = (wavelength >= WMIN) & (wavelength <= WMAX)
    return wavelength[keep], fwhm[keep]


def aviris4_bands(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return AVIRIS-4 band centres and widths inside the modelled window."""
    with xr.open_dataset(path) as radiance:
        wavelength = radiance["wavelength"].values
        fwhm = radiance["fwhm"].values
    keep = (wavelength >= WMIN) & (wavelength <= WMAX)
    return wavelength[keep], fwhm[keep]


def write(dataset: xr.Dataset, path: Path) -> None:
    """Write a NetCDF file, replacing any existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    dataset.to_netcdf(path)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def record_case(
    name: str,
    case: dict,
    bands: tuple[np.ndarray, np.ndarray],
    *,
    uvspec_path: Path,
    sol_reference_path: Path,
    arts_path: Path,
    out: Path,
) -> dict[str, object]:
    """Build and write the table and 10 ppm spectrum of one case."""
    wavelength, fwhm = bands
    backgrounds = case["backgrounds"]
    model = mf.UnitAbsorption(
        gas="CH4",
        scenario=SCENARIO,
        lat=case["lat"],
        lon=case["lon"],
        time_utc=np.datetime64(case["time_utc"]),
        wmin=WMIN,
        wmax=WMAX,
        vza=case["vza"],
        cw=wavelength,
        fwhm=fwhm,
        rho=1.0,
        elevation=case["elevation"],
        altitude=case["altitude"],
        uvspec_path=uvspec_path,
        sol_reference_path=sol_reference_path,
        arts_path=arts_path,
        xch4_bg=backgrounds["CH4"],
        xco2_bg=backgrounds["CO2"],
        xh2o_bg=backgrounds["H2O"],
    )
    sza = float(model.sza)
    lut = build_slant_absorption_lut(model, NODES, layer_height=LAYER_HEIGHT)
    write(lut, out / f"{name}_lut.nc")
    target = model.compute_unit_absorption_spectrum(enhancement=10.0)
    target = target[["unit_abs_spect"]].assign_coords(
        fwhm=(("wavelength",), fwhm, {"units": "nm"})
    )
    target = target.reset_coords(
        [c for c in target.coords if c not in ("wavelength", "fwhm")], drop=True
    )
    write(target, out / f"{name}_target_10ppm.nc")
    return {
        **case,
        "gas": "CH4",
        "scenario": SCENARIO,
        "wmin": WMIN,
        "wmax": WMAX,
        "rho": 1.0,
        "sza": sza,
        "layer_height": LAYER_HEIGHT,
        "nodes_ppm": NODES.tolist(),
        "n_bands": int(wavelength.size),
        "band_dtype": str(wavelength.dtype),
    }


def record_profiles(
    uvspec_path: Path, sol_reference_path: Path, arts_path: Path
) -> dict[str, object]:
    """Return the lxml-parsed ARTS profiles and the reference column averages."""
    model = mf.UnitAbsorption(
        gas="CH4",
        scenario=SCENARIO,
        lat=0.0,
        lon=0.0,
        time_utc=np.datetime64("2025-06-21T12:00:00"),
        wmin=WMIN,
        wmax=WMAX,
        vza=0.0,
        cw=np.array([1500.0, 2300.0]),
        fwhm=np.array([8.0, 8.0]),
        rho=1.0,
        elevation=0.0,
        altitude=1000.0,
        xch4_bg=1.9,
        uvspec_path=uvspec_path,
        sol_reference_path=sol_reference_path,
        arts_path=arts_path,
    )
    profiles: dict[str, object] = {}
    for gas in PROFILE_GASES:
        z, c, p, t = model.read_standard_profile(gas)
        profiles[gas] = {
            "z": z.tolist(),
            "c": c.tolist(),
            "p": p.tolist(),
            "t": t.tolist(),
            "xgas_ref_ppmv": float(
                model.compute_column_averaged_dry_air_mole_fraction(gas, units="ppmv")
            ),
        }
    return profiles


def main() -> None:
    """Record the references from the command-line inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvspec-path", type=Path, required=True)
    parser.add_argument("--sol-reference-path", type=Path, required=True)
    parser.add_argument("--arts-path", type=Path, required=True)
    parser.add_argument("--emit-radiance", type=Path, required=True)
    parser.add_argument("--aviris4-radiance", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="tests/golden of the checkout"
    )
    parser.add_argument(
        "--aviris4-ch4-repo",
        type=Path,
        default=None,
        help="aviris4-ch4 checkout, to record its commit",
    )
    args = parser.parse_args()

    reference = args.out / "reference"
    paths = {
        "uvspec_path": args.uvspec_path,
        "sol_reference_path": args.sol_reference_path,
        "arts_path": args.arts_path,
    }
    bands = {
        "emit": emit_bands(args.emit_radiance),
        "aviris4": aviris4_bands(args.aviris4_radiance),
    }
    cases = {
        name: record_case(name, case, bands[case["sensor"]], out=reference, **paths)
        for name, case in CASES.items()
    }
    profiles = record_profiles(**paths)
    (reference / "profiles.json").write_text(json.dumps(profiles, indent=1) + "\n")

    arts_out = args.out / "inputs" / "arts" / SCENARIO
    arts_out.mkdir(parents=True, exist_ok=True)
    for name in ARTS_FILES:
        shutil.copyfile(
            args.arts_path / SCENARIO / f"{SCENARIO}.{name}.xml",
            arts_out / f"{SCENARIO}.{name}.xml",
        )

    record = {
        "built_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "aviris4_ch4_version": version("aviris4-ch4"),
        "aviris4_ch4_git": (
            git_state(args.aviris4_ch4_repo) if args.aviris4_ch4_repo else None
        ),
        "spectra_version": version("spectra"),
        "band_sources": {
            "emit": args.emit_radiance.name,
            "aviris4": args.aviris4_radiance.name,
        },
        "resources": {
            "tau_files": [
                f"UVSPEC_moltau_{SCENARIO}_{gas}_{WMIN}-{WMAX}nm-0.001nm.nc"
                for gas in ("all", *PROFILE_GASES)
            ],
            "solar_reference": (
                "hybrid_reference_spectrum_p005nm_resolution_c2021-03-04_with_unc.nc"
            ),
        },
        "cases": cases,
        "sha256": {
            name: sha256(reference / name)
            for name in [
                *(
                    f"{case}_{kind}.nc"
                    for case in CASES
                    for kind in ("lut", "target_10ppm")
                ),
                "profiles.json",
            ]
        },
    }
    (reference / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"record written to {reference / 'record.json'}")


if __name__ == "__main__":
    main()
