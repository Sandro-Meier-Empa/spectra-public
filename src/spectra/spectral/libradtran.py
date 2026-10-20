"""libRadtran/ARTS forward model for unit absorption spectra.

Ported from ``aviris4_ch4.mf.UnitAbsorption`` (aviris4-ch4 commit 6d4ed8b)
without changing the arithmetic, so tables built here are bit-identical to the
recorded references in ``tests/golden``. The model combines libRadtran
molecular optical depths, ARTS standard-atmosphere profiles and the TSIS-1
hybrid solar reference spectrum in a simple Lambertian radiance model and
convolves it with Gaussian slit functions.

Differences to the original:

- every resource file is given explicitly by role (:class:`LibRadtranArtsFiles`);
  nothing is derived from directory layouts or file-name patterns;
- the solar zenith angle is taken from the scene instead of an ephemeris;
- the resource arrays are loaded once per run and shared between models,
  and pure intermediate results are memoised per model;
- surface elevations below the lowest model level are clamped to 0 m, where
  the original selected the top layer as surface layer.
"""

from __future__ import annotations

import logging
import warnings
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import scipy.special
import ucat
import xarray as xr

import spectra
from spectra.contracts.errors import ContractError
from spectra.geometry.amf import effective_layer_height, path_amf
from spectra.spectral.lut import build_slant_absorption_lut
from spectra.types import RadianceLeg, SpectralResponse, ViewingGeometry

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Gases whose background the model rescales; others are kept at the reference.
BACKGROUND_GASES = ("CH4", "CO2", "H2O")
UNIT_ABS_SPECT_UNITS = "ppm-1 m-1"

# Fraction of a band's slit function that must fall inside the modelled spectral
# window. Below the minimum the convolution underflows to zero radiance.
_MIN_SLIT_COVERAGE = 1e-2
_FULL_SLIT_COVERAGE = 0.99


@dataclass(frozen=True)
class LibRadtranArtsFiles:
    """Resource files of the libRadtran/ARTS forward model, given by role.

    Attributes:
        solar_reference: TSIS-1 hybrid solar reference spectrum (NetCDF with
            ``Vacuum Wavelength`` and ``SSI``).
        tau: libRadtran molecular optical depths: ``all`` (every gas) and one
            file per gas whose background is set and for the embedded gas.
        profiles: ARTS standard-atmosphere profiles: ``z`` (altitude), ``t``
            (temperature) and one file per gas whose background is set.
    """

    solar_reference: Path
    tau: Mapping[str, Path]
    profiles: Mapping[str, Path]

    def __post_init__(self) -> None:
        """Normalise the paths and upper-case the gas keys."""
        object.__setattr__(self, "solar_reference", Path(self.solar_reference))
        tau = {_role(key): Path(value) for key, value in self.tau.items()}
        profiles = {_role(key): Path(value) for key, value in self.profiles.items()}
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "profiles", profiles)

    def required(self, gas: str, backgrounds: Mapping[str, float]) -> dict[str, Path]:
        """Return every file the model needs for ``gas`` and ``backgrounds``.

        Raises:
            ContractError: If a role is missing.
        """
        gas = gas.upper()
        gases = sorted({gas, *backgrounds})
        needed: dict[str, Path | None] = {
            "solar_reference": self.solar_reference,
            "tau.all": self.tau.get("all"),
            "profiles.z": self.profiles.get("z"),
            "profiles.t": self.profiles.get("t"),
        }
        for name in gases:
            needed[f"tau.{name}"] = self.tau.get(name)
            needed[f"profiles.{name}"] = self.profiles.get(name)
        missing = [role for role, path in needed.items() if path is None]
        if missing:
            raise ContractError(
                f"forward_model lacks the resource files {missing} needed for "
                f"{gas} with backgrounds {sorted(backgrounds)}."
            )
        return {role: path for role, path in needed.items() if path is not None}


def _role(key: str) -> str:
    """Return a resource role: gas names upper case, ``all``/``z``/``t`` lower."""
    lowered = str(key).lower()
    return lowered if lowered in ("all", "z", "t") else str(key).upper()


@dataclass(frozen=True)
class LibRadtranArtsConfig:
    """Settings of the libRadtran/ARTS forward model.

    Attributes:
        files: Resource files by role.
        backgrounds: Background column-averaged dry-air mole fractions in ppm
            by gas (CH4, CO2, H2O); the embedded gas must be present.
        scenario: Label of the atmospheric scenario, recorded as provenance.
        rho: Lambertian surface albedo of the radiance model; it cancels in the
            unit absorption spectrum.
        wmin: Lower bound of the modelled window in nm; defaults to the
            coverage of the ``all`` optical-depth file.
        wmax: Upper bound of the modelled window in nm; same default.
        enhancement_min_ppm: Smallest absorption-table node.
        enhancement_max_ppm: Largest absorption-table node.
        n_enhancements: Number of log-spaced nodes.
        layer_height_m: Nominal lowest-layer height of the model.
    """

    files: LibRadtranArtsFiles
    backgrounds: Mapping[str, float]
    scenario: str | None = None
    rho: float = 1.0
    wmin: float | None = None
    wmax: float | None = None
    enhancement_min_ppm: float = 0.01
    enhancement_max_ppm: float = 200.0
    n_enhancements: int = 12
    layer_height_m: float = 1000.0

    def __post_init__(self) -> None:
        """Validate backgrounds, window and table nodes."""
        backgrounds = {}
        for gas, value in self.backgrounds.items():
            name = str(gas).upper()
            if name not in BACKGROUND_GASES:
                raise ContractError(
                    f"Background of {name} is not supported; choose from "
                    f"{BACKGROUND_GASES}."
                )
            if value is None:
                continue
            if not np.isfinite(value) or value <= 0:
                raise ContractError(f"Background of {name} must be positive.")
            backgrounds[name] = float(value)
        object.__setattr__(self, "backgrounds", backgrounds)
        if (self.wmin is None) != (self.wmax is None):
            raise ContractError("Set both wmin and wmax, or neither.")
        if self.wmin is not None and not self.wmin < cast(float, self.wmax):
            raise ContractError("wmin must be smaller than wmax.")
        if not 0 < self.enhancement_min_ppm < self.enhancement_max_ppm:
            raise ContractError("Need 0 < enhancement_min_ppm < enhancement_max_ppm.")
        if self.n_enhancements < 2:
            raise ContractError("n_enhancements must be at least 2.")
        if self.layer_height_m <= 0:
            raise ContractError("layer_height_m must be positive.")

    @property
    def enhancements_ppm(self) -> tuple[float, ...]:
        """Return the log-spaced absorption-table nodes in ppm."""
        return tuple(
            np.logspace(
                np.log10(self.enhancement_min_ppm),
                np.log10(self.enhancement_max_ppm),
                self.n_enhancements,
            )
        )


@dataclass(eq=False)
class RadiativeTransferResources:
    """Resource arrays of one run, loaded lazily and shared between models.

    Cached arrays are marked read-only; models copy them before mutating, as
    the original code always worked on fresh copies.

    Args:
        files: Resource files by role.
        wmin: Lower bound of the modelled window in nm, or ``None`` for the
            coverage of the ``all`` optical-depth file.
        wmax: Upper bound of the modelled window in nm, or ``None``.
    """

    files: LibRadtranArtsFiles
    wmin: float | None = None
    wmax: float | None = None
    _cache: dict[tuple[Any, ...], Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Resolve the window from the optical-depth file when not given."""
        if self.wmin is None or self.wmax is None:
            path = self.files.tau.get("all")
            if path is None:
                raise ContractError("forward_model.tau lacks the 'all' file.")
            with _open_tau(path) as data:
                lower = float(np.asarray(data["wvlmin"].values).reshape(-1)[0])
                upper = float(np.asarray(data["wvlmax"].values).reshape(-1)[0])
            self.wmin = _window_bound(lower if self.wmin is None else self.wmin)
            self.wmax = _window_bound(upper if self.wmax is None else self.wmax)
        else:
            self.wmin = _window_bound(self.wmin)
            self.wmax = _window_bound(self.wmax)

    @property
    def window(self) -> tuple[int | float, int | float]:
        """Return the modelled window ``(wmin, wmax)`` in nanometres."""
        if self.wmin is None or self.wmax is None:  # pragma: no cover - set in init
            raise ContractError("The modelled window is not resolved.")
        return self.wmin, self.wmax

    def _cached(self, key: tuple[Hashable, ...], build: Callable[[], T]) -> T:
        if key not in self._cache:
            value = build()
            for item in value if isinstance(value, tuple) else (value,):
                if isinstance(item, np.ndarray):
                    item.flags.writeable = False
            self._cache[key] = value
        return self._cache[key]

    def solar(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the solar reference wavelengths and irradiance in the window."""

        def build() -> tuple[np.ndarray, np.ndarray]:
            wmin, wmax = self.window
            with xr.open_dataset(self.files.solar_reference) as sol:
                sol = sol.where(
                    (wmin <= sol["Vacuum Wavelength"])
                    & (sol["Vacuum Wavelength"] <= wmax),
                    drop=True,
                )
                w0 = sol["Vacuum Wavelength"].values  # wavelength in vacuum
                solar_irradiance = sol["SSI"].values
            return w0, solar_irradiance

        return self._cached(("solar",), build)

    def profile(
        self, gas: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return altitude, mixing ratio, pressure and temperature of ``gas``.

        The levels are cut to the part also present in the libRadtran
        optical-depth files, as in the original.
        """
        gas = gas.upper()

        def build() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            gas_file = self._profile_file(gas)
            p = _parse_pressure(gas_file)
            c = _parse_tensor(gas_file)
            z = _parse_tensor(self._profile_file("z"))
            temperature = _parse_tensor(self._profile_file("t"))
            return z[1:], c[1:-1], p[1:-1], temperature[1:-1]

        return self._cached(("profile", gas), build)

    def _profile_file(self, role: str) -> Path:
        path = self.files.profiles.get(role)
        if path is None:
            raise ContractError(f"forward_model.profiles lacks the {role!r} file.")
        return path

    def column_average(self, gas: str, units: str = "ppmv") -> np.ndarray:
        """Return the column-averaged dry-air mole fraction of the profile."""
        gas = gas.upper()

        def build() -> np.ndarray:
            z, volume_mixing_ratio, p, temperature = self.profile(gas)
            concentration = ucat.convert_points(
                cast(Any, volume_mixing_ratio),
                "mol/mol",
                "mol/m3",
                p=cast(Any, p),
                T=cast(Any, temperature),
                molar_mass=gas,
            )
            x = ucat.convert_columns(
                np.sum(np.diff(z) * concentration),
                "mol/m2",
                units,
                p_surface=p[0],
                molar_mass=gas,
            )
            return np.asarray(x)

        return self._cached(("column_average", gas, units), build)

    def tau(self, role: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the level altitudes in km and the optical depths of a file."""
        role = _role(role)

        def build() -> tuple[np.ndarray, np.ndarray]:
            path = self.files.tau.get(role)
            if path is None:
                raise ContractError(f"forward_model.tau lacks the {role!r} file.")
            with _open_tau(path) as data:
                return data["z"].values.copy(), data["tau"].values.copy()

        return self._cached(("tau", role), build)

    def background_tau(self, gas: str, backgrounds: Mapping[str, float]) -> np.ndarray:
        """Return the optical depths with the gas backgrounds rescaled.

        Only CH4, CO2 and H2O are rescaled, as in the original.
        """
        gas = gas.upper()
        key = ("background_tau", gas, tuple(sorted(backgrounds.items())))

        def build() -> np.ndarray:
            if backgrounds.get(gas) is None:
                raise ContractError(f"Missing background concentration for {gas}.")
            tau = self.tau("all")[1].copy()
            # subtract and add scaled XCO2 and XCH4 and XH2O
            for name in BACKGROUND_GASES:
                if backgrounds.get(name) is not None:
                    tau_gas = self.tau(name)[1]
                    xbg_ref = self.column_average(name, units="ppmv")
                    # subtract reference trace gas
                    tau -= tau_gas
                    # add scaled trace gas
                    tau += tau_gas / xbg_ref * backgrounds[name]
            return tau

        return self._cached(key, build)

    def scaled_gas_tau(
        self, gas: str, xgas: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return level altitudes in m and the gas optical depths at ``xgas``."""
        gas = gas.upper()

        def build() -> tuple[np.ndarray, np.ndarray]:
            z_km, tau = self.tau(gas)
            tau = tau.copy()
            z = 1e3 * z_km.copy()  # convert to m
            if xgas is not None:
                xgas_ref = self.column_average(gas, units="ppmv")
                tau = tau / xgas_ref * xgas
            return z, tau

        return self._cached(("scaled_gas_tau", gas, xgas), build)


def _open_tau(path: Path) -> xr.Dataset:
    """Open a libRadtran optical-depth file, silencing its duplicate-dim warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Duplicate dimension names present: dimensions .* "
                r"appear more than once"
            ),
            category=UserWarning,
        )
        return xr.open_dataset(path)


def _parse_tensor(path: Path) -> np.ndarray:
    """Read the tensor payload of an ARTS XML profile file."""
    tensor = ElementTree.parse(path).getroot().find("GriddedField3/Tensor3")
    if tensor is None or tensor.text is None:
        raise ContractError(f"No GriddedField3/Tensor3 in ARTS profile {path}.")
    return np.array([float(v) for v in tensor.text.split() if v.strip()])


def _parse_pressure(path: Path) -> np.ndarray:
    """Read the pressure vector of an ARTS XML profile file."""
    for element in ElementTree.parse(path).getroot().findall("GriddedField3/Vector"):
        if element.attrib.get("name") == "Pressure" and element.text is not None:
            return np.array([float(v) for v in element.text.split() if v.strip()])
    raise ContractError(f"Pressure vector not found in {path}.")


def _window_bound(value: float) -> int | float:
    """Return a window bound as int when integral, as the original files spell it."""
    number = float(value)
    return int(number) if number.is_integer() else number


class UnitAbsorption:
    """Unit absorption spectrum of a trace gas for one observation geometry.

    Args:
        gas: Gas name, one of :data:`BACKGROUND_GASES`.
        sza: Solar zenith angle in degrees.
        vza: Viewing zenith angle in degrees.
        cw: Sensor centre wavelengths in nanometres.
        fwhm: Sensor full widths at half maximum in nanometres.
        elevation: Surface elevation in metres; negative values are clamped
            to 0 m with a warning.
        altitude: Sensor altitude in metres; any value above the profile top
            (95 km) acts as top of atmosphere.
        resources: Shared resource arrays.
        backgrounds: Background column-averaged dry-air mole fractions in ppm.
        rho: Lambertian surface albedo of the radiance model.
        scenario: Scenario label recorded in the absorption table.
        warn_truncated: Whether to warn about bands whose slit function the
            modelled window truncates.
    """

    def __init__(
        self,
        gas: str,
        *,
        sza: float,
        vza: float,
        cw: np.ndarray,
        fwhm: np.ndarray | None,
        elevation: float,
        altitude: float,
        resources: RadiativeTransferResources,
        backgrounds: Mapping[str, float],
        rho: float = 1.0,
        scenario: str | None = None,
        warn_truncated: bool = True,
    ) -> None:
        """Validate the geometry and load the solar reference spectrum."""
        self.gas = gas.upper()
        if self.gas not in BACKGROUND_GASES:
            raise ContractError(
                f"Gas {self.gas} is not supported; choose from {BACKGROUND_GASES}."
            )
        for name, value in (
            ("sza", sza),
            ("vza", vza),
            ("elevation", elevation),
            ("altitude", altitude),
        ):
            if not np.isfinite(value):
                raise ContractError(f"Forward-model geometry {name} is not finite.")
        if not 0 <= sza < 90 or not 0 <= vza < 90:
            raise ContractError("sza and vza must lie in [0, 90) degrees.")
        self.elevation_input = float(elevation)
        if elevation < 0:
            logger.warning(
                "Surface elevation %.1f m lies below the lowest model level; "
                "the forward model uses 0 m.",
                elevation,
            )
            elevation = 0.0
        self._sza = float(sza)
        self.vza = vza
        self.cw = cw
        self.fwhm = fwhm
        self.rho = rho
        self.elevation = elevation
        self.altitude = altitude
        self.resources = resources
        self.scenario = scenario
        self.wmin, self.wmax = resources.window
        self.bg_dict = {gas: backgrounds.get(gas) for gas in BACKGROUND_GASES}
        if self.bg_dict[self.gas] is None:
            raise ContractError(f"Missing background concentration for {self.gas}.")
        self.w0, self.E0 = resources.solar()
        self._optical_depths: tuple[np.ndarray, np.ndarray] | None = None
        self._radiance: dict[tuple[float, str], np.ndarray] = {}
        self._validate_spectral_coverage(warn_truncated)

    @property
    def elevation_clamped(self) -> bool:
        """Whether the input elevation was clamped to 0 m."""
        return self.elevation_input < 0

    def _validate_spectral_coverage(self, warn_truncated: bool) -> None:
        """Check that every sensor band is supported by the modelled window.

        The solar reference spectrum and the libRadtran optical depths only
        cover ``wmin`` to ``wmax``. A band whose slit function falls outside
        that window convolves to zero radiance, which turns the unit absorption
        spectrum into ``NaN`` through ``log(0)``.

        Args:
            warn_truncated: Whether to warn about bands only partly inside the
                window.

        Raises:
            ContractError: If any band lies essentially outside the window.
        """
        cw = np.atleast_1d(np.asarray(self.cw, dtype=float))

        if self.fwhm is None:
            coverage = ((cw >= self.wmin) & (cw <= self.wmax)).astype(float)
        else:
            fwhm = np.broadcast_to(
                np.atleast_1d(np.asarray(self.fwhm, dtype=float)), cw.shape
            )
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # A zero width degenerates to a point sample inside the window.
            scale = np.maximum(sigma, np.finfo(float).tiny) * np.sqrt(2.0)
            with np.errstate(over="ignore"):
                coverage = 0.5 * (
                    scipy.special.erf((self.wmax - cw) / scale)
                    - scipy.special.erf((self.wmin - cw) / scale)
                )

        unsupported = coverage < _MIN_SLIT_COVERAGE
        if np.any(unsupported):
            bands = cw[unsupported]
            raise ContractError(
                f"{bands.size} of {cw.size} sensor bands "
                f"({bands.min():.1f}-{bands.max():.1f} nm) lie outside the modelled "
                f"{self.wmin}-{self.wmax} nm window. Their radiance convolves to "
                "zero and the unit absorption spectrum would be NaN."
            )

        truncated = coverage < _FULL_SLIT_COVERAGE
        if warn_truncated and np.any(truncated):
            bands = cw[truncated]
            logger.warning(
                "%d sensor bands at %s nm sit on the edge of the %s-%s nm window; "
                "their slit function is truncated and the unit absorption "
                "spectrum is biased there.",
                bands.size,
                ", ".join(f"{band:.1f}" for band in bands),
                self.wmin,
                self.wmax,
            )

    def _require_positive_radiance(self, *spectra: np.ndarray) -> None:
        """Reject radiance spectra that make the logarithm undefined.

        Raises:
            ContractError: If any band is non-finite or not strictly positive.
        """
        cw = np.atleast_1d(np.asarray(self.cw, dtype=float))
        invalid = np.zeros(cw.shape, dtype=bool)
        for spectrum in spectra:
            values = np.squeeze(np.asarray(spectrum, dtype=float))
            invalid |= ~np.isfinite(values) | (values <= 0.0)

        if np.any(invalid):
            bands = cw[invalid]
            preview = ", ".join(f"{value:.1f}" for value in bands[:5])
            suffix = "..." if bands.size > 5 else ""
            raise ContractError(
                f"Radiance is not strictly positive at {bands.size} of {cw.size} "
                f"bands [{preview}{suffix}] nm, so the unit absorption spectrum is "
                "undefined. Check that the bands lie inside the modelled "
                f"{self.wmin}-{self.wmax} nm window and that the optical depths are "
                "not saturated."
            )

    @property
    def sza(self) -> float:
        """Solar zenith angle in degrees."""
        return self._sza

    @property
    def mu0(self) -> float:
        """Cosine of solar zenith angle."""
        return np.cos(np.deg2rad(self.sza))

    @property
    def mu(self) -> float:
        """Cosine of viewing zenith angle."""
        return np.cos(np.deg2rad(self.vza))

    @property
    def surface_index(self) -> int:
        """Index corresponding to the surface layer based on elevation."""
        return -1 * (int(self.elevation // 1000.0) + 1)

    def compute_unit_absorption_spectrum(
        self,
        enhancement: float,
        layer_height: float = 1000.0,
        radiance_leg: RadianceLeg = "both",
    ) -> xr.Dataset:
        """Compute the unit absorption spectrum for the configured gas.

        The spectrum follows from the change in at-sensor radiance caused by a
        gas enhancement over the lowest model layer:
        ``s = -log(L / L0) / (dz * enhancement)``.

        Args:
            enhancement: Gas enhancement in ppmv in the lowest layer.
            layer_height: Lowest model-layer height in metres.
            radiance_leg: Radiance path segment intersecting the plume.

        Returns:
            Dataset with ``unit_abs_spect(wavelength)`` in ppm-1 m-1.

        Raises:
            ContractError: If the modelled radiance is not strictly positive.
        """
        scaling_factor = self.enhancement_to_scaling_factor(enhancement)

        radiance_reference = self.compute_radiance_spectrum(1.0, radiance_leg)
        radiance_enhanced = self.compute_radiance_spectrum(scaling_factor, radiance_leg)
        self._require_positive_radiance(radiance_reference, radiance_enhanced)

        # ppmv * m
        height = effective_layer_height(self.elevation, layer_height)

        alpha = enhancement * height
        s = (np.log(radiance_reference) - np.log(radiance_enhanced)) / alpha

        # assume negative values to make s consistent with previous papers
        return self.create_dataset(
            np.squeeze(-s),
            enhancement=enhancement,
            layer_height=layer_height,
            radiance_leg=radiance_leg,
        )

    def enhancement_to_scaling_factor(self, enhancement: float) -> float:
        """Convert a gas enhancement to a lowest-layer concentration scale factor."""
        # Read scenario profile and scale with new background
        _z, volume_mixing_ratio, _p, _temperature = self.resources.profile(self.gas)
        xgas_ref = self.resources.column_average(self.gas)

        # mol/mol -> ppmv
        x_surface_ppmv = (
            1e6
            * volume_mixing_ratio[::-1][self.surface_index]
            / xgas_ref
            * self.bg_dict[self.gas]
        )

        return (x_surface_ppmv + enhancement) / x_surface_ppmv

    def compute_radiance_spectrum(
        self, scaling_factor: float, radiance_leg: RadianceLeg = "both"
    ) -> np.ndarray:
        """Compute at-sensor radiance after scaling the gas optical depth."""
        key = (float(scaling_factor), radiance_leg)
        if key in self._radiance:
            return self._radiance[key]
        amf_plume = path_amf(self.sza, self.vza, radiance_leg)
        amf_bg = path_amf(self.sza, self.vza, "both")

        tau_bg, tau_gas = self._scaled_optical_depths()

        # Apply the AMF to the background and GAS optical depths
        tau = amf_plume * tau_gas * scaling_factor + amf_bg * tau_bg

        radiance = self._compute_radiance(self.E0, tau)

        # Apply instrument convolution if needed
        if self.fwhm is not None:
            radiance = convolve(x=self.w0, y=radiance, cw=self.cw, spf=self.fwhm)

        self._radiance[key] = radiance
        return radiance

    def _scaled_optical_depths(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the AMF-scaled background and lowest-layer gas optical depths.

        Both depend only on the gas, backgrounds, elevation and altitude, so
        they are computed once per model.
        """
        if self._optical_depths is None:
            tau_bg = self.resources.background_tau(self.gas, self._backgrounds)
            tau_bg = tau_bg[:, :, 0, 0].copy()
            z, tau = self.resources.scaled_gas_tau(self.gas, self.bg_dict[self.gas])
            tau_gas = tau[self.surface_index, :, 0, 0].copy()

            # Scale background optical depth
            scaling = self._calculate_amf_scaling(z)
            self._optical_depths = self._apply_amf_scaling_to_tau(
                tau_bg, tau_gas, scaling
            )
        return self._optical_depths

    @property
    def _backgrounds(self) -> dict[str, float]:
        return {gas: value for gas, value in self.bg_dict.items() if value is not None}

    def _calculate_amf_scaling(self, z: np.ndarray) -> np.ndarray:
        """Calculate scaling factors for the optical depths."""
        amfs = self.calculate_geometric_1d_layer_amfs(
            z[::-1], sza=0.0, vza=0.0, flip=True, sum_amfs=True
        )
        assert isinstance(amfs, np.ndarray)
        return amfs[1:] / 2.0  # divide by 2 to get amf at layer mid point

    def _apply_amf_scaling_to_tau(
        self, tau_bg: np.ndarray, tau_gas: np.ndarray, scaling: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply AMF scaling to the background and gas optical depths."""
        # remove GAS in lowest layer from background
        tau_bg[self.surface_index] -= tau_gas

        # apply normalized AMF to background optical depth and vertically integrate
        tau_bg = (tau_bg * scaling[:, np.newaxis]).sum(0)

        # scale lowest layer with surface elevation correction
        tau_gas *= scaling[self.surface_index]

        return tau_bg, tau_gas

    def _compute_radiance(
        self, solar_irradiance: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """Compute the radiance using the scaled optical depth."""
        radiance_unattenuated = self.rho * self.mu0 / np.pi * solar_irradiance
        return radiance_unattenuated * np.exp(-tau)

    def calculate_geometric_1d_layer_amfs(
        self,
        levels: np.ndarray,
        sza: float,
        vza: float,
        flip: bool = True,
        sum_amfs: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Compute layer AMFs split above and below the sensor.

        Args:
            levels: Vertical layer lower boundaries in metres.
            sza: Solar zenith angle in degrees.
            vza: Viewing zenith angle in degrees.
            flip: Reverse layer order before returning weights.
            sum_amfs: Return the summed AMF instead of separate arrays.

        Returns:
            Summed AMF array, or separate below-sensor and above-sensor AMFs.
        """
        elevation = np.atleast_2d(self.elevation)
        altitude = np.atleast_2d(self.altitude)

        mask = np.isfinite(elevation) & np.isfinite(altitude)

        # compute index and fraction
        i0 = np.interp(elevation, levels, np.arange(levels.size))
        i1 = np.interp(altitude, levels, np.arange(levels.size))

        i0, f0 = i0.astype(int), i0 % 1
        i1, f1 = i1.astype(int), i1 % 1

        # compute weights
        weights_below = np.zeros((*elevation.shape, levels.size))
        weights_above = np.zeros((*elevation.shape, levels.size))

        for i, j in np.ndindex(i0.shape):
            if mask[i, j]:
                weights_below[i, j, i0[i, j]] = 1.0 - f0[i, j]
                weights_below[i, j, i0[i, j] + 1 : i1[i, j]] = 1.0
                weights_below[i, j, i1[i, j]] = f1[i, j]

                weights_above[i, j, i1[i, j]] = 1.0 - f1[i, j]
                weights_above[i, j, i1[i, j] + 1 :] = 1.0

        if flip:
            weights_below = weights_below[:, :, ::-1]
            weights_above = weights_above[:, :, ::-1]

        # geometric AMF => mu and mu0 are 0
        mu0 = np.cos(np.deg2rad(sza))
        mu = np.cos(np.deg2rad(vza))

        amf_below = (1.0 / mu0 + 1.0 / mu) * weights_below
        amf_above = (1.0 / mu0) * weights_above

        if sum_amfs:
            return np.squeeze(amf_below + amf_above)
        return np.squeeze(amf_below), np.squeeze(amf_above)

    def create_dataset(
        self,
        unit_absorption_spectrum: np.ndarray,
        *,
        enhancement: float,
        layer_height: float,
        radiance_leg: RadianceLeg,
    ) -> xr.Dataset:
        """Return the unit absorption spectrum as a dataset."""
        coords: dict[str, Any] = {
            "wavelength": (["wavelength"], self.cw, {"units": "nm"})
        }
        if self.fwhm is not None:
            coords["fwhm"] = (["wavelength"], self.fwhm, {"units": "nm"})
        return xr.Dataset(
            data_vars={
                "unit_abs_spect": (
                    ["wavelength"],
                    unit_absorption_spectrum,
                    {
                        "units": UNIT_ABS_SPECT_UNITS,
                        "long_name": (
                            "unit absorption spectrum per vertical path amount"
                        ),
                    },
                )
            },
            coords=coords,
            attrs={
                "description": f"Unit absorption spectrum for {self.gas}",
                "gas": self.gas,
                "enhancement_ppm": float(enhancement),
                "layer_height_m": float(layer_height),
                "radiance_leg": radiance_leg,
                "sza": self.sza,
                "vza": float(self.vza),
                "surface_elevation_m": float(self.elevation),
                "sensor_altitude_m": float(self.altitude),
                "wmin": self.wmin,
                "wmax": self.wmax,
            },
        )


def convolve(
    x: np.ndarray, y: np.ndarray, cw: np.ndarray, spf: np.ndarray
) -> np.ndarray:
    """Convolve a spectrum with Gaussian slit functions.

    Args:
        x: Input spectral grid in nanometres.
        y: Input spectral values on ``x``.
        cw: Output centre wavelengths.
        spf: Full widths at half maximum of the slit functions.

    Returns:
        Convolved spectral values on ``cw``.
    """
    x, y, cw, spf = map(np.asarray, (x, y, cw, spf))
    fwhm2sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    values = np.zeros(cw.size)
    for i in range(cw.size):
        g = gauss(x, cw[i], spf[i] * fwhm2sigma)
        values[i] = np.dot(y, g)
    return values


def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate a Gaussian normalised by the sampling interval of ``x``."""
    dx = sampling_interval(x)
    tau = np.sqrt(2.0 * np.pi)
    g = np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    g *= dx / (sigma * tau)
    return g


def sampling_interval(x: np.ndarray) -> np.ndarray:
    """Return sample intervals from adjacent cell centres."""
    x = np.asarray(x)
    c = 0.5 * (x[1:] + x[:-1])
    c0 = x[0] - (c[0] - x[0])
    cn = x[-1] + (x[-1] - c[-1])
    return np.diff(np.concatenate([[c0], c, [cn]]))


class LibRadtranArts:
    """Forward model satisfying :class:`spectra.protocols.ForwardModel`.

    Models are cached per gas, bands and geometry; the resource arrays are
    shared between them.

    Args:
        config: Resource files and model settings.
    """

    name = "libradtran_arts"

    def __init__(self, config: LibRadtranArtsConfig) -> None:
        """Store the settings; resources load on first use."""
        self.config = config
        self.resources = RadiativeTransferResources(
            config.files, wmin=config.wmin, wmax=config.wmax
        )
        self._models: dict[tuple[Any, ...], UnitAbsorption] = {}
        # Band sets already warned about: one warning per run, not per cluster.
        self._coverage_warned: set[tuple[bytes, bytes]] = set()

    @property
    def spectral_range_nm(self) -> tuple[float, float]:
        """Wavelength range the model covers; other bands pass unattenuated."""
        return (
            float(self.resources.window[0]),
            float(self.resources.window[1]),
        )

    @property
    def enhancements_ppm(self) -> tuple[float, ...]:
        """Absorption-table nodes in ppm."""
        return self.config.enhancements_ppm

    @property
    def layer_height_m(self) -> float:
        """Nominal lowest-layer height in metres."""
        return float(self.config.layer_height_m)

    def check(self, gas: str) -> None:
        """Validate that every resource needed for ``gas`` is configured and exists.

        Raises:
            ContractError: If a background or a resource file is missing.
        """
        gas = gas.upper()
        if gas not in BACKGROUND_GASES:
            raise ContractError(
                f"Gas {gas} is not supported; choose from {BACKGROUND_GASES}."
            )
        if gas not in self.config.backgrounds:
            raise ContractError(
                f"forward_model.backgrounds lacks the background of {gas}."
            )
        files = self.config.files.required(gas, self.config.backgrounds)
        missing = [
            f"{role}: {path}" for role, path in files.items() if not path.is_file()
        ]
        if missing:
            raise ContractError(f"Forward-model resource files not found: {missing}.")

    def model(
        self, response: SpectralResponse, geometry: ViewingGeometry, *, gas: str
    ) -> UnitAbsorption:
        """Return the cached forward model for one geometry."""
        gas = gas.upper()
        key = (
            gas,
            response.wavelength.tobytes(),
            response.fwhm.tobytes(),
            float(geometry.sza),
            float(geometry.vza),
            float(geometry.surface_elevation),
            float(geometry.sensor_altitude),
        )
        model = self._models.get(key)
        if model is None:
            bands = (response.wavelength.tobytes(), response.fwhm.tobytes())
            model = UnitAbsorption(
                gas,
                sza=float(geometry.sza),
                vza=float(geometry.vza),
                cw=response.wavelength,
                fwhm=response.fwhm,
                elevation=float(geometry.surface_elevation),
                altitude=float(geometry.sensor_altitude),
                resources=self.resources,
                backgrounds=self.config.backgrounds,
                rho=self.config.rho,
                scenario=self.config.scenario,
                warn_truncated=bands not in self._coverage_warned,
            )
            self._coverage_warned.add(bands)
            self._models[key] = model
        return model

    def lut(
        self, response: SpectralResponse, geometry: ViewingGeometry, *, gas: str
    ) -> xr.Dataset:
        """Return the absorption table of one geometry.

        Args:
            response: Band centres and widths inside :attr:`spectral_range_nm`.
            geometry: Observation geometry.
            gas: Gas name.

        Returns:
            Table from :func:`~spectra.spectral.lut.build_slant_absorption_lut`.
        """
        model = self.model(response, geometry, gas=gas)
        table = build_slant_absorption_lut(
            model, self.enhancements_ppm, layer_height=self.layer_height_m
        )
        if model.elevation_clamped:
            table.attrs["surface_elevation_clamped_from"] = model.elevation_input
        return table

    def provenance(self) -> dict[str, Any]:
        """Return the settings recorded as ``spectra_forward_model_*`` attributes."""
        files = self.config.files
        record: dict[str, Any] = {
            "name": self.name,
            "spectra_version": spectra.__version__,
            "wmin_nm": float(self.resources.window[0]),
            "wmax_nm": float(self.resources.window[1]),
            "rho": float(self.config.rho),
            "layer_height_m": self.layer_height_m,
            "enhancement_min_ppm": float(self.config.enhancement_min_ppm),
            "enhancement_max_ppm": float(self.config.enhancement_max_ppm),
            "n_enhancements": int(self.config.n_enhancements),
            "solar_reference": str(files.solar_reference),
        }
        if self.config.scenario is not None:
            record["scenario"] = self.config.scenario
        for gas, value in sorted(self.config.backgrounds.items()):
            record[f"background_{gas.lower()}_ppm"] = float(value)
        for role, path in sorted(files.tau.items()):
            record[f"tau_{role.lower()}"] = str(path)
        for role, path in sorted(files.profiles.items()):
            record[f"profile_{role.lower()}"] = str(path)
        return record


__all__ = [
    "BACKGROUND_GASES",
    "LibRadtranArts",
    "LibRadtranArtsConfig",
    "LibRadtranArtsFiles",
    "RadiativeTransferResources",
    "UnitAbsorption",
    "convolve",
]
