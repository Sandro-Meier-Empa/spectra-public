"""Exceptions shared by the contracts, plugins and regridders."""


class ContractError(ValueError):
    """A dataset violates one of the SPECTRA data contracts."""


class MissingExtraError(ImportError):
    """An optional dependency of a plugin is not installed."""

    def __init__(
        self, extra: str, package: str, cause: Exception | None = None
    ) -> None:
        """Describe the missing optional dependency.

        Args:
            extra: Name of the pip extra providing the dependency.
            package: Import name that failed.
            cause: Original import error.
        """
        self.extra = extra
        self.package = package
        detail = f" ({cause})" if cause is not None else ""
        super().__init__(
            f"'{package}' is not installed{detail}. "
            f'Install it with: pip install "spectra[{extra}]".'
        )


class PluginError(RuntimeError):
    """A registered plugin could not be loaded or does not satisfy its protocol."""


class RegridderUnavailableError(RuntimeError):
    """A regridder needs an external tool that is not available."""
