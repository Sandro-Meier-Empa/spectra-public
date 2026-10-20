"""Exceptions shared by SPECTRA."""


class ContractError(ValueError):
    """A dataset, file or run violates one of the SPECTRA contracts."""


__all__ = ["ContractError"]
