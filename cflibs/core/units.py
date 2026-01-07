"""
Unit conversion utilities for CF-LIBS.

Provides functions to convert between common units used in plasma physics
and spectroscopy.
"""

import numpy as np
from typing import Union

# ============================================================================
# Temperature Conversions
# ============================================================================


def convert_temperature(
    value: Union[float, np.ndarray], from_unit: str, to_unit: str
) -> Union[float, np.ndarray]:
    """
    Convert temperature between units.

    Parameters
    ----------
    value : float or array
        Temperature value(s) to convert
    from_unit : str
        Source unit: 'K', 'eV', 'C' (Celsius)
    to_unit : str
        Target unit: 'K', 'eV', 'C' (Celsius)

    Returns
    -------
    float or array
        Converted temperature value(s)

    Examples
    --------
    >>> convert_temperature(10000, 'K', 'eV')
    0.8617333262
    >>> convert_temperature(1.0, 'eV', 'K')
    11604.5250061598
    """
    from cflibs.core.constants import KB_EV, EV_TO_K

    # Normalize to Kelvin
    if from_unit.upper() == "K":
        kelvin = value
    elif from_unit.upper() == "EV":
        kelvin = value * EV_TO_K
    elif from_unit.upper() == "C":
        kelvin = value + 273.15
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")

    # Convert from Kelvin
    if to_unit.upper() == "K":
        return kelvin
    elif to_unit.upper() == "EV":
        return kelvin * KB_EV
    elif to_unit.upper() == "C":
        return kelvin - 273.15
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")


# ============================================================================
# Density Conversions
# ============================================================================


def convert_density(
    value: Union[float, np.ndarray], from_unit: str, to_unit: str
) -> Union[float, np.ndarray]:
    """
    Convert number density between units.

    Parameters
    ----------
    value : float or array
        Number density value(s) to convert
    from_unit : str
        Source unit: 'm^-3', 'cm^-3', 'm^-3'
    to_unit : str
        Target unit: 'm^-3', 'cm^-3', 'm^-3'

    Returns
    -------
    float or array
        Converted density value(s)

    Examples
    --------
    >>> convert_density(1e17, 'cm^-3', 'm^-3')
    1e23
    >>> convert_density(1e23, 'm^-3', 'cm^-3')
    1e17
    """
    # Normalize to m^-3
    if from_unit.lower() in ["m^-3", "m-3", "m**-3"]:
        per_m3 = value
    elif from_unit.lower() in ["cm^-3", "cm-3", "cm**-3"]:
        per_m3 = value * 1e6  # 1 cm^-3 = 1e6 m^-3
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")

    # Convert from m^-3
    if to_unit.lower() in ["m^-3", "m-3", "m**-3"]:
        return per_m3
    elif to_unit.lower() in ["cm^-3", "cm-3", "cm**-3"]:
        return per_m3 / 1e6
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")


# ============================================================================
# Wavelength Conversions
# ============================================================================


def convert_wavelength(
    value: Union[float, np.ndarray], from_unit: str, to_unit: str
) -> Union[float, np.ndarray]:
    """
    Convert wavelength between units.

    Parameters
    ----------
    value : float or array
        Wavelength value(s) to convert
    from_unit : str
        Source unit: 'm', 'nm', 'um', 'A' (Angstrom), 'cm^-1' (wavenumber)
    to_unit : str
        Target unit: 'm', 'nm', 'um', 'A' (Angstrom), 'cm^-1' (wavenumber)

    Returns
    -------
    float or array
        Converted wavelength value(s)

    Examples
    --------
    >>> convert_wavelength(500, 'nm', 'm')
    5e-7
    >>> convert_wavelength(500, 'nm', 'cm^-1')
    20000.0
    """

    # Normalize to meters
    if from_unit.lower() == "m":
        meters = value
    elif from_unit.lower() == "nm":
        meters = value * 1e-9
    elif from_unit.lower() in ["um", "μm"]:
        meters = value * 1e-6
    elif from_unit.lower() in ["a", "angstrom", "ang"]:
        meters = value * 1e-10
    elif from_unit.lower() in ["cm^-1", "cm-1", "wavenumber"]:
        # Convert wavenumber to wavelength
        meters = 1.0 / (value * 100.0)  # cm^-1 to m^-1, then to m
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")

    # Convert from meters
    if to_unit.lower() == "m":
        return meters
    elif to_unit.lower() == "nm":
        return meters * 1e9
    elif to_unit.lower() in ["um", "μm"]:
        return meters * 1e6
    elif to_unit.lower() in ["a", "angstrom", "ang"]:
        return meters * 1e10
    elif to_unit.lower() in ["cm^-1", "cm-1", "wavenumber"]:
        # Convert wavelength to wavenumber
        return 1.0 / (meters * 100.0)  # m to m^-1, then to cm^-1
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")


# ============================================================================
# Energy Conversions
# ============================================================================


def convert_energy(
    value: Union[float, np.ndarray], from_unit: str, to_unit: str
) -> Union[float, np.ndarray]:
    """
    Convert energy between units.

    Parameters
    ----------
    value : float or array
        Energy value(s) to convert
    from_unit : str
        Source unit: 'J', 'eV', 'cm^-1' (wavenumber)
    to_unit : str
        Target unit: 'J', 'eV', 'cm^-1' (wavenumber)

    Returns
    -------
    float or array
        Converted energy value(s)
    """
    from cflibs.core.constants import EV_TO_J, CM_TO_EV

    # Normalize to Joules
    if from_unit.lower() == "j":
        joules = value
    elif from_unit.lower() == "ev":
        joules = value * EV_TO_J
    elif from_unit.lower() in ["cm^-1", "cm-1", "wavenumber"]:
        # Convert wavenumber to eV, then to J
        ev = value * CM_TO_EV
        joules = ev * EV_TO_J
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")

    # Convert from Joules
    if to_unit.lower() == "j":
        return joules
    elif to_unit.lower() == "ev":
        return joules / EV_TO_J
    elif to_unit.lower() in ["cm^-1", "cm-1", "wavenumber"]:
        ev = joules / EV_TO_J
        return ev / CM_TO_EV
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")
