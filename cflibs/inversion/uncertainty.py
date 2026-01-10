"""
Uncertainty propagation utilities for CF-LIBS classical inversion.

This module provides functions for propagating uncertainties through the
CF-LIBS inversion pipeline using the `uncertainties` package, which
automatically tracks correlations via a computational graph.

Why uncertainties over alternatives?
-----------------------------------
- **Scipp**: Doesn't track correlations (critical for CF-LIBS where Boltzmann
  slope and intercept are correlated from the same regression)
- **AutoUncertainties**: No correlation tracking (x - x != 0 exactly)
- **uncertainties**: Automatic correlation tracking via computational graph

Usage
-----
```python
from uncertainties import correlated_values
from cflibs.inversion.uncertainty import (
    create_boltzmann_uncertainties,
    propagate_through_closure,
)

# From BoltzmannFitResult with covariance_matrix
slope_u, intercept_u = create_boltzmann_uncertainties(fit_result)

# Temperature with correlated uncertainty
T_eV_u = -1.0 / (KB_EV * slope_u)  # Uncertainty auto-propagated

# Through closure equation
concentrations_u = propagate_through_closure(intercepts_u, partition_funcs)
```

Dependencies
------------
Requires: `pip install uncertainties>=3.2.0`
Install with: `pip install cflibs[uncertainty]`

References
----------
- uncertainties: https://uncertainties-python-package.readthedocs.io/
- Lebigot, E. "uncertainties: a Python package for calculations with uncertainties"
"""

from typing import Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.uncertainty")

# Type checking imports
if TYPE_CHECKING:
    from uncertainties import UFloat

# Check for uncertainties package
try:
    from uncertainties import ufloat, correlated_values
    from uncertainties import umath

    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False
    ufloat = None
    correlated_values = None
    umath = None


def check_uncertainties_available() -> None:
    """Raise ImportError if uncertainties package is not available."""
    if not HAS_UNCERTAINTIES:
        raise ImportError(
            "uncertainties package required for uncertainty propagation. "
            "Install with: pip install uncertainties>=3.2.0 or pip install cflibs[uncertainty]"
        )


def create_boltzmann_uncertainties(
    slope: float,
    intercept: float,
    covariance_matrix: Optional[np.ndarray],
    slope_err: Optional[float] = None,
    intercept_err: Optional[float] = None,
) -> Tuple["UFloat", "UFloat"]:
    """
    Create correlated ufloat objects for Boltzmann fit slope and intercept.

    Parameters
    ----------
    slope : float
        Boltzmann plot slope = -1/(kB*T)
    intercept : float
        Boltzmann plot intercept (related to ln(C*F/U))
    covariance_matrix : np.ndarray, optional
        2x2 covariance matrix [[var(slope), cov], [cov, var(intercept)]].
        If provided, creates correlated uncertainties.
    slope_err : float, optional
        Fallback slope uncertainty if covariance_matrix is None
    intercept_err : float, optional
        Fallback intercept uncertainty if covariance_matrix is None

    Returns
    -------
    tuple[UFloat, UFloat]
        Correlated (slope_u, intercept_u) ufloat objects

    Raises
    ------
    ImportError
        If uncertainties package is not installed
    """
    check_uncertainties_available()

    if covariance_matrix is not None:
        # Create correlated values from covariance matrix
        slope_u, intercept_u = correlated_values([slope, intercept], covariance_matrix)
    else:
        # Fall back to independent uncertainties
        s_err = slope_err if slope_err is not None and np.isfinite(slope_err) else 0.0
        i_err = intercept_err if intercept_err is not None and np.isfinite(intercept_err) else 0.0
        slope_u = ufloat(slope, s_err)
        intercept_u = ufloat(intercept, i_err)

    return slope_u, intercept_u


def temperature_from_slope(slope_u: "UFloat") -> "UFloat":
    """
    Calculate temperature with uncertainty from Boltzmann slope.

    T = -1 / (kB * slope)

    Parameters
    ----------
    slope_u : UFloat
        Boltzmann slope with uncertainty

    Returns
    -------
    UFloat
        Temperature in Kelvin with propagated uncertainty
    """
    check_uncertainties_available()
    from cflibs.core.constants import EV_TO_K

    # Temperature in eV
    T_eV_u = -1.0 / (KB_EV * slope_u)
    # Convert to Kelvin
    T_K_u = T_eV_u * EV_TO_K

    return T_K_u


def saha_factor_with_uncertainty(
    T_eV_u: "UFloat",
    n_e: float,
    ionization_potential_eV: float,
    U_I: float,
    U_II: float,
    saha_const: float,
) -> "UFloat":
    """
    Calculate Saha factor with uncertainty propagation.

    S = (SAHA_CONST / n_e) * T^1.5 * exp(-IP/T) * 2 * (U_II/U_I)

    Note: Currently n_e is treated as exact (no uncertainty).
    For full uncertainty propagation, n_e would need to be a ufloat.

    Parameters
    ----------
    T_eV_u : UFloat
        Temperature in eV with uncertainty
    n_e : float
        Electron density in cm^-3
    ionization_potential_eV : float
        Ionization potential in eV
    U_I : float
        Partition function of neutral species
    U_II : float
        Partition function of ionized species
    saha_const : float
        Saha constant (SAHA_CONST_CM3)

    Returns
    -------
    UFloat
        Saha factor with propagated uncertainty
    """
    check_uncertainties_available()

    S_raw = (saha_const / n_e) * (T_eV_u**1.5) * umath.exp(-ionization_potential_eV / T_eV_u)
    S = S_raw * 2.0 * (U_II / U_I)

    return S


def propagate_through_closure_standard(
    intercepts_u: Dict[str, "UFloat"],
    partition_funcs: Dict[str, float],
) -> Dict[str, "UFloat"]:
    """
    Apply standard closure equation with uncertainty propagation.

    C_s = (U_s * exp(q_s)) / F
    F = sum(U_s * exp(q_s))

    Parameters
    ----------
    intercepts_u : Dict[str, UFloat]
        Boltzmann intercepts with uncertainties for each element
    partition_funcs : Dict[str, float]
        Partition function values U_s(T) for each element

    Returns
    -------
    Dict[str, UFloat]
        Concentrations with propagated uncertainties
    """
    check_uncertainties_available()

    # Calculate relative concentrations with uncertainty
    rel_concentrations_u: Dict[str, "UFloat"] = {}
    for element, q_s_u in intercepts_u.items():
        if element not in partition_funcs:
            logger.warning(f"Missing partition function for {element}")
            continue

        U_s = partition_funcs[element]
        rel_C_u = U_s * umath.exp(q_s_u)
        rel_concentrations_u[element] = rel_C_u

    if not rel_concentrations_u:
        return {}

    # F = sum of relative concentrations (with correlations tracked)
    F_u = sum(rel_concentrations_u.values())

    # Normalize
    concentrations_u = {el: rel_C / F_u for el, rel_C in rel_concentrations_u.items()}

    return concentrations_u


def propagate_through_closure_matrix(
    intercepts_u: Dict[str, "UFloat"],
    partition_funcs: Dict[str, float],
    matrix_element: str,
    matrix_fraction: float,
) -> Dict[str, "UFloat"]:
    """
    Apply matrix-mode closure with uncertainty propagation.

    Matrix element has known concentration, determines F.

    Parameters
    ----------
    intercepts_u : Dict[str, UFloat]
        Boltzmann intercepts with uncertainties
    partition_funcs : Dict[str, float]
        Partition functions
    matrix_element : str
        Element with known concentration
    matrix_fraction : float
        Known concentration of matrix element

    Returns
    -------
    Dict[str, UFloat]
        Concentrations with propagated uncertainties
    """
    check_uncertainties_available()

    if matrix_element not in intercepts_u or matrix_element not in partition_funcs:
        logger.error(f"Matrix element {matrix_element} missing from data")
        return propagate_through_closure_standard(intercepts_u, partition_funcs)

    U_m = partition_funcs[matrix_element]
    q_m_u = intercepts_u[matrix_element]
    rel_C_m_u = U_m * umath.exp(q_m_u)

    # F determined by matrix element
    F_u = rel_C_m_u / matrix_fraction

    # Calculate all concentrations
    concentrations_u: Dict[str, "UFloat"] = {}
    for element, q_s_u in intercepts_u.items():
        if element in partition_funcs:
            U_s = partition_funcs[element]
            rel_C_u = U_s * umath.exp(q_s_u)
            concentrations_u[element] = rel_C_u / F_u

    return concentrations_u


def extract_values_and_uncertainties(
    ufloat_dict: Dict[str, "UFloat"],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract nominal values and uncertainties from a dict of ufloats.

    Parameters
    ----------
    ufloat_dict : Dict[str, UFloat]
        Dictionary of ufloat values

    Returns
    -------
    tuple[Dict[str, float], Dict[str, float]]
        (nominal_values, uncertainties) dictionaries
    """
    check_uncertainties_available()

    nominal = {k: v.nominal_value for k, v in ufloat_dict.items()}
    uncertainties = {k: v.std_dev for k, v in ufloat_dict.items()}

    return nominal, uncertainties
