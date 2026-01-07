"""
Line emissivity calculations.
"""

import numpy as np
from typing import Dict, Tuple, List

from cflibs.core.constants import H_PLANCK, C_LIGHT
from cflibs.atomic.structures import Transition
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.emissivity")


def calculate_line_emissivity(
    transition: Transition, upper_level_population_cm3: float, wavelength_nm: float = None
) -> float:
    """
    Calculate spectral emissivity for a transition.

    For spontaneous emission:
    ε_λ = (hc / 4πλ) * A_ki * n_k

    Parameters
    ----------
    transition : Transition
        Atomic transition
    upper_level_population_cm3 : float
        Population of upper level in cm^-3
    wavelength_nm : float, optional
        Wavelength in nm (defaults to transition.wavelength_nm)

    Returns
    -------
    float
        Spectral emissivity in W m^-3 nm^-1
    """
    if wavelength_nm is None:
        wavelength_nm = transition.wavelength_nm

    # Convert wavelength to meters
    wavelength_m = wavelength_nm * 1e-9

    # Convert population to m^-3
    n_k_m3 = upper_level_population_cm3 * 1e6

    # Emissivity: ε = (hc / 4πλ) * A_ki * n_k
    # Units: (J·s * m/s) / (m) * (1/s) * (1/m^3) = J/(s·m^3) = W/m^3
    # For spectral emissivity, we need per nm, so we'll handle that in the profile
    epsilon = (H_PLANCK * C_LIGHT / (4 * np.pi * wavelength_m)) * transition.A_ki * n_k_m3

    return epsilon


def calculate_spectrum_emissivity(
    transitions: List[Transition],
    populations: Dict[Tuple[str, int, float], float],
    wavelength_grid: np.ndarray,
    sigma_nm: float,
    use_jax: bool = False,
) -> np.ndarray:
    """
    Calculate total spectral emissivity on a wavelength grid.

    Parameters
    ----------
    transitions : List[Transition]
        List of transitions to include
    populations : Dict[Tuple[str, int, float], float]
        Level populations from Saha-Boltzmann solver
    wavelength_grid : array
        Wavelength grid in nm
    sigma_nm : float
        Gaussian broadening width (standard deviation) in nm
    use_jax : bool
        Use JAX acceleration for broadening when available

    Returns
    -------
    array
        Spectral emissivity in W m^-3 nm^-1
    """
    from cflibs.radiation.profiles import apply_gaussian_broadening

    line_wavelengths = []
    line_emissivities = []

    for trans in transitions:
        # Find population for upper level
        key = (trans.element, trans.ionization_stage, trans.E_k_ev)
        if key in populations:
            n_k = populations[key]
            epsilon = calculate_line_emissivity(trans, n_k)
            line_wavelengths.append(trans.wavelength_nm)
            line_emissivities.append(epsilon)
        else:
            logger.debug(
                f"No population found for {trans.element} {trans.ionization_stage} "
                f"E_k={trans.E_k_ev:.3f} eV"
            )

    if not line_wavelengths:
        return np.zeros_like(wavelength_grid)

    # Apply Gaussian broadening
    line_wavelengths = np.array(line_wavelengths)
    line_emissivities = np.array(line_emissivities)

    # Convert to spectral emissivity (per nm)
    # The Gaussian profile integrates to 1, so we just need to scale by line intensity
    if use_jax:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError("JAX is not installed. Install with: pip install jax jaxlib") from exc
        from cflibs.radiation.profiles import apply_gaussian_broadening_jax

        spectrum = apply_gaussian_broadening_jax(
            jnp.asarray(wavelength_grid),
            jnp.asarray(line_wavelengths),
            jnp.asarray(line_emissivities),
            float(sigma_nm),
        )
        return np.array(spectrum)

    spectrum = apply_gaussian_broadening(
        wavelength_grid, line_wavelengths, line_emissivities, sigma_nm
    )

    return spectrum
