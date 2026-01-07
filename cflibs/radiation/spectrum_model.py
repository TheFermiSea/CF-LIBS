"""
Forward spectrum model that ties together all components.
"""

import numpy as np
from typing import Tuple

from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.atomic.database import AtomicDatabase
from cflibs.instrument.model import InstrumentModel
from cflibs.radiation.emissivity import calculate_spectrum_emissivity
from cflibs.instrument.convolution import apply_instrument_function
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.spectrum_model")


class SpectrumModel:
    """
    Forward model for computing synthetic LIBS spectra.

    This class integrates:
    - Plasma state (temperature, density, composition)
    - Saha-Boltzmann solver (ionization and excitation balance)
    - Line emissivity calculations
    - Instrument response and convolution
    """

    def __init__(
        self,
        plasma: SingleZoneLTEPlasma,
        atomic_db: AtomicDatabase,
        instrument: InstrumentModel,
        lambda_min: float,
        lambda_max: float,
        delta_lambda: float,
        path_length_m: float = 0.01,  # 1 cm default
        use_jax: bool = False,
    ):
        """
        Initialize spectrum model.

        Parameters
        ----------
        plasma : SingleZoneLTEPlasma
            Plasma state
        atomic_db : AtomicDatabase
            Atomic database
        instrument : InstrumentModel
            Instrument model
        lambda_min : float
            Minimum wavelength in nm
        lambda_max : float
            Maximum wavelength in nm
        delta_lambda : float
            Wavelength step in nm
        path_length_m : float
            Plasma path length in meters (for optically thin approximation)
        use_jax : bool
            Use JAX acceleration for broadening when available
        """
        self.plasma = plasma
        self.atomic_db = atomic_db
        self.instrument = instrument
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.delta_lambda = delta_lambda
        self.path_length_m = path_length_m
        self.use_jax = use_jax

        # Create wavelength grid
        self.wavelength = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)

        # Initialize solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        logger.info(
            f"Initialized SpectrumModel: λ=[{lambda_min:.1f}, {lambda_max:.1f}] nm, "
            f"Δλ={delta_lambda:.3f} nm, {len(self.wavelength)} points"
        )

    def compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute synthetic spectrum.

        Returns
        -------
        wavelength : array
            Wavelength grid in nm
        intensity : array
            Spectral intensity in W m^-2 nm^-1 sr^-1
        """
        # Validate plasma
        self.plasma.validate()

        # 1. Solve Saha-Boltzmann for level populations
        logger.debug("Solving Saha-Boltzmann equations...")
        populations = self.solver.solve_plasma(self.plasma)

        # 2. Get transitions for all species
        logger.debug("Loading transitions...")
        all_transitions = []
        for element in self.plasma.species.keys():
            transitions = self.atomic_db.get_transitions(
                element,
                wavelength_min=self.lambda_min,
                wavelength_max=self.lambda_max,
                min_relative_intensity=10.0,  # Filter weak lines
            )
            all_transitions.extend(transitions)

        logger.debug(f"Found {len(all_transitions)} transitions")

        # 3. Calculate line emissivity
        logger.debug("Calculating line emissivity...")
        # Use Doppler broadening for now (Gaussian only in Phase 1)
        # Approximate sigma from temperature
        T_eV = self.plasma.T_e_eV
        # Rough estimate: sigma ~ 0.01 nm for typical conditions
        # Note: Proper Doppler width calculation will be implemented in Phase 2
        # For now, use a simple temperature-dependent estimate
        # Typical Doppler width: ~0.01 nm at 10000 K for medium-mass elements
        sigma_nm = 0.01 * np.sqrt(T_eV / 0.86)  # Scale with sqrt(T) in eV

        emissivity = calculate_spectrum_emissivity(
            all_transitions, populations, self.wavelength, sigma_nm, use_jax=self.use_jax
        )

        # 4. Convert to intensity (optically thin: I = ε * L)
        intensity = emissivity * self.path_length_m

        # 5. Apply instrument response
        if self.instrument.response_curve is not None:
            logger.debug("Applying instrument response...")
            intensity = self.instrument.apply_response(self.wavelength, intensity)

        # 6. Apply instrument function (convolution)
        logger.debug("Applying instrument function...")
        if self.use_jax:
            from cflibs.instrument.convolution import apply_instrument_function_jax

            intensity = apply_instrument_function_jax(
                self.wavelength, intensity, self.instrument.resolution_sigma_nm
            )
        else:
            intensity = apply_instrument_function(
                self.wavelength, intensity, self.instrument.resolution_sigma_nm
            )

        logger.info("Spectrum computation complete")

        return self.wavelength, intensity
