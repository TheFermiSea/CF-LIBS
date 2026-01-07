"""
Self-absorption correction for CF-LIBS analysis.

Self-absorption occurs when strong emission lines are partially reabsorbed
by cooler atoms in the outer plasma regions, leading to intensity underestimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from cflibs.core.constants import EV_TO_K
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("inversion.self_absorption")

# Physical constants
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 2.99792458e8  # m/s
K_BOLTZMANN = 1.380649e-23  # J/K


@dataclass
class AbsorptionCorrectionResult:
    """Result of self-absorption correction for a single line."""

    original_intensity: float
    corrected_intensity: float
    optical_depth: float
    correction_factor: float
    is_optically_thick: bool
    iterations: int = 1


@dataclass
class SelfAbsorptionResult:
    """Result of self-absorption correction for all lines."""

    corrected_observations: List[LineObservation]
    masked_observations: List[LineObservation]
    corrections: Dict[float, AbsorptionCorrectionResult]  # wavelength -> result
    n_corrected: int
    n_masked: int
    max_optical_depth: float
    warnings: List[str] = field(default_factory=list)


class SelfAbsorptionCorrector:
    """
    Corrects for self-absorption in optically thick plasmas.

    Uses the curve-of-growth approach:
    1. Estimate optical depth τ₀ at line center
    2. Apply correction factor: I_true = I_measured / f(τ)
    3. Where f(τ) = (1 - exp(-τ)) / τ (for Gaussian profile)

    For very high optical depth, lines can be masked instead of corrected.
    """

    def __init__(
        self,
        optical_depth_threshold: float = 0.1,
        mask_threshold: float = 3.0,
        max_iterations: int = 5,
        convergence_tolerance: float = 0.01,
        plasma_length_cm: float = 0.1,
    ):
        """
        Initialize corrector.

        Parameters
        ----------
        optical_depth_threshold : float
            Minimum τ₀ to apply correction (below this, line is optically thin)
        mask_threshold : float
            τ₀ above which to mask line instead of correct
        max_iterations : int
            Maximum iterations for recursive correction
        convergence_tolerance : float
            Relative change threshold for convergence
        plasma_length_cm : float
            Estimated plasma depth (path length) in cm
        """
        self.optical_depth_threshold = optical_depth_threshold
        self.mask_threshold = mask_threshold
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.plasma_length_cm = plasma_length_cm

    def correct(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        concentrations: Dict[str, float],
        total_number_density_cm3: float,
        partition_funcs: Dict[str, float],
        lower_level_energies: Optional[Dict[float, float]] = None,
    ) -> SelfAbsorptionResult:
        """
        Apply self-absorption correction to line observations.

        Parameters
        ----------
        observations : List[LineObservation]
            Line observations to correct
        temperature_K : float
            Plasma temperature
        concentrations : Dict[str, float]
            Elemental concentrations (mass fractions)
        total_number_density_cm3 : float
            Total number density of plasma
        partition_funcs : Dict[str, float]
            Partition functions U(T) for each element
        lower_level_energies : Dict[float, float], optional
            Lower level energies E_i by wavelength (nm -> eV)

        Returns
        -------
        SelfAbsorptionResult
        """
        if lower_level_energies is None:
            lower_level_energies = {}

        warnings = []
        corrected_obs = []
        masked_obs = []
        corrections = {}
        max_tau = 0.0

        for obs in observations:
            # Get lower level energy (default to 0 = ground state, worst case)
            E_i_ev = lower_level_energies.get(obs.wavelength_nm, 0.0)

            # Estimate optical depth
            tau = self._estimate_optical_depth(
                obs,
                temperature_K,
                concentrations,
                total_number_density_cm3,
                partition_funcs,
                E_i_ev,
            )

            max_tau = max(max_tau, tau)

            if tau > self.mask_threshold:
                # Too optically thick - mask this line
                masked_obs.append(obs)
                corrections[obs.wavelength_nm] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=0.0,
                    optical_depth=tau,
                    correction_factor=0.0,
                    is_optically_thick=True,
                )
                warnings.append(
                    f"Line {obs.wavelength_nm:.2f} nm masked: τ={tau:.2f} > {self.mask_threshold}"
                )

            elif tau > self.optical_depth_threshold:
                # Apply correction
                result = self._apply_recursive_correction(obs, tau)
                corrected_obs.append(self._create_corrected_observation(obs, result))
                corrections[obs.wavelength_nm] = result

            else:
                # Optically thin - no correction needed
                corrected_obs.append(obs)
                corrections[obs.wavelength_nm] = AbsorptionCorrectionResult(
                    original_intensity=obs.intensity,
                    corrected_intensity=obs.intensity,
                    optical_depth=tau,
                    correction_factor=1.0,
                    is_optically_thick=False,
                )

        return SelfAbsorptionResult(
            corrected_observations=corrected_obs,
            masked_observations=masked_obs,
            corrections=corrections,
            n_corrected=len([c for c in corrections.values() if c.correction_factor != 1.0]),
            n_masked=len(masked_obs),
            max_optical_depth=max_tau,
            warnings=warnings,
        )

    def _estimate_optical_depth(
        self,
        obs: LineObservation,
        temperature_K: float,
        concentrations: Dict[str, float],
        total_n_cm3: float,
        partition_funcs: Dict[str, float],
        E_i_ev: float,
    ) -> float:
        """
        Estimate optical depth at line center.

        Uses:
        τ₀ = (π e² / m_e c) × f_ik × n_i × L × φ(ν₀)

        Simplified to:
        τ₀ ≈ B × (g_k A_ki λ³ / 8π) × (n_s / U(T)) × exp(-E_i/kT) × L

        Where B is a constant including physical constants.
        """
        element = obs.element
        C_s = concentrations.get(element, 0.0)
        U_T = partition_funcs.get(element, 25.0)

        if C_s <= 0 or U_T <= 0:
            return 0.0

        # Species number density
        n_s = C_s * total_n_cm3

        # Lower level population (Boltzmann)
        T_eV = temperature_K / EV_TO_K
        if T_eV <= 0:
            return 0.0

        # Statistical weight of lower level (approximate as g_k for now)
        g_i = obs.g_k  # Should be lower level g, but often similar order

        exp_factor = np.exp(-E_i_ev / T_eV)
        n_i = n_s * (g_i / U_T) * exp_factor

        # Wavelength in cm
        lambda_cm = obs.wavelength_nm * 1e-7

        # Absorption oscillator strength from A_ki
        # f_ik ≈ (m_e c / 8π² e²) × (g_k/g_i) × λ² × A_ki
        # Simplified: use A_ki directly with scaling

        # Optical depth estimate (order of magnitude)
        # τ ≈ σ × n_i × L
        # σ ≈ (π e² / m_e c) × f × φ(ν) ≈ 10^-12 cm² (typical)

        # Using simpler scaling based on Einstein A coefficient:
        # τ ∝ A_ki × λ³ × n_i × L
        SCALE_FACTOR = 1e-25  # Empirical scaling to get reasonable τ values

        tau = SCALE_FACTOR * obs.A_ki * (lambda_cm**3) * n_i * self.plasma_length_cm

        return max(0.0, tau)

    def _apply_recursive_correction(
        self,
        obs: LineObservation,
        tau_initial: float,
    ) -> AbsorptionCorrectionResult:
        """
        Apply recursive self-absorption correction.

        The correction factor for a Gaussian line profile is:
        f(τ) = (1 - exp(-τ)) / τ

        I_true = I_measured / f(τ)
        """
        tau = tau_initial
        I_corrected = obs.intensity
        iteration = 0

        for iteration in range(self.max_iterations):
            # Correction factor
            if tau < 1e-10:
                f_tau = 1.0
            elif tau > 50:
                # Saturated limit: f(τ) ≈ 1/τ
                f_tau = 1.0 / tau
            else:
                f_tau = (1.0 - np.exp(-tau)) / tau

            # Corrected intensity
            I_new = obs.intensity / f_tau

            # Check convergence
            if I_corrected > 0:
                rel_change = abs(I_new - I_corrected) / I_corrected
                if rel_change < self.convergence_tolerance:
                    I_corrected = I_new
                    break

            I_corrected = I_new

            # Update tau for next iteration (intensity affects population estimate)
            # This is a simplification - full iteration would recalculate τ
            # based on updated concentrations from the solver
            tau = tau * (I_new / obs.intensity) if obs.intensity > 0 else tau

        return AbsorptionCorrectionResult(
            original_intensity=obs.intensity,
            corrected_intensity=I_corrected,
            optical_depth=tau_initial,
            correction_factor=obs.intensity / I_corrected if I_corrected > 0 else 0.0,
            is_optically_thick=tau_initial > 1.0,
            iterations=iteration + 1,
        )

    def _create_corrected_observation(
        self,
        obs: LineObservation,
        result: AbsorptionCorrectionResult,
    ) -> LineObservation:
        """Create a new observation with corrected intensity."""
        return LineObservation(
            wavelength_nm=obs.wavelength_nm,
            intensity=result.corrected_intensity,
            intensity_uncertainty=(
                obs.intensity_uncertainty / result.correction_factor
                if result.correction_factor > 0
                else obs.intensity_uncertainty
            ),
            element=obs.element,
            ionization_stage=obs.ionization_stage,
            E_k_ev=obs.E_k_ev,
            g_k=obs.g_k,
            A_ki=obs.A_ki,
        )


def estimate_optical_depth_from_intensity_ratio(
    intensity_strong: float,
    intensity_weak: float,
    theoretical_ratio: float,
) -> float:
    """
    Estimate optical depth from intensity ratio of doublet lines.

    For two lines from the same upper level with theoretical ratio R_0:
    R_observed / R_0 = f(τ_strong) / f(τ_weak)

    This is commonly used with doublets where the stronger line
    may be self-absorbed while the weaker line remains optically thin.

    Parameters
    ----------
    intensity_strong : float
        Measured intensity of stronger line
    intensity_weak : float
        Measured intensity of weaker line
    theoretical_ratio : float
        Theoretical intensity ratio (from A*g values)

    Returns
    -------
    float
        Estimated optical depth of stronger line
    """
    if intensity_weak <= 0 or theoretical_ratio <= 0:
        return 0.0

    observed_ratio = intensity_strong / intensity_weak
    ratio_reduction = observed_ratio / theoretical_ratio

    if ratio_reduction >= 1.0:
        # No absorption detected
        return 0.0

    # Solve (1 - exp(-τ))/τ = ratio_reduction
    # Use Newton-Raphson or bisection

    def f_tau(tau: float) -> float:
        if tau < 1e-10:
            return 1.0
        return (1.0 - np.exp(-tau)) / tau

    # Bisection search
    tau_low, tau_high = 0.0, 10.0
    tau_mid = (tau_low + tau_high) / 2

    for _ in range(50):
        tau_mid = (tau_low + tau_high) / 2
        f_mid = f_tau(tau_mid)

        if f_mid > ratio_reduction:
            tau_low = tau_mid
        else:
            tau_high = tau_mid

        if tau_high - tau_low < 0.01:
            break

    return tau_mid
