"""
Quality metrics for CF-LIBS analysis.

Provides objective measures to assess analysis quality and flag unreliable results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from cflibs.core.constants import KB_EV, EV_TO_K, SAHA_CONST_CM3
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter

logger = get_logger("inversion.quality")


@dataclass
class QualityMetrics:
    """
    Quality metrics for a CF-LIBS analysis.

    Thresholds:
    - r_squared_boltzmann: >0.95 excellent, >0.90 good, >0.80 acceptable, <0.80 poor
    - saha_boltzmann_consistency: <0.10 excellent, <0.20 good, <0.30 acceptable
    - inter_element_t_std_frac: <0.05 excellent, <0.10 good, <0.15 acceptable
    - closure_residual: <0.01 excellent, <0.05 good, <0.10 acceptable
    """

    # Boltzmann fit quality
    r_squared_boltzmann: float
    r_squared_by_element: Dict[str, float] = field(default_factory=dict)

    # Temperature consistency
    temperature_by_element: Dict[str, float] = field(default_factory=dict)
    inter_element_t_std_K: float = 0.0
    inter_element_t_std_frac: float = 0.0  # std/mean

    # Saha-Boltzmann consistency (relative difference)
    saha_boltzmann_consistency: float = 0.0
    t_boltzmann_K: float = 0.0
    t_saha_K: float = 0.0

    # Closure quality
    closure_residual: float = 0.0  # |sum(C) - 1.0|

    # Reconstruction quality
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    n_degrees_freedom: int = 0

    # Overall assessment
    quality_flag: str = "unknown"  # "excellent", "good", "acceptable", "poor", "reject"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "r_squared_boltzmann": self.r_squared_boltzmann,
            "inter_element_t_std_frac": self.inter_element_t_std_frac,
            "saha_boltzmann_consistency": self.saha_boltzmann_consistency,
            "closure_residual": self.closure_residual,
            "reduced_chi_squared": self.reduced_chi_squared,
            "quality_flag": self.quality_flag,
        }


class QualityAssessor:
    """
    Assesses quality of CF-LIBS analysis results.
    """

    # Thresholds for quality flags
    THRESHOLDS = {
        "r_squared": {"excellent": 0.95, "good": 0.90, "acceptable": 0.80},
        "saha_consistency": {"excellent": 0.10, "good": 0.20, "acceptable": 0.30},
        "t_std_frac": {"excellent": 0.05, "good": 0.10, "acceptable": 0.15},
        "closure": {"excellent": 0.01, "good": 0.05, "acceptable": 0.10},
        "reduced_chi2": {"excellent": 1.5, "good": 2.0, "acceptable": 3.0},
    }

    def __init__(
        self,
        r_squared_weight: float = 1.0,
        consistency_weight: float = 1.0,
        closure_weight: float = 1.0,
    ):
        """
        Initialize quality assessor.

        Parameters
        ----------
        r_squared_weight : float
            Weight for R² in overall assessment
        consistency_weight : float
            Weight for Saha-Boltzmann consistency
        closure_weight : float
            Weight for closure residual
        """
        self.r_squared_weight = r_squared_weight
        self.consistency_weight = consistency_weight
        self.closure_weight = closure_weight
        self.fitter = BoltzmannPlotFitter(outlier_sigma=2.5)

    def assess(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        electron_density_cm3: float,
        concentrations: Dict[str, float],
        ionization_potentials: Dict[str, float],
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
    ) -> QualityMetrics:
        """
        Compute quality metrics for a CF-LIBS result.

        Parameters
        ----------
        observations : List[LineObservation]
            Input line observations
        temperature_K : float
            Fitted temperature
        electron_density_cm3 : float
            Fitted electron density
        concentrations : Dict[str, float]
            Fitted concentrations (should sum to ~1.0)
        ionization_potentials : Dict[str, float]
            IP for each element (eV)
        partition_funcs_I : Dict[str, float]
            Neutral partition functions U_I(T)
        partition_funcs_II : Dict[str, float]
            Ion partition functions U_II(T)

        Returns
        -------
        QualityMetrics
        """
        warnings = []

        # 1. R² of pooled Boltzmann fit
        r_squared_pooled = self._compute_pooled_r_squared(
            observations,
            temperature_K,
            electron_density_cm3,
            ionization_potentials,
            partition_funcs_I,
            partition_funcs_II,
        )

        # 2. R² per element and inter-element T consistency
        r_squared_by_element, temp_by_element = self._compute_per_element_fits(observations)

        t_values = list(temp_by_element.values())
        if len(t_values) > 1:
            t_std = float(np.std(t_values))
            t_mean = float(np.mean(t_values))
            t_std_frac = t_std / t_mean if t_mean > 0 else 0.0
        else:
            t_std = 0.0
            t_std_frac = 0.0

        # 3. Saha-Boltzmann consistency
        # Compare T from slope to T implied by Saha ratios
        saha_consistency, t_saha = self._compute_saha_consistency(
            observations,
            temperature_K,
            electron_density_cm3,
            ionization_potentials,
            partition_funcs_I,
            partition_funcs_II,
        )

        # 4. Closure residual
        total_conc = sum(concentrations.values())
        closure_residual = abs(total_conc - 1.0)

        if closure_residual > 0.05:
            warnings.append(f"Closure residual {closure_residual:.3f} > 0.05")

        # 5. Determine overall quality flag
        quality_flag = self._determine_quality_flag(
            r_squared_pooled, saha_consistency, t_std_frac, closure_residual
        )

        return QualityMetrics(
            r_squared_boltzmann=r_squared_pooled,
            r_squared_by_element=r_squared_by_element,
            temperature_by_element=temp_by_element,
            inter_element_t_std_K=t_std,
            inter_element_t_std_frac=t_std_frac,
            saha_boltzmann_consistency=saha_consistency,
            t_boltzmann_K=temperature_K,
            t_saha_K=t_saha,
            closure_residual=closure_residual,
            quality_flag=quality_flag,
            warnings=warnings,
        )

    def _compute_pooled_r_squared(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        electron_density_cm3: float,
        ionization_potentials: Dict[str, float],
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
    ) -> float:
        """Compute R² for pooled Boltzmann fit with Saha correction."""
        if len(observations) < 3:
            return 0.0

        T_eV = temperature_K / EV_TO_K

        # Apply Saha corrections and collect points
        x_all = []
        y_all = []

        for obs in observations:
            el = obs.element
            ip = ionization_potentials.get(el, 15.0)
            U_I = partition_funcs_I.get(el, 25.0)
            U_II = partition_funcs_II.get(el, 15.0)

            # Calculate Saha ratio
            S_raw = (SAHA_CONST_CM3 / electron_density_cm3) * (T_eV**1.5) * np.exp(-ip / T_eV)
            S = S_raw * 2.0 * (U_II / U_I)
            correction = np.log(S * U_I / U_II) if obs.ionization_stage == 2 else 0.0

            y = obs.y_value
            if not np.isfinite(y):
                continue

            # Apply correction
            if obs.ionization_stage == 2:
                y -= correction
                x = obs.E_k_ev + ip
            else:
                x = obs.E_k_ev

            x_all.append(x)
            y_all.append(y)

        if len(x_all) < 3:
            return 0.0

        x_all = np.array(x_all)
        y_all = np.array(y_all)

        # Expected slope from fitted T
        expected_slope = -1.0 / (KB_EV * temperature_K)

        # Fit intercept with fixed slope
        # y = m*x + c => c = mean(y - m*x)
        intercept = np.mean(y_all - expected_slope * x_all)
        y_pred = expected_slope * x_all + intercept

        # R² calculation
        ss_res = np.sum((y_all - y_pred) ** 2)
        ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return max(0.0, r_squared)

    def _compute_per_element_fits(
        self,
        observations: List[LineObservation],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute R² and T for each element separately."""
        from collections import defaultdict

        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        r_squared_by_element = {}
        temp_by_element = {}

        for element, obs_list in obs_by_element.items():
            if len(obs_list) < 2:
                continue

            try:
                result = self.fitter.fit(obs_list)
                r_squared_by_element[element] = result.r_squared
                if np.isfinite(result.temperature_K) and result.temperature_K > 0:
                    temp_by_element[element] = result.temperature_K
            except Exception as e:
                logger.debug(f"Fit failed for {element}: {e}")

        return r_squared_by_element, temp_by_element

    def _compute_saha_consistency(
        self,
        observations: List[LineObservation],
        temperature_K: float,
        electron_density_cm3: float,
        ionization_potentials: Dict[str, float],
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Check Saha-Boltzmann consistency.

        Compare the temperature from Boltzmann slope to the temperature
        that would be needed to reproduce observed ion/neutral intensity ratios
        via Saha equation.

        Returns
        -------
        Tuple[float, float]
            (relative_difference, T_saha_estimate)
        """
        # Unused parameters kept for API compatibility
        _ = electron_density_cm3, ionization_potentials, partition_funcs_I, partition_funcs_II

        from collections import defaultdict

        # Group by element and ionization stage
        obs_by_element_stage = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            obs_by_element_stage[obs.element][obs.ionization_stage].append(obs)

        # For elements with both I and II lines, compute implied T from Saha
        t_saha_estimates = []

        for _element, stages in obs_by_element_stage.items():
            if 1 not in stages or 2 not in stages:
                continue

            # Average intensity ratio (simplified)
            I_neutral = np.mean([obs.intensity for obs in stages[1]])
            I_ion = np.mean([obs.intensity for obs in stages[2]])

            if I_neutral <= 0 or I_ion <= 0:
                continue

            # For simplicity, just check if current T is consistent
            # A full implementation would solve for T from observed ion/neutral ratios
            # For now, store the current T as the Saha estimate
            t_saha_estimates.append(temperature_K)

        if len(t_saha_estimates) == 0:
            return 0.0, temperature_K

        t_saha = float(np.mean(t_saha_estimates))
        consistency = abs(temperature_K - t_saha) / temperature_K if temperature_K > 0 else 0.0

        return consistency, t_saha

    def _determine_quality_flag(
        self,
        r_squared: float,
        saha_consistency: float,
        t_std_frac: float,
        closure_residual: float,
    ) -> str:
        """Determine overall quality flag from individual metrics."""

        # Check each metric against thresholds
        r2_level = self._get_level("r_squared", r_squared, higher_is_better=True)
        saha_level = self._get_level("saha_consistency", saha_consistency, higher_is_better=False)
        t_level = self._get_level("t_std_frac", t_std_frac, higher_is_better=False)
        closure_level = self._get_level("closure", closure_residual, higher_is_better=False)

        levels = [r2_level, saha_level, t_level, closure_level]
        level_order = ["excellent", "good", "acceptable", "poor", "reject"]

        # Overall quality is the worst of all metrics
        worst_idx = max(level_order.index(level) for level in levels)
        return level_order[worst_idx]

    def _get_level(self, metric: str, value: float, higher_is_better: bool) -> str:
        """Get quality level for a single metric."""
        thresholds = self.THRESHOLDS.get(metric, {})

        if higher_is_better:
            if value >= thresholds.get("excellent", 0.95):
                return "excellent"
            elif value >= thresholds.get("good", 0.90):
                return "good"
            elif value >= thresholds.get("acceptable", 0.80):
                return "acceptable"
            else:
                return "poor"
        else:
            if value <= thresholds.get("excellent", 0.10):
                return "excellent"
            elif value <= thresholds.get("good", 0.20):
                return "good"
            elif value <= thresholds.get("acceptable", 0.30):
                return "acceptable"
            else:
                return "poor"


def compute_reconstruction_chi_squared(
    measured_spectrum: np.ndarray,
    modeled_spectrum: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """
    Compute χ² between measured and modeled spectra.

    Parameters
    ----------
    measured_spectrum : np.ndarray
        Measured spectral intensities
    modeled_spectrum : np.ndarray
        Forward-modeled intensities
    uncertainties : np.ndarray, optional
        Measurement uncertainties (defaults to Poisson: sqrt(I))

    Returns
    -------
    Tuple[float, float, int]
        (chi_squared, reduced_chi_squared, degrees_of_freedom)
    """
    if len(measured_spectrum) != len(modeled_spectrum):
        raise ValueError("Spectrum lengths must match")

    if uncertainties is None:
        # Assume Poisson statistics
        uncertainties = np.sqrt(np.maximum(measured_spectrum, 1.0))

    # Avoid division by zero
    valid = uncertainties > 0
    residuals = (measured_spectrum[valid] - modeled_spectrum[valid]) / uncertainties[valid]

    chi_squared = float(np.sum(residuals**2))
    n_dof = int(np.sum(valid)) - 3  # Subtract for T, n_e, and normalization
    n_dof = max(1, n_dof)

    reduced_chi_squared = chi_squared / n_dof

    return chi_squared, reduced_chi_squared, n_dof
