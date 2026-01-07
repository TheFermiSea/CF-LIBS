"""
Iterative solver for Classic CF-LIBS.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from collections import defaultdict

from cflibs.core.constants import KB_EV, SAHA_CONST_CM3, STP_PRESSURE, EV_TO_K
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter
from cflibs.inversion.closure import ClosureEquation
from cflibs.plasma.partition import PartitionFunctionEvaluator
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.solver")


@dataclass
class CFLIBSResult:
    """
    Result of the iterative CF-LIBS inversion.
    """

    temperature_K: float
    temperature_uncertainty_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    concentration_uncertainties: Dict[str, float]
    iterations: int
    converged: bool
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class IterativeCFLIBSSolver:
    """
    Implements the iterative self-consistent CF-LIBS algorithm.

    Algorithm:
    1. Guess T, ne
    2. Saha-Boltzmann correction to map ionic lines to neutral plane
    3. Multi-species Boltzmann fit to find common T and species intercepts
    4. Closure equation to find relative concentrations
    5. Enforce Pressure/Charge balance to update ne
    6. Iterate until convergence
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        max_iterations: int = 20,
        t_tolerance_k: float = 100.0,
        ne_tolerance_frac: float = 0.1,
        pressure_pa: float = STP_PRESSURE,
    ):
        self.atomic_db = atomic_db
        self.max_iterations = max_iterations
        self.t_tolerance_k = t_tolerance_k
        self.ne_tolerance_frac = ne_tolerance_frac
        self.pressure_pa = pressure_pa
        self.boltzmann_fitter = BoltzmannPlotFitter(outlier_sigma=2.5)

    def solve(
        self, observations: List[LineObservation], closure_mode: str = "standard", **closure_kwargs
    ) -> CFLIBSResult:
        """
        Solve for plasma parameters.

        Parameters
        ----------
        observations : List[LineObservation]
            Spectral lines
        closure_mode : str
            'standard', 'matrix', or 'oxide'
        closure_kwargs : dict
            Arguments for closure equation (e.g. matrix_element)

        Returns
        -------
        CFLIBSResult
        """
        # 1. Initialization
        T_K = 10000.0
        n_e = 1.0e17

        # Cache static data (IPs, atomic data)
        # Group observations by element
        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Pre-fetch Ionization Potentials
        ips = {}
        for el in elements:
            # Need IP of neutral (I -> II)
            ip = self.atomic_db.get_ionization_potential(el, 1)
            if ip is None:
                logger.warning(f"No IP for {el} I, assuming high")
                ip = 15.0  # Fallback
            ips[el] = ip

        # Iteration loop
        converged = False
        history = []

        for iteration in range(1, self.max_iterations + 1):
            T_prev = T_K
            ne_prev = n_e

            T_eV = T_K / EV_TO_K
            if T_eV < 0.1:
                T_eV = 0.1  # clamp

            # 2. Calculate Partition Functions & Saha Corrections
            partition_funcs = {}  # U_I for each element
            corrected_obs_map = defaultdict(list)

            for el in elements:
                # Get coeffs
                pf_I = self.atomic_db.get_partition_coefficients(el, 1)
                pf_II = self.atomic_db.get_partition_coefficients(el, 2)

                # Evaluate U(T)
                # We need U_I and U_II for Saha correction
                # Fallback to constant if no coeffs (should use database direct sum in robust impl)
                # For simplicity here, use evaluator if available, else 25/15

                U_I = 25.0
                if pf_I:
                    U_I = PartitionFunctionEvaluator.evaluate(T_K, pf_I.coefficients)
                elif hasattr(self.atomic_db, "get_energy_levels"):
                    # Fallback to direct sum
                    # This is slow inside loop, but acceptable for Phase 2b
                    # Ideally should cache
                    pass

                U_II = 15.0
                if pf_II:
                    U_II = PartitionFunctionEvaluator.evaluate(T_K, pf_II.coefficients)

                partition_funcs[el] = U_I

                # Calculate Saha Correction Factor (log scale) to subtract from Ion lines
                # y_neutral_plane = y_ion - delta
                # delta = ln(C_saha) + 1.5 ln T - ln ne + ln 2 - IP/T
                # SAHA_CONST_CM3 includes prefactors.
                # Formula: n_II/n_I = S
                # S = (SAHA_CONST / ne) * T^1.5 * exp(-IP/T) * (2 U_II / U_I)
                # ln S = ln(SAHA) - ln ne + 1.5 ln T - IP/T + ln 2 + ln U_II - ln U_I
                #
                # We want y_neutral = y_ion - ln(S * U_I / U_II) ??
                # Re-derivation from boltzmann.py logic:
                # Neutral: ln(I L / g A) = ln(n_I / U_I) - E/T
                # Ion:     ln(I L / g A) = ln(n_II / U_II) - E/T
                # Substitute n_II = n_I * S:
                # Ion:     y = ln(n_I * S / U_II) - E/T
                # Ion:     y = ln(n_I / U_I) + ln(S * U_I / U_II) - E/T
                # So Intercept_Ion = Intercept_Neutral + ln(S * U_I / U_II)
                # Therefore: Intercept_Neutral = Intercept_Ion - ln(S * U_I / U_II)
                # Correction term to subtract: D = ln(S) + ln(U_I) - ln(U_II)
                # D = ln(S) + ln(U_I / U_II)

                # S = (SAHA_CONST_CM3 / n_e) * T_eV**1.5 * exp(-IP/T_eV) * 2 * (U_II/U_I) ??
                # Wait, standard Saha formula usually includes U ratios.
                # My constants.py SAHA_CONST_CM3 is just the (2pi mk/h2)^1.5 part.
                # So S_raw = SAHA_CONST_CM3 * T_eV**1.5 / n_e * exp(-IP/T)
                # Actual ratio n_II/n_I = S_raw * 2 * (U_II/U_I)

                S_raw = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5) * np.exp(-ips[el] / T_eV)
                S_actual = S_raw * 2.0 * (U_II / U_I)

                correction_term = np.log(S_actual * (U_I / U_II))
                # Note: ln(S_actual * U_I / U_II) = ln(S_raw * 2)

                # Apply to observations
                for obs in obs_by_element[el]:
                    # Clone obs to avoid modifying original
                    new_obs = LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity,
                        intensity_uncertainty=obs.intensity_uncertainty,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )

                    # Apply correction if Ion (Stage 2)
                    if obs.ionization_stage == 2:
                        # We hack the intensity to apply the logarithmic shift
                        # y_new = y_old - correction
                        # ln(I_new) = ln(I_old) - correction
                        # I_new = I_old * exp(-correction)
                        new_obs.intensity = obs.intensity * np.exp(-correction_term)
                        # Correct energy?
                        # Ion lines emit from E_k relative to Ion ground.
                        # To map to Neutral ground, we usually add IP.
                        # E_total = E_k + IP.
                        # Slope should be -1/T over the whole range.
                        # Yes, we must add IP to E_k for ionic lines to place them on the unified energy scale.
                        new_obs.E_k_ev = obs.E_k_ev + ips[el]

                    corrected_obs_map[el].append(new_obs)

            # 3. Multi-species Boltzmann Fit
            # Fit common slope

            # Calculate centroids
            centroids = {}
            pooled_x = []
            pooled_y = []
            pooled_w = []

            for el, obs_list in corrected_obs_map.items():
                if len(obs_list) < 2:
                    continue

                # Get valid points
                xs = np.array([o.E_k_ev for o in obs_list])
                ys = np.array([o.y_value for o in obs_list])
                ws = np.array(
                    [1.0 / o.y_uncertainty**2 if o.y_uncertainty > 0 else 1.0 for o in obs_list]
                )

                # Filter invalid
                mask = np.isfinite(ys)
                xs = xs[mask]
                ys = ys[mask]
                ws = ws[mask]

                if len(xs) == 0:
                    continue

                # Centroids
                x_bar = np.average(xs, weights=ws)
                y_bar = np.average(ys, weights=ws)
                centroids[el] = (x_bar, y_bar)

                # Center data
                pooled_x.extend(xs - x_bar)
                pooled_y.extend(ys - y_bar)
                pooled_w.extend(ws)

            if len(pooled_x) < 3:
                logger.warning("Insufficient points for fit")
                break

            # Fit slope through origin
            pooled_x = np.array(pooled_x)
            pooled_y = np.array(pooled_y)
            pooled_w = np.array(pooled_w)

            # Simple linear regression through origin: m = sum(wxy) / sum(wx^2)
            slope = np.sum(pooled_w * pooled_x * pooled_y) / np.sum(pooled_w * pooled_x**2)

            # Update T
            if slope >= 0:
                T_new = 50000.0  # Clamp max
            else:
                T_new = -1.0 / (slope * KB_EV)

            T_new_K = T_new

            # Damping
            T_K = 0.5 * T_prev + 0.5 * T_new_K

            # Calculate Intercepts
            intercepts = {}
            for el in centroids:
                x_bar, y_bar = centroids[el]
                q_s = y_bar - slope * x_bar
                intercepts[el] = q_s

            # 4. Closure
            if closure_mode == "matrix":
                closure_res = ClosureEquation.apply_matrix_mode(
                    intercepts, partition_funcs, **closure_kwargs
                )
            elif closure_mode == "oxide":
                closure_res = ClosureEquation.apply_oxide_mode(
                    intercepts, partition_funcs, **closure_kwargs
                )
            else:
                closure_res = ClosureEquation.apply_standard(intercepts, partition_funcs)

            concentrations = closure_res.concentrations

            # 5. Update Electron Density
            # Use pressure balance: P = (n_tot + n_e) k T
            # n_tot = P/kT - n_e
            # Also n_e = n_tot * avg_ionization
            # n_e = (P/kT - n_e) * avg_Z => n_e (1 + avg_Z) = avg_Z * P/kT
            # n_e = (avg_Z / (1 + avg_Z)) * (P / (k * T_K))

            # Calculate avg_Z based on Saha ratios
            # n_II / n_I = S(T, ne)
            # Z_bar_s = (1*n_I + 2*n_II) / (n_I + n_II) - 1  (since Z=0 for neutral? No, Z=1 neutral in our code)
            # In code sp_num=1 is neutral (charge 0), sp_num=2 is ion (charge +1)
            # So electron contribution: Neutral=0, Ion=1
            # Avg electrons per atom of species s:
            # eps_s = (n_II) / (n_I + n_II) = S / (1+S)

            total_eps = 0.0
            for el, C_s in concentrations.items():
                S_raw = (
                    (SAHA_CONST_CM3 / n_e)
                    * ((T_K / EV_TO_K) ** 1.5)
                    * np.exp(-ips[el] / (T_K / EV_TO_K))
                )
                # Use current U values
                U_I = partition_funcs.get(el, 25.0)
                # We need U_II again, maybe fetch or cache
                # For efficiency reuse approximate or previous
                pf_II = self.atomic_db.get_partition_coefficients(el, 2)
                if pf_II:
                    U_II = PartitionFunctionEvaluator.evaluate(T_K, pf_II.coefficients)
                else:
                    U_II = 15.0

                S = S_raw * 2.0 * (U_II / U_I)
                eps_s = S / (1.0 + S)
                total_eps += C_s * eps_s

            avg_Z = total_eps

            # Total particle density (nuclei)
            # n_tot = P / (k T (1 + avg_Z))
            # n_e = avg_Z * n_tot

            k_joule = 1.380649e-23
            n_tot = self.pressure_pa / (k_joule * T_K * (1.0 + avg_Z))
            # Convert to cm^-3
            n_tot_cm3 = n_tot * 1e-6

            ne_new = avg_Z * n_tot_cm3

            # Damping
            n_e = 0.5 * ne_prev + 0.5 * ne_new

            history.append((T_K, n_e))

            # Check convergence
            if (
                abs(T_K - T_prev) < self.t_tolerance_k
                and abs(n_e - ne_prev) / ne_prev < self.ne_tolerance_frac
            ):
                converged = True
                break

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=0.0,  # TODO: Propagate
            electron_density_cm3=n_e,
            concentrations=concentrations,
            concentration_uncertainties={},
            iterations=len(history),
            converged=converged,
            quality_metrics={"r_squared_last": 0.0},  # TODO
        )
