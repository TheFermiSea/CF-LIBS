"""
ALIAS (Automated Line Identification Algorithm for Spectroscopy) implementation.

Based on Noel et al. (2025) arXiv:2501.01057. The ALIAS algorithm identifies elements
in LIBS spectra through a 7-step process: peak detection, theoretical emissivity
calculation, line fusion, matching, threshold determination, scoring, and decision.
"""

from typing import List, Tuple, Optional
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import binom

from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import KB_EV
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.preprocessing import estimate_baseline, estimate_noise


class ALIASIdentifier:
    """
    ALIAS algorithm for automated element identification in LIBS spectra.

    The algorithm operates in 7 steps:
    1. Peak detection via 2nd derivative enhancement
    2. Theoretical emissivity calculation over (T, n_e) grid
    3. Line fusion within resolution element
    4. Matching theoretical lines to experimental peaks
    5. Emissivity threshold determination via detection rate
    6. Score computation (k_sim, k_rate, k_shift)
    7. Decision and confidence level calculation

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for transitions and partition functions
    resolving_power : float, optional
        Instrument resolving power R = λ/Δλ (default: 5000.0)
    T_range_K : Tuple[float, float], optional
        Temperature grid range in K (default: (8000.0, 12000.0))
    n_e_range_cm3 : Tuple[float, float], optional
        Electron density grid range in cm^-3 (default: (3e16, 3e17))
    T_steps : int, optional
        Number of temperature grid points (default: 5)
    n_e_steps : int, optional
        Number of electron density grid points (default: 3)
    intensity_threshold_factor : float, optional
        Peak detection threshold = factor × noise_estimate (default: 4.0)
    detection_threshold : float, optional
        Minimum confidence level for element detection (default: 0.02)
    chance_window_scale : float, optional
        Scale factor for chance-coincidence windows used in fill-factor estimation.
        The chance half-window is `chance_window_scale * (lambda / R)`.
    elements : Optional[List[str]], optional
        List of elements to search for. If None, searches all available (default: None)
    """

    # Crustal abundance in log10(ppm) — from CRC Handbook / USGS
    CRUSTAL_ABUNDANCE_LOG_PPM = {
        "O": 5.67, "Si": 5.44, "Al": 4.91, "Fe": 4.70, "Ca": 4.57,
        "Na": 4.36, "Mg": 4.33, "K": 4.32, "Ti": 3.75, "H": 3.15,
        "Mn": 2.98, "P": 2.97, "F": 2.80, "Ba": 2.70, "C": 2.30,
        "Sr": 2.57, "S": 2.56, "Zr": 2.23, "V": 2.10, "Cl": 2.20,
        "Cr": 2.00, "Ni": 1.88, "Zn": 1.88, "Cu": 1.78, "Co": 1.40,
        "Li": 1.30, "N": 1.30, "Ga": 1.28, "Pb": 1.15, "Rb": 1.95,
        "B": 1.00, "Sn": 0.35, "W": 0.18, "Mo": 0.18, "Ag": -0.62,
        "Cd": -0.82, "Au": -2.40,
    }

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        resolving_power: float = 5000.0,
        T_range_K: Tuple[float, float] = (8000.0, 12000.0),
        n_e_range_cm3: Tuple[float, float] = (3e16, 3e17),
        T_steps: int = 5,
        n_e_steps: int = 3,
        intensity_threshold_factor: float = 4.0,
        detection_threshold: float = 0.03,
        chance_window_scale: float = 0.4,
        elements: Optional[List[str]] = None,
        max_lines_per_element: int = 50,
        reference_temperature: float = 10000.0,
    ):
        self.atomic_db = atomic_db
        self.resolving_power = resolving_power
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps
        self.intensity_threshold_factor = intensity_threshold_factor
        self.detection_threshold = detection_threshold
        self.chance_window_scale = chance_window_scale
        self.elements = elements
        self.max_lines_per_element = max_lines_per_element
        self.reference_temperature = reference_temperature

        # Create Saha-Boltzmann solver
        self.solver = SahaBoltzmannSolver(atomic_db)

        # Create (T, n_e) grid
        self.T_grid_K = np.linspace(T_range_K[0], T_range_K[1], T_steps)
        self.n_e_grid_cm3 = np.linspace(n_e_range_cm3[0], n_e_range_cm3[1], n_e_steps)

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """
        Identify elements in experimental spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array (arbitrary units)

        Returns
        -------
        ElementIdentificationResult
            Complete identification result with detected/rejected elements
        """
        # Step 1: Detect peaks
        peaks = self._detect_peaks(wavelength, intensity)

        wl_min = np.min(wavelength)
        wl_max = np.max(wavelength)

        # Get elements to search
        if self.elements is None:
            # Get all available elements from database
            # For now, use common LIBS elements as default
            search_elements = ["Fe", "H", "Cu", "Al", "Ti", "Ca", "Mg", "Si"]
        else:
            search_elements = self.elements

        all_element_ids = []

        for element in search_elements:
            # Step 2: Compute theoretical emissivities
            element_lines = self._compute_element_emissivities(element, wl_min, wl_max)

            if not element_lines:
                continue

            # Step 3: Fuse nearby lines
            fused_lines = self._fuse_lines(element_lines, wavelength)

            if not fused_lines:
                continue

            # Step 4: Match lines to experimental peaks
            matched_mask, wavelength_shifts, matched_peak_idx = self._match_lines(
                fused_lines, peaks
            )

            # Step 5: Determine emissivity threshold
            if np.any(matched_mask):
                emissivity_threshold = self._determine_emissivity_threshold(
                    fused_lines, matched_mask
                )
            else:
                emissivity_threshold = -np.inf  # No matches, keep all for scoring

            # Step 6: Compute scores
            k_sim, k_rate, k_shift, P_maj, N_expected = (
                self._compute_scores(
                    fused_lines, matched_mask, matched_peak_idx,
                    wavelength_shifts, intensity, peaks, emissivity_threshold,
                )
            )

            # N_matched: lines both above threshold AND matched (for metadata)
            emissivities_arr = np.array([line["avg_emissivity"] for line in fused_lines])
            N_matched = int(np.sum(
                matched_mask & (emissivities_arr >= 10**emissivity_threshold)
            ))

            P_sig, fill_factor, p_chance, p_tail = self._compute_random_match_significance(
                peaks=peaks,
                wavelength=wavelength,
                N_expected=N_expected,
                N_matched=N_matched,
            )

            # Step 7: Decision — use N_expected in blend so k_sim always
            # gets weight when many lines predicted but few matched.
            k_det, CL = self._decide(
                k_sim, k_rate, k_shift, N_expected, intensity, peaks,
                element=element, P_maj=P_maj, P_sig=P_sig,
            )

            # Build ElementIdentification
            detected = CL >= self.detection_threshold

            # Create IdentifiedLine objects using matched_peak_idx for
            # consistent peak association (same peak _match_lines chose).
            matched_lines = []
            unmatched_lines = []
            for i, line_data in enumerate(fused_lines):
                trans = line_data["transition"]
                if matched_mask[i]:
                    pidx = matched_peak_idx[i]
                    peak_wl = peaks[pidx][1]
                    peak_int = intensity[peaks[pidx][0]]

                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=peak_wl,
                            wavelength_th_nm=line_data["wavelength_nm"],
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=peak_int,
                            emissivity_th=line_data["avg_emissivity"],
                            transition=trans,
                            correlation=k_sim,
                        )
                    )
                else:
                    unmatched_lines.append(trans)

            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=k_det,
                confidence=CL,
                n_matched_lines=np.sum(matched_mask),
                n_total_lines=len(fused_lines),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "k_sim": k_sim,
                    "k_rate": k_rate,
                    "k_shift": k_shift,
                    "k_det": k_det,
                    "emissivity_threshold": emissivity_threshold,
                    "N_expected": N_expected,
                    "N_matched": N_matched,
                    "P_maj": P_maj,
                    "P_ab": self._compute_P_ab(element),
                    "P_sig": P_sig,
                    "p_tail": p_tail,
                    "p_chance": p_chance,
                    "fill_factor": fill_factor,
                    "N_penalty": (
                        0.2 if N_expected <= 1
                        else (0.5 if N_expected == 2 else 1.0)
                    ),
                },
            )

            all_element_ids.append(element_id)

        # Split into detected/rejected
        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Count matched peaks (peak matched if any element matched it)
        matched_peak_indices = set()
        for element_id in detected_elements:
            for line in element_id.matched_lines:
                # Find peak index
                peak_idx = np.argmin(
                    np.abs(np.array([p[1] for p in peaks]) - line.wavelength_exp_nm)
                )
                matched_peak_indices.add(peak_idx)

        n_matched_peaks = len(matched_peak_indices)
        n_unmatched_peaks = len(peaks) - n_matched_peaks

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=peaks,
            n_peaks=len(peaks),
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=n_unmatched_peaks,
            algorithm="alias",
            parameters={
                "resolving_power": self.resolving_power,
                "T_min_K": self.T_range_K[0],
                "T_max_K": self.T_range_K[1],
                "n_e_min_cm3": self.n_e_range_cm3[0],
                "n_e_max_cm3": self.n_e_range_cm3[1],
                "intensity_threshold_factor": self.intensity_threshold_factor,
                "detection_threshold": self.detection_threshold,
            },
        )

    def _detect_peaks(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Detect peaks using 2nd derivative enhancement.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array
        intensity : np.ndarray
            Intensity array

        Returns
        -------
        List[Tuple[int, float]]
            List of (peak_index, peak_wavelength) tuples
        """
        # Estimate baseline and noise using sigma-clipped MAD
        baseline = estimate_baseline(wavelength, intensity)
        noise_estimate = estimate_noise(intensity, baseline)

        # Threshold in intensity domain (well-calibrated)
        threshold = noise_estimate * self.intensity_threshold_factor

        # Find peaks in baseline-corrected intensity
        corrected = intensity - baseline
        peak_indices, _ = find_peaks(corrected, height=threshold, prominence=threshold / 3)

        # Paper (Noël et al. 2025): enhance peak detection using negative 2nd derivative
        # Compute -d²I/dλ², zero negatives — true peaks have positive curvature here
        d2 = -np.gradient(np.gradient(corrected, wavelength), wavelength)
        d2[d2 < 0] = 0.0

        # Filter: keep peaks where d2 > 0 in a ±2-point neighborhood around peak center
        # This handles discretization effects where d2 peak may be slightly offset
        confirmed = []
        for idx in peak_indices:
            lo = max(0, idx - 2)
            hi = min(len(d2), idx + 3)
            if np.max(d2[lo:hi]) > 0:
                confirmed.append(idx)
        peak_indices = np.array(confirmed, dtype=int) if confirmed else np.array([], dtype=int)

        # Return as list of (index, wavelength) tuples
        peaks = [(int(idx), float(wavelength[idx])) for idx in peak_indices]

        return peaks

    def _compute_element_emissivities(
        self, element: str, wl_min: float, wl_max: float
    ) -> List[dict]:
        """
        Compute theoretical emissivities for element over (T, n_e) grid.

        Parameters
        ----------
        element : str
            Element symbol
        wl_min : float
            Minimum wavelength in nm
        wl_max : float
            Maximum wavelength in nm

        Returns
        -------
        List[dict]
            List of dicts with keys: transition, avg_emissivity, wavelength_nm
        """
        # Get transitions for element (try both neutral and ionized)
        transitions = []
        for ion_stage in [1, 2]:
            try:
                trans_list = self.atomic_db.get_transitions(
                    element, ion_stage, wavelength_min=wl_min, wavelength_max=wl_max
                )
                if trans_list:
                    transitions.extend(trans_list)
            except Exception:
                # No data for this ionization stage
                continue

        if not transitions:
            return []

        # Cap to strongest lines by estimated emissivity to avoid line-count disparity
        if len(transitions) > self.max_lines_per_element:
            kT = KB_EV * self.reference_temperature
            transitions = sorted(
                transitions,
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                reverse=True,
            )
            transitions = transitions[: self.max_lines_per_element]

        # Compute emissivities
        line_data = []
        total_density = 1e15  # Arbitrary reference density

        for transition in transitions:
            emissivities = []

            for T_K in self.T_grid_K:
                for n_e in self.n_e_grid_cm3:
                    T_eV = T_K * KB_EV

                    # Get ionization balance
                    try:
                        stage_densities = self.solver.solve_ionization_balance(
                            element, T_eV, n_e, total_density
                        )
                        stage_density = stage_densities.get(transition.ionization_stage, 0.0)
                        W_q = stage_density / total_density

                        # Get partition function
                        U_T = self.solver.calculate_partition_function(
                            element, transition.ionization_stage, T_eV
                        )

                        # Emissivity: eps = W^q * A_ki * g_k * exp(-E_k/kT) / U(T)
                        boltzmann_factor = np.exp(-transition.E_k_ev / T_eV)
                        eps = W_q * transition.A_ki * transition.g_k * boltzmann_factor / U_T

                        emissivities.append(eps)
                    except Exception:
                        # Failed for this grid point, skip
                        continue

            if emissivities:
                avg_emissivity = np.mean(emissivities)
                line_data.append(
                    {
                        "transition": transition,
                        "avg_emissivity": avg_emissivity,
                        "wavelength_nm": transition.wavelength_nm,
                    }
                )

        return line_data

    def _fuse_lines(self, line_data: List[dict], wavelength_nm: np.ndarray) -> List[dict]:
        """
        Fuse lines within resolution element.

        Parameters
        ----------
        line_data : List[dict]
            List of line dicts from _compute_element_emissivities
        wavelength_nm : np.ndarray
            Experimental wavelength array (for reference wavelength)

        Returns
        -------
        List[dict]
            Fused line list with combined emissivities
        """
        if not line_data:
            return []

        # Sort by wavelength
        sorted_lines = sorted(line_data, key=lambda x: x["wavelength_nm"])

        # Resolution element at mean wavelength
        mean_wl = np.mean(wavelength_nm)
        delta_lambda = mean_wl / self.resolving_power

        # Group lines within delta_lambda
        fused = []
        current_group = [sorted_lines[0]]

        for i in range(1, len(sorted_lines)):
            line = sorted_lines[i]
            prev_line = current_group[-1]

            if abs(line["wavelength_nm"] - prev_line["wavelength_nm"]) <= delta_lambda:
                # Add to current group
                current_group.append(line)
            else:
                # Finalize current group
                fused.append(self._finalize_group(current_group))
                current_group = [line]

        # Finalize last group
        if current_group:
            fused.append(self._finalize_group(current_group))

        return fused

    def _finalize_group(self, group: List[dict]) -> dict:
        """
        Finalize a group of lines by summing emissivities.

        Parameters
        ----------
        group : List[dict]
            Group of line dicts

        Returns
        -------
        dict
            Fused line dict
        """
        # Sum emissivities
        total_emissivity = sum(line["avg_emissivity"] for line in group)

        # Position = wavelength of strongest line
        strongest = max(group, key=lambda x: x["avg_emissivity"])

        return {
            "transition": strongest["transition"],
            "avg_emissivity": total_emissivity,
            "wavelength_nm": strongest["wavelength_nm"],
            "n_fused": len(group),
        }

    def _match_lines(
        self, fused_lines: List[dict], peaks: List[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match theoretical lines to experimental peaks.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (matched_mask, wavelength_shifts, matched_peak_idx) where
            matched_mask is bool array, wavelength_shifts is float array of
            shifts in nm, and matched_peak_idx is int array (-1 if unmatched)
        """
        n = len(fused_lines)
        if not peaks or not fused_lines:
            return (
                np.zeros(n, dtype=bool),
                np.zeros(n),
                np.full(n, -1, dtype=int),
            )

        peak_wavelengths = np.array([p[1] for p in peaks])

        # Resolution element for matching window
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        # Estimate global wavelength offset from strongest peaks
        sorted_by_emissivity = sorted(
            fused_lines, key=lambda x: x["avg_emissivity"], reverse=True
        )
        top_lines = sorted_by_emissivity[: min(5, len(sorted_by_emissivity))]

        shifts = []
        for line in top_lines:
            wl_th = line["wavelength_nm"]
            distances = np.abs(peak_wavelengths - wl_th)
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist <= delta_lambda:  # within one resolution element
                    closest_idx = np.argmin(distances)
                    shifts.append(peak_wavelengths[closest_idx] - wl_th)

        global_shift = np.median(shifts) if shifts else 0.0

        matched_mask = np.zeros(n, dtype=bool)
        wavelength_shifts = np.zeros(n)
        matched_peak_idx = np.full(n, -1, dtype=int)

        for i, line in enumerate(fused_lines):
            wl_th = line["wavelength_nm"] + global_shift  # Apply correction

            # Find peaks within +/- delta_lambda
            distances = np.abs(peak_wavelengths - wl_th)
            within_window = distances <= delta_lambda

            if np.any(within_window):
                matched_mask[i] = True
                closest_idx = int(np.argmin(distances))
                matched_peak_idx[i] = closest_idx
                # Shift relative to uncorrected wavelength
                wavelength_shifts[i] = peak_wavelengths[closest_idx] - line["wavelength_nm"]

        # Enforce one-to-one: each experimental peak is assigned to at most
        # one theoretical line (highest emissivity wins).  This prevents a
        # single broad peak from "confirming" multiple theoretical lines,
        # which inflates k_rate at low resolving power.
        claimed_peaks: dict = {}  # peak_idx -> (line_idx, emissivity)
        for i in range(n):
            if not matched_mask[i]:
                continue
            pidx = int(matched_peak_idx[i])
            emiss = fused_lines[i]["avg_emissivity"]
            if pidx not in claimed_peaks or emiss > claimed_peaks[pidx][1]:
                if pidx in claimed_peaks:
                    old_i = claimed_peaks[pidx][0]
                    matched_mask[old_i] = False
                    wavelength_shifts[old_i] = 0.0
                    matched_peak_idx[old_i] = -1
                claimed_peaks[pidx] = (i, emiss)
            else:
                # Peak already claimed by a stronger line
                matched_mask[i] = False
                wavelength_shifts[i] = 0.0
                matched_peak_idx[i] = -1

        return matched_mask, wavelength_shifts, matched_peak_idx

    def _determine_emissivity_threshold(
        self, fused_lines: List[dict], matched_mask: np.ndarray
    ) -> float:
        """
        Determine emissivity threshold where detection rate > 50%.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines

        Returns
        -------
        float
            Log10 emissivity threshold
        """
        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])

        # Avoid log(0)
        emissivities = np.maximum(emissivities, 1e-100)

        log_emissivities = np.log10(emissivities)

        # Bin in log decades
        min_log = np.floor(np.min(log_emissivities))
        max_log = np.ceil(np.max(log_emissivities))
        n_bins = int(max_log - min_log) + 1

        if n_bins < 2:
            # Not enough dynamic range, return minimum
            return min_log

        bins = np.linspace(min_log, max_log, n_bins + 1)

        # Compute detection rate per bin
        bin_indices = np.digitize(log_emissivities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        detection_rates = []
        thresholds = []

        for bin_idx in range(n_bins):
            in_bin = bin_indices == bin_idx
            if np.sum(in_bin) > 0:
                detection_rate = np.sum(matched_mask & in_bin) / np.sum(in_bin)
                detection_rates.append(detection_rate)
                thresholds.append(bins[bin_idx])

        # Find threshold where detection_rate > 0.5
        detection_rates = np.array(detection_rates)
        thresholds = np.array(thresholds)

        above_50 = detection_rates > 0.5
        if np.any(above_50):
            # Return lowest threshold with >50% detection
            return thresholds[np.where(above_50)[0][0]]
        else:
            # No threshold meets criterion, return minimum
            return min_log

    def _compute_scores(
        self,
        fused_lines: List[dict],
        matched_mask: np.ndarray,
        matched_peak_idx: np.ndarray,
        wavelength_shifts: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        emissivity_threshold: float,
    ) -> Tuple[float, float, float, float, int]:
        """
        Compute k_sim, k_rate, k_shift scores, P_maj, and N_expected.

        Parameters
        ----------
        fused_lines : List[dict]
            Fused theoretical lines
        matched_mask : np.ndarray
            Boolean mask of matched lines
        matched_peak_idx : np.ndarray
            Index of matched peak per line (-1 if unmatched)
        wavelength_shifts : np.ndarray
            Wavelength shifts in nm
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks
        emissivity_threshold : float
            Log10 emissivity threshold

        Returns
        -------
        Tuple[float, float, float, float, int]
            (k_sim, k_rate, k_shift, P_maj, N_expected)
        """
        if not np.any(matched_mask):
            return 0.0, 0.0, 0.0, 0.5, 0

        emissivities = np.array([line["avg_emissivity"] for line in fused_lines])
        above_threshold = emissivities >= 10**emissivity_threshold

        # N_expected: ALL above-threshold theoretical lines (matched or not).
        # Used in k_det blend so k_sim always gets weight when many lines
        # are predicted but few matched (prevents N_X=1 singularity).
        N_expected = int(np.sum(above_threshold))

        # Filter to lines above threshold that are also matched
        matched_above = matched_mask & above_threshold
        n_matched_above = int(np.sum(matched_above))

        if n_matched_above == 0:
            return 0.0, 0.0, 0.0, 0.5, N_expected

        # Soft P_maj: weighted coverage of top-k strongest above-threshold
        # lines.  Binary P_maj (strongest matched → 1.0, else 0.5) causes
        # false negatives when the major line is obscured by matrix
        # emission (e.g. V in Ti6Al4V where Ti dominates).
        top_k = min(3, N_expected)
        if top_k > 0:
            above_emissivities = emissivities * above_threshold.astype(float)
            sorted_indices = np.argsort(above_emissivities)[::-1][:top_k]
            weights = np.sqrt(emissivities[sorted_indices])
            matched_weights = float(np.sum(weights * matched_above[sorted_indices]))
            total_weights = float(np.sum(weights))
            P_maj = (
                0.5 + 0.5 * (matched_weights / total_weights)
                if total_weights > 0
                else 0.5
            )
        else:
            P_maj = 0.5

        # k_rate: geometric mean of emissivity-weighted and count-based rates.
        # Emissivity-weighted alone lets a single high-emissivity match inflate
        # k_rate to ~1.0 even when most lines are undetected.  The geometric
        # mean with the count-based rate corrects this.
        total_emissivity_above = np.sum(emissivities[above_threshold])
        matched_emissivity = np.sum(emissivities[matched_above])

        if total_emissivity_above > 0 and N_expected > 0:
            k_rate_emissivity = matched_emissivity / total_emissivity_above
            k_rate_count = n_matched_above / N_expected
            k_rate = math.sqrt(k_rate_emissivity * k_rate_count)
        else:
            k_rate = 0.0

        # k_shift: wavelength match quality
        mean_wl = np.mean([line["wavelength_nm"] for line in fused_lines])
        delta_lambda = mean_wl / self.resolving_power

        shifts_matched = np.abs(wavelength_shifts[matched_above])
        if len(shifts_matched) > 0:
            mean_shift_frac = np.mean(shifts_matched) / delta_lambda
            k_shift = max(0.0, 1.0 - mean_shift_frac)
        else:
            k_shift = 0.0

        # k_sim: cosine similarity between theoretical and experimental intensities
        # Include ALL above-threshold lines; unmatched get experimental = 0,
        # penalising elements that predict lines the spectrum doesn't contain.
        # Use matched_peak_idx (from _match_lines) for consistent peak mapping.
        theoretical_intensities = []
        experimental_intensities = []
        unique_peak_set: set = set()

        for i in range(len(fused_lines)):
            if above_threshold[i]:
                theoretical_intensities.append(emissivities[i])
                if matched_above[i]:
                    pidx = matched_peak_idx[i]
                    experimental_intensities.append(intensity[peaks[pidx][0]])
                    unique_peak_set.add(pidx)
                else:
                    experimental_intensities.append(0.0)

        if len(theoretical_intensities) > 1:
            th_vec = np.array(theoretical_intensities)
            exp_vec = np.array(experimental_intensities)

            dot_product = np.dot(th_vec, exp_vec)
            norm_th = np.linalg.norm(th_vec)
            norm_exp = np.linalg.norm(exp_vec)

            if norm_th > 0 and norm_exp > 0:
                k_sim = dot_product / (norm_th * norm_exp)
                k_sim = max(0.0, min(1.0, k_sim))
            else:
                k_sim = 0.0
        else:
            # Single above-threshold line: cosine similarity undefined → 0
            k_sim = 0.0

        # Uniqueness penalty: many-to-one mapping lowers k_sim
        n_unique_peaks = len(unique_peak_set)
        if n_matched_above > 0:
            uniqueness_factor = n_unique_peaks / n_matched_above
            k_sim *= uniqueness_factor

        return k_sim, k_rate, k_shift, P_maj, N_expected

    def _compute_P_ab(self, element: str) -> float:
        """
        Compute crustal-abundance prior P_ab for an element.

        3-tier weighting (Noel et al. 2025):
        - ppm >= 100    → 1.0  (common, > 0.01%)
        - ppm >= 0.001  → 0.75 (intermediate)
        - ppm < 0.001   → 0.5  (rare)

        Parameters
        ----------
        element : str
            Element symbol

        Returns
        -------
        float
            P_ab weighting factor
        """
        log_ppm = self.CRUSTAL_ABUNDANCE_LOG_PPM.get(element, 0.0)
        ppm = 10**log_ppm
        if ppm >= 100:
            return 1.0
        elif ppm >= 1e-3:
            return 0.75
        else:
            return 0.5

    def _compute_fill_factor(
        self,
        peaks: List[Tuple[int, float]],
        wavelength: np.ndarray,
    ) -> float:
        """
        Compute spectral fill factor from merged peak-match windows.

        Each peak contributes an interval centered at its wavelength with half-width:
            chance_window_scale * (lambda / resolving_power)
        Overlapping intervals are merged before computing covered span fraction.

        Parameters
        ----------
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.
        wavelength : np.ndarray
            Full spectral wavelength axis in nm.

        Returns
        -------
        float
            Fraction of spectral span covered by merged intervals in [0, 1].
        """
        if len(peaks) == 0 or len(wavelength) < 2:
            return 0.0

        wl_min = float(np.min(wavelength))
        wl_max = float(np.max(wavelength))
        span = wl_max - wl_min
        if span <= 0:
            return 0.0

        intervals: List[Tuple[float, float]] = []
        for _, peak_wl in peaks:
            half_window = self.chance_window_scale * (peak_wl / self.resolving_power)
            if half_window <= 0:
                continue
            start = max(wl_min, peak_wl - half_window)
            end = min(wl_max, peak_wl + half_window)
            if end > start:
                intervals.append((start, end))

        if not intervals:
            return 0.0

        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        covered = sum(end - start for start, end in merged)
        return float(np.clip(covered / span, 0.0, 1.0))

    def _compute_random_match_significance(
        self,
        peaks: List[Tuple[int, float]],
        wavelength: np.ndarray,
        N_expected: int,
        N_matched: int,
    ) -> Tuple[float, float, float, float]:
        """
        Compute chance-coincidence significance from a binomial tail test.

        Parameters
        ----------
        peaks : List[Tuple[int, float]]
            Experimental peaks as (index, wavelength) tuples.
        wavelength : np.ndarray
            Full spectral wavelength axis in nm.
        N_expected : int
            Number of above-threshold theoretical lines.
        N_matched : int
            Number of above-threshold lines matched to peaks.

        Returns
        -------
        Tuple[float, float, float, float]
            (P_sig, fill_factor, p_chance, p_tail), where:
            - fill_factor is merged-window coverage fraction
            - p_chance is per-line random-match probability
            - p_tail = P(X >= N_matched | n=N_expected, p=p_chance)
            - P_sig = 1 - p_tail
        """
        fill_factor = self._compute_fill_factor(peaks, wavelength)
        p_chance = float(np.clip(fill_factor, 1e-6, 1.0 - 1e-6))

        if N_expected <= 0 or N_matched <= 0:
            return 1.0, fill_factor, p_chance, 1.0

        n_trials = max(N_expected, N_matched)
        n_success = min(N_matched, n_trials)

        p_tail = float(binom.sf(n_success - 1, n_trials, p_chance))
        P_sig = float(np.clip(1.0 - p_tail, 0.0, 1.0))

        return P_sig, fill_factor, p_chance, p_tail

    def _decide(
        self,
        k_sim: float,
        k_rate: float,
        k_shift: float,
        N_expected: int,
        intensity: np.ndarray,
        peaks: List[Tuple[int, float]],
        element: str = "",
        P_maj: float = 0.5,
        P_sig: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute detection score k_det and confidence level CL.

        Parameters
        ----------
        k_sim : float
            Similarity score
        k_rate : float
            Detection rate score
        k_shift : float
            Wavelength shift score
        N_expected : int
            Number of above-threshold theoretical lines (matched or not).
            Using expected count (not matched count) prevents the N=1
            singularity where k_sim gets zero weight from a single
            coincidental match.
        intensity : np.ndarray
            Experimental intensity array
        peaks : List[Tuple[int, float]]
            Experimental peaks
        element : str
            Element symbol (for crustal abundance weighting)
        P_maj : float
            Major-line coverage factor (0.5–1.0), computed from top-k
            strongest theoretical lines
        P_sig : float
            Statistical significance factor against random coincidence
            (computed from a binomial survival test)

        Returns
        -------
        Tuple[float, float]
            (k_det, CL) detection score and confidence level
        """
        # k_sim gate: require minimum intensity-pattern correlation.
        # Pure wavelength coincidences with uncorrelated intensities should
        # not yield a detection.  For N_expected >= 2, require k_sim >= 0.15.
        # For N_expected == 1, k_sim is 0 by construction (cosine similarity
        # undefined for single point), so rely on N_penalty instead.
        if N_expected >= 2 and k_sim < 0.15:
            return 0.0, 0.0

        # k_det formula — uses N_expected (above-threshold count) so that
        # k_sim always receives weight when many lines are predicted.
        if N_expected > 1:
            k_det = k_rate * (
                (1.0 / N_expected) * k_shift
                + ((N_expected - 1.0) / N_expected) * k_sim
            )
        elif N_expected == 1:
            # Single above-threshold line: statistically insufficient for
            # pattern confirmation.  Apply a 50/50 blend of k_shift and
            # k_sim (which is 0 in this regime) so a single coincidental
            # wavelength match cannot produce a near-perfect score.
            k_det = k_rate * (0.5 * k_shift + 0.5 * k_sim)
        elif k_rate > 0:
            # Partial credit: some lines matched but below emissivity threshold
            k_det = k_rate * 0.3
        else:
            k_det = 0.0

        # P_SNR: approximate SNR quality
        if len(peaks) > 0:
            peak_intensities = [intensity[p[0]] for p in peaks]
            median_peak = np.median(peak_intensities)
            noise_estimate = np.median(np.abs(intensity - np.median(intensity))) * 1.4826
            snr_estimate = median_peak / max(noise_estimate, 1e-10)
            P_SNR = min(1.0, snr_estimate / 10.0)
        else:
            P_SNR = 0.5

        # Fix 4: P_ab — crustal abundance prior
        P_ab = self._compute_P_ab(element)

        # N_penalty: penalize sparse spectral evidence.  Elements with very
        # few above-threshold lines in the spectral window cannot be confirmed
        # by pattern matching; a single coincidental wavelength match should
        # not yield a high confidence level.
        if N_expected <= 1:
            N_penalty = 0.2
        elif N_expected == 2:
            N_penalty = 0.5
        else:
            N_penalty = 1.0

        # Confidence level:
        # CL = k_det × P_SNR × P_maj × P_ab × N_penalty × P_sig
        CL = k_det * P_SNR * P_maj * P_ab * N_penalty * P_sig

        return k_det, CL
