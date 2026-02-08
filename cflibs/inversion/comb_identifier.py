"""
Comb template correlation algorithm for element identification.

Based on Gajarska et al. (2024), "Identification of elements in LIBS spectra
using the comb template matching method." This method uses triangular templates
to correlate with spectral peaks, treating atomic spectral lines as teeth in a comb.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import pearsonr

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.core.logging_config import get_logger

logger = get_logger(__name__)


class CombIdentifier:
    """
    Automated element identification using comb template correlation.

    This algorithm identifies elements by correlating triangular templates (teeth)
    with spectral peaks at known transition wavelengths. For each element, it
    computes a fingerprint score based on the correlation strength of active teeth.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Database containing atomic transitions
    baseline_window_nm : float, optional
        Window size for moving median baseline in nm (default: 10.0)
    threshold_percentile : float, optional
        Percentile for peak detection threshold (default: 85.0)
    min_correlation : float, optional
        Minimum correlation to count a tooth as active (default: 0.5)
    max_shift_pts : int, optional
        Maximum shift in data points for template matching (default: 5)
    min_width_pts : int, optional
        Minimum tooth width in data points (default: 3)
    max_width_factor : float, optional
        Maximum width as fraction of resolution element (default: 1.0)
    elements : List[str], optional
        List of elements to search for (default: None means all in database)

    Attributes
    ----------
    atomic_db : AtomicDatabase
        Atomic database instance
    baseline_window_nm : float
        Baseline window size in nm
    threshold_percentile : float
        Peak detection threshold percentile
    min_correlation : float
        Minimum tooth correlation threshold
    max_shift_pts : int
        Maximum template shift in points
    min_width_pts : int
        Minimum tooth width in points
    max_width_factor : float
        Maximum width scaling factor
    elements : Optional[List[str]]
        Elements to search (None = all)

    References
    ----------
    Gajarska et al. (2024), "Identification of elements in LIBS spectra
    using the comb template matching method."
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        baseline_window_nm: float = 10.0,
        threshold_percentile: float = 85.0,
        min_correlation: float = 0.5,
        max_shift_pts: int = 5,
        min_width_pts: int = 3,
        max_width_factor: float = 1.0,
        elements: Optional[List[str]] = None,
    ):
        """
        Initialize the CombIdentifier with its atomic transition database and matching configuration.
        
        Parameters:
            atomic_db (AtomicDatabase): Source of atomic transitions used to build element templates.
            baseline_window_nm (float): Window size in nanometers for moving-median baseline estimation.
            threshold_percentile (float): Percentile of positive residuals used to set the peak detection threshold.
            min_correlation (float): Minimum Pearson correlation required for a template tooth to be considered active.
            max_shift_pts (int): Maximum allowed shift (in data points) when aligning templates to the spectrum.
            min_width_pts (int): Minimum triangular template width in data points to consider during matching.
            max_width_factor (float): Maximum allowed tooth width expressed as a fraction of the instrument resolution element.
            elements (Optional[List[str]]): Optional list of element symbols to restrict the search; if None, defaults are used.
        """
        self.atomic_db = atomic_db
        self.baseline_window_nm = baseline_window_nm
        self.threshold_percentile = threshold_percentile
        self.min_correlation = min_correlation
        self.max_shift_pts = max_shift_pts
        self.min_width_pts = min_width_pts
        self.max_width_factor = max_width_factor
        self.elements = elements

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """
        Identify atomic elements present in a spectrum by correlating triangular comb templates with spectral peaks.
        
        Performs baseline and threshold estimation, correlates per-element template "teeth" with the baseline-corrected spectrum across allowable shifts and widths, computes per-element fingerprint scores, analyzes inter-element interferences, and returns a structured ElementIdentificationResult summarizing detections and peak matches.
        
        Parameters:
            wavelength (np.ndarray): Wavelength array in nanometers.
            intensity (np.ndarray): Intensity array (same shape as `wavelength`).
        
        Returns:
            ElementIdentificationResult: Result containing detected and rejected element identifications, all per-element details (matched/unmatched lines, scores, metadata), experimental peaks and peak-match counts, algorithm name ("comb"), and the parameters used for the identification run.
        """
        logger.info(
            f"Starting comb identification on spectrum: "
            f"{wavelength[0]:.1f}-{wavelength[-1]:.1f} nm, {len(wavelength)} points"
        )

        # Step 1: Estimate baseline and threshold
        baseline, threshold = self._estimate_baseline_threshold(wavelength, intensity)
        logger.debug(f"Baseline estimated, threshold={threshold:.2f}")

        # Step 2: Determine elements to search
        if self.elements is None:
            # TODO: Get all elements from database
            # For now, use common LIBS elements
            elements_to_search = ["Fe", "Cu", "Al", "Ca", "Mg", "Ti", "H"]
        else:
            elements_to_search = self.elements

        # Step 3: For each element, get lines and correlate teeth
        element_teeth: Dict[str, List[dict]] = {}
        element_identifications = []

        for element in elements_to_search:
            # Get transitions for this element in wavelength range
            transitions = self._get_element_lines(element, wavelength[0], wavelength[-1])

            if not transitions:
                logger.debug(f"No transitions found for {element}")
                continue

            # Correlate each transition (tooth)
            teeth = []
            matched_lines = []
            unmatched_lines = []

            for trans in transitions:
                tooth_result = self._correlate_tooth(
                    wavelength, intensity, baseline, trans.wavelength_nm, threshold
                )
                tooth_result["transition"] = trans
                teeth.append(tooth_result)

                if tooth_result["active"]:
                    # Create IdentifiedLine for active tooth
                    matched_lines.append(
                        IdentifiedLine(
                            wavelength_exp_nm=tooth_result["center_nm"]
                            + tooth_result["best_shift"] * np.median(np.diff(wavelength)),
                            wavelength_th_nm=trans.wavelength_nm,
                            element=element,
                            ionization_stage=trans.ionization_stage,
                            intensity_exp=intensity[
                                int(np.argmin(np.abs(wavelength - tooth_result["center_nm"])))
                            ],
                            emissivity_th=0.0,
                            transition=trans,
                            correlation=tooth_result["best_correlation"],
                            is_interfered=False,
                            interfering_elements=[],
                        )
                    )
                else:
                    unmatched_lines.append(trans)

            # Compute fingerprint for this element
            fingerprint = self._compute_fingerprint(teeth)
            element_teeth[element] = teeth

            # Create ElementIdentification
            detected = fingerprint >= self.min_correlation
            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=fingerprint,
                confidence=fingerprint,
                n_matched_lines=len(matched_lines),
                n_total_lines=len(transitions),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={
                    "fingerprint": fingerprint,
                    "n_active_teeth": sum(1 for t in teeth if t["active"]),
                    "n_total_teeth": len(teeth),
                },
            )
            element_identifications.append(element_id)

        # Step 4: Analyze interferences across all elements
        element_teeth = self._analyze_interferences(element_teeth)

        # Update interfered status in element identifications
        for element_id in element_identifications:
            element = element_id.element
            if element in element_teeth:
                for line in element_id.matched_lines:
                    # Check if this line's wavelength is interfered
                    for tooth in element_teeth[element]:
                        if (
                            abs(tooth["center_nm"] - line.wavelength_th_nm) < 0.01
                            and "interfering_elements" in tooth
                        ):
                            line.is_interfered = True
                            line.interfering_elements = tooth["interfering_elements"]

        # Step 5: Split into detected and rejected
        detected_elements = [e for e in element_identifications if e.detected]
        rejected_elements = [e for e in element_identifications if not e.detected]

        # Step 6: Identify experimental peaks (simple threshold-based for now)
        residual = intensity - baseline
        peak_mask = residual > threshold
        peak_indices = np.where(peak_mask)[0]
        experimental_peaks = [(i, wavelength[i]) for i in peak_indices]

        # Count matched peaks (peaks that have at least one identified line)
        matched_peak_wavelengths = set()
        for element_id in detected_elements:
            for line in element_id.matched_lines:
                matched_peak_wavelengths.add(line.wavelength_exp_nm)

        n_matched_peaks = sum(
            1
            for _, wl in experimental_peaks
            if any(abs(wl - mwl) < 0.1 for mwl in matched_peak_wavelengths)
        )

        result = ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=element_identifications,
            experimental_peaks=experimental_peaks,
            n_peaks=len(experimental_peaks),
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=len(experimental_peaks) - n_matched_peaks,
            algorithm="comb",
            parameters={
                "baseline_window_nm": self.baseline_window_nm,
                "threshold_percentile": self.threshold_percentile,
                "min_correlation": self.min_correlation,
                "max_shift_pts": float(self.max_shift_pts),
                "min_width_pts": float(self.min_width_pts),
                "max_width_factor": self.max_width_factor,
            },
        )

        logger.info(
            f"Comb identification complete: {len(detected_elements)} detected, "
            f"{len(rejected_elements)} rejected"
        )

        return result

    def _get_element_lines(self, element: str, wl_min: float, wl_max: float) -> List[Transition]:
        """
        Retrieve atomic transitions for the given element within the specified wavelength interval.
        
        Parameters:
            element (str): Element symbol (e.g., "Fe").
            wl_min (float): Minimum wavelength in nanometers.
            wl_max (float): Maximum wavelength in nanometers.
        
        Returns:
            List[Transition]: Transitions for the element (all ionization stages) whose wavelengths fall within [wl_min, wl_max].
        """
        # Get all ionization stages for this element
        transitions = self.atomic_db.get_transitions(
            element, wavelength_min=wl_min, wavelength_max=wl_max
        )
        return transitions

    def _estimate_baseline_threshold(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate a moving-median baseline and compute a peak-detection threshold from positive residuals.
        
        Parameters:
            wavelength (np.ndarray): Wavelength array in nanometers; used to derive the median wavelength spacing for window sizing.
            intensity (np.ndarray): Measured intensity array matching `wavelength`.
        
        Returns:
            baseline (np.ndarray): Moving-median baseline computed with a window size derived from `baseline_window_nm`.
            threshold (float): Peak detection threshold equal to the `threshold_percentile` percentile of positive residuals (intensity - baseline); `0.0` if no positive residuals are present.
        """
        # Compute window size in points
        dwl_median = np.median(np.diff(wavelength))
        window_pts = int(self.baseline_window_nm / dwl_median)
        # Ensure odd window size for median filter
        if window_pts % 2 == 0:
            window_pts += 1
        window_pts = max(3, window_pts)  # Minimum window of 3

        # Compute moving median baseline
        baseline = median_filter(intensity, size=window_pts)

        # Compute residual
        residual = intensity - baseline

        # Threshold based on percentile of positive residuals
        positive_residual = residual[residual > 0]
        if len(positive_residual) > 0:
            threshold = np.percentile(positive_residual, self.threshold_percentile)
        else:
            threshold = 0.0

        return baseline, threshold

    def _build_triangular_template(self, width_pts: int) -> np.ndarray:
        """
        Create a symmetric isosceles triangular template of the given width in data points.
        
        If `width_pts` is even it will be increased to the next odd integer so the triangle has a single center sample.
        
        Parameters:
            width_pts (int): Desired template width in data points (will be adjusted to odd if even).
        
        Returns:
            np.ndarray: 1D array containing a centered triangular shape normalized so its maximum equals 1.0.
        """
        if width_pts % 2 == 0:
            width_pts += 1  # Ensure odd width

        template = np.zeros(width_pts)
        center_idx = width_pts // 2

        # Build triangle
        for i in range(width_pts):
            distance = abs(i - center_idx)
            template[i] = 1.0 - (distance / (center_idx + 1))

        # Normalize to max=1.0
        template = template / np.max(template)

        return template

    def _correlate_tooth(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        baseline: np.ndarray,
        center_nm: float,
        threshold: float,
    ) -> dict:
        """
        Find the best-matching triangular template for a single spectral tooth centered at a given wavelength by searching allowed shifts and widths.
        
        Searches over template widths and integer data-point shifts around center_nm, compares each template to the baseline-corrected data segment, and selects the width and shift that maximize the Pearson correlation. Marks the tooth as active when the best correlation meets or exceeds the configured minimum correlation.
        
        Parameters:
            wavelength (np.ndarray): Wavelength array in nanometers.
            intensity (np.ndarray): Measured intensity array.
            baseline (np.ndarray): Estimated baseline array to subtract from intensity.
            center_nm (float): Nominal center wavelength of the tooth in nanometers.
            threshold (float): Peak detection threshold (used by callers for gating; not modified).
        
        Returns:
            dict: Mapping with keys:
                - "center_nm" (float): The input center wavelength.
                - "best_correlation" (float): Highest Pearson correlation found (0–1, or negative if applicable).
                - "best_shift" (int): Integer shift in data points that produced best_correlation.
                - "best_width" (int): Template width in data points that produced best_correlation.
                - "active" (bool): `true` if best_correlation >= configured minimum correlation, `false` otherwise.
        """
        # Find nearest index to center_nm
        center_idx = np.argmin(np.abs(wavelength - center_nm))

        # Estimate resolution element from wavelength spacing
        dwl = np.median(np.diff(wavelength))
        # Assume typical LIBS resolution of ~0.1 nm
        resolution_nm = 0.1
        max_width_pts = int((resolution_nm * self.max_width_factor) / dwl)
        max_width_pts = max(self.min_width_pts, max_width_pts)

        best_correlation = -1.0
        best_shift = 0
        best_width = self.min_width_pts

        # Search over widths (odd values only)
        for width in range(self.min_width_pts, max_width_pts + 1, 2):
            # Search over shifts
            for shift in range(-self.max_shift_pts, self.max_shift_pts + 1):
                shifted_idx = center_idx + shift

                # Extract data segment
                half_width = width // 2
                start_idx = max(0, shifted_idx - half_width)
                end_idx = min(len(intensity), shifted_idx + half_width + 1)

                if end_idx - start_idx < self.min_width_pts:
                    continue

                data_segment = intensity[start_idx:end_idx] - baseline[start_idx:end_idx]

                # Build template
                template = self._build_triangular_template(width)

                # Ensure same length
                if len(data_segment) != len(template):
                    # Truncate or skip
                    continue

                # Compute Pearson correlation
                if np.std(data_segment) < 1e-10 or np.std(template) < 1e-10:
                    correlation = 0.0
                else:
                    correlation, _ = pearsonr(data_segment, template)
                    if np.isnan(correlation):
                        correlation = 0.0

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_shift = shift
                    best_width = width

        # Determine if tooth is active
        active = best_correlation >= self.min_correlation

        return {
            "center_nm": center_nm,
            "best_correlation": best_correlation,
            "best_shift": best_shift,
            "best_width": best_width,
            "active": active,
        }

    def _analyze_interferences(
        self, element_teeth: Dict[str, List[dict]], wl_tolerance_nm: float = 0.1
    ) -> Dict[str, List[dict]]:
        """
        Analyze interferences between elements based on overlapping teeth.

        Parameters
        ----------
        element_teeth : Dict[str, List[dict]]
            Dictionary mapping element symbols to lists of tooth results
        wl_tolerance_nm : float, optional
            Wavelength tolerance for interference detection (default: 0.1 nm)

        Returns
        -------
        Dict[str, List[dict]]
            Updated element_teeth with interference information added
        """
        # Build list of all active teeth with element labels
        all_active_teeth = []
        for element, teeth in element_teeth.items():
            for tooth in teeth:
                if tooth["active"]:
                    all_active_teeth.append((element, tooth))

        # Check each active tooth against all others
        for i, (element_i, tooth_i) in enumerate(all_active_teeth):
            interfering = []
            for j, (element_j, tooth_j) in enumerate(all_active_teeth):
                if i != j and element_i != element_j:
                    # Check wavelength overlap
                    if abs(tooth_i["center_nm"] - tooth_j["center_nm"]) < wl_tolerance_nm:
                        interfering.append(element_j)

            if interfering:
                tooth_i["is_interfered"] = True
                tooth_i["interfering_elements"] = interfering
            else:
                tooth_i["is_interfered"] = False
                tooth_i["interfering_elements"] = []

        return element_teeth

    def _compute_fingerprint(self, teeth: List[dict]) -> float:
        """
        Compute an element fingerprint as the mean correlation of the active template teeth.
        
        Parameters:
            teeth (List[dict]): List of tooth result dictionaries (as returned by _correlate_tooth). Each dict is expected to contain at least the keys `"active"` (bool) and `"best_correlation"` (float).
        
        Returns:
            float: Mean of `best_correlation` across teeth where `"active"` is true; returns 0.0 if no teeth are active.
        """
        active_teeth = [t for t in teeth if t["active"]]
        if not active_teeth:
            return 0.0

        correlations = [t["best_correlation"] for t in active_teeth]
        fingerprint = np.mean(correlations)

        return fingerprint