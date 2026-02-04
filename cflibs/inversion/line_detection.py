"""
Line detection utilities for CF-LIBS inversion.

Provides a lightweight peak detection + line matching pipeline to convert
raw spectra into LineObservation objects for classic CF-LIBS solvers.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation

logger = get_logger("inversion.line_detection")

try:
    from scipy.signal import find_peaks

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    find_peaks = None


@dataclass
class LineDetectionResult:
    """Result of line detection and matching."""

    observations: List[LineObservation]
    resonance_lines: Set[Tuple[str, int, float]]
    total_peaks: int
    matched_peaks: int
    unmatched_peaks: int
    warnings: List[str] = field(default_factory=list)


def detect_line_observations(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_tolerance_nm: float = 0.1,
    min_peak_height: float = 0.01,
    peak_width_nm: float = 0.2,
    min_relative_intensity: Optional[float] = None,
    ground_state_threshold_ev: float = 0.1,
) -> LineDetectionResult:
    """
    Detect spectral peaks and match them to known atomic transitions.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength axis in nm (monotonic)
    intensity : np.ndarray
        Spectral intensity values
    atomic_db : AtomicDatabase
        Atomic database instance
    elements : List[str]
        Elements to match against
    wavelength_tolerance_nm : float
        Matching tolerance for known lines in nm
    min_peak_height : float
        Minimum peak height as fraction of max intensity
    peak_width_nm : float
        Expected peak width for integration (nm)
    min_relative_intensity : float, optional
        Minimum relative intensity threshold for database lines
    ground_state_threshold_ev : float
        Lower-level energy threshold for resonance detection

    Returns
    -------
    LineDetectionResult
        Detected line observations and resonance set
    """
    if wavelength.size == 0 or intensity.size == 0:
        return LineDetectionResult([], set(), 0, 0, 0, ["empty_spectrum"])

    if wavelength.size != intensity.size:
        raise ValueError("Wavelength and intensity arrays must be the same length")

    if not elements:
        return LineDetectionResult([], set(), 0, 0, 0, ["no_elements_specified"])

    wl_min = float(np.min(wavelength))
    wl_max = float(np.max(wavelength))

    transitions = _load_transitions(
        atomic_db,
        elements,
        wavelength_min=wl_min,
        wavelength_max=wl_max,
        min_relative_intensity=min_relative_intensity,
    )

    if not transitions:
        return LineDetectionResult([], set(), 0, 0, 0, ["no_transitions_found"])

    peaks = _find_peaks(wavelength, intensity, min_peak_height, peak_width_nm)

    observations: List[LineObservation] = []
    resonance_lines: Set[Tuple[str, int, float]] = set()
    seen_keys: Set[Tuple[str, int, float]] = set()

    matched_peaks = 0

    wl_step = _estimate_wl_step(wavelength)
    half_width_px = max(int((peak_width_nm / max(wl_step, 1e-9)) / 2), 1)

    for peak_idx, peak_wl in peaks:
        transition = _match_transition(peak_wl, transitions, wavelength_tolerance_nm)
        if transition is None:
            continue

        key = (transition.element, transition.ionization_stage, transition.wavelength_nm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        matched_peaks += 1

        start_idx = max(0, peak_idx - half_width_px)
        end_idx = min(len(intensity), peak_idx + half_width_px + 1)
        segment_wl = wavelength[start_idx:end_idx]
        segment_intensity = intensity[start_idx:end_idx]

        line_area = float(np.trapezoid(segment_intensity, segment_wl))
        line_area = max(line_area, float(segment_intensity.max()))

        # Poisson noise approximation for integrated intensity
        counts = np.maximum(segment_intensity, 1.0)
        line_unc = float(np.sqrt(np.sum(counts)) * wl_step)
        if line_area <= 0:
            continue

        observations.append(
            LineObservation(
                wavelength_nm=float(transition.wavelength_nm),
                intensity=line_area,
                intensity_uncertainty=max(line_unc, 1e-6),
                element=transition.element,
                ionization_stage=transition.ionization_stage,
                E_k_ev=transition.E_k_ev,
                g_k=transition.g_k,
                A_ki=transition.A_ki,
            )
        )

        is_resonance = transition.is_resonance
        if is_resonance is None:
            is_resonance = transition.E_i_ev < ground_state_threshold_ev
        if is_resonance:
            resonance_lines.add(key)

    total_peaks = len(peaks)
    unmatched_peaks = max(total_peaks - matched_peaks, 0)

    warnings: List[str] = []
    if matched_peaks == 0 and total_peaks > 0:
        warnings.append("no_peaks_matched")
    if matched_peaks == 0 and total_peaks == 0:
        warnings.append("no_peaks_detected")

    return LineDetectionResult(
        observations=observations,
        resonance_lines=resonance_lines,
        total_peaks=total_peaks,
        matched_peaks=matched_peaks,
        unmatched_peaks=unmatched_peaks,
        warnings=warnings,
    )


def _load_transitions(
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_min: float,
    wavelength_max: float,
    min_relative_intensity: Optional[float],
) -> List[Transition]:
    transitions: List[Transition] = []
    for element in elements:
        transitions.extend(
            atomic_db.get_transitions(
                element,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                min_relative_intensity=min_relative_intensity,
            )
        )
    return transitions


def _match_transition(
    peak_wavelength: float,
    transitions: List[Transition],
    tolerance_nm: float,
) -> Optional[Transition]:
    best_match = None
    best_distance = float("inf")
    for transition in transitions:
        distance = abs(transition.wavelength_nm - peak_wavelength)
        if distance <= tolerance_nm and distance < best_distance:
            best_match = transition
            best_distance = distance
    return best_match


def _estimate_wl_step(wavelength: np.ndarray) -> float:
    if wavelength.size < 2:
        return 1.0
    diffs = np.diff(wavelength)
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else 1.0


def _find_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    min_peak_height: float,
    peak_width_nm: float,
) -> List[Tuple[int, float]]:
    max_intensity = float(np.max(intensity))
    if max_intensity <= 0:
        return []

    normalized = intensity / max_intensity
    threshold = max(min_peak_height, 0.0)

    if HAS_SCIPY and find_peaks is not None:
        wl_step = _estimate_wl_step(wavelength)
        min_distance_px = max(int(peak_width_nm / max(wl_step, 1e-9)), 1)
        peak_indices, _ = find_peaks(
            normalized,
            height=threshold,
            distance=min_distance_px,
            prominence=threshold / 2.0,
        )
        return [(int(idx), float(wavelength[idx])) for idx in peak_indices]

    # Simple fallback: local maxima above threshold
    peaks: List[Tuple[int, float]] = []
    for i in range(1, len(intensity) - 1):
        if normalized[i] >= threshold and intensity[i] > intensity[i - 1] and intensity[i] > intensity[i + 1]:
            peaks.append((i, float(wavelength[i])))
    return peaks
