"""
Line detection utilities for CF-LIBS inversion.

Provides a lightweight peak detection + line matching pipeline to convert
raw spectra into LineObservation objects for classic CF-LIBS solvers.

Uses the canonical preprocessing pipeline (baseline subtraction, noise
estimation, prominence-based detection) from ``cflibs.inversion.preprocessing``
to ensure consistency across all CF-LIBS modules.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation
from cflibs.inversion.preprocessing import detect_peaks_auto

logger = get_logger("inversion.line_detection")

# Minimum Einstein A coefficient — lines weaker than this are undetectable
# in typical LIBS conditions and should not participate in matching.
_MIN_AKI = 1e4  # s^-1

# Reference temperature for emissivity-weighted matching (K)
_REFERENCE_T_K = 10000.0


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
    resolving_power: Optional[float] = None,
    wavelength_tolerance_nm: float = 0.1,
    min_peak_height: float = 0.01,
    peak_width_nm: float = 0.2,
    min_relative_intensity: Optional[float] = None,
    ground_state_threshold_ev: float = 0.1,
    reference_temperature_K: float = _REFERENCE_T_K,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
) -> LineDetectionResult:
    """
    Detect spectral peaks and match them to known atomic transitions.

    Uses the canonical preprocessing pipeline: median-filter baseline
    estimation, sigma-clipped MAD noise estimation, and prominence-based
    peak detection with cosmic-ray rejection.

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
    resolving_power : float, optional
        Instrument resolving power (lambda/delta_lambda). When provided,
        the matching tolerance scales as wavelength/resolving_power instead
        of using the fixed ``wavelength_tolerance_nm``.
    wavelength_tolerance_nm : float
        Fallback matching tolerance in nm (used only when resolving_power
        is not provided). Default 0.1.
    min_peak_height : float
        Deprecated — kept for API compatibility. The canonical pipeline
        uses noise-scaled thresholds instead.
    peak_width_nm : float
        Expected peak width for integration window (nm). Default 0.2.
    min_relative_intensity : float, optional
        Minimum relative intensity threshold for database lines
    ground_state_threshold_ev : float
        Lower-level energy threshold for resonance detection
    reference_temperature_K : float
        Reference temperature for emissivity-weighted matching (K)
    threshold_factor : float
        Peak height threshold in noise units (default 4.0)
    prominence_factor : float
        Peak prominence threshold in noise units (default 1.5)

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

    # Use the canonical preprocessing pipeline for peak detection
    peaks, baseline, _noise = detect_peaks_auto(
        wavelength,
        intensity,
        threshold_factor=threshold_factor,
        prominence_factor=prominence_factor,
        resolving_power=resolving_power,
    )

    observations: List[LineObservation] = []
    resonance_lines: Set[Tuple[str, int, float]] = set()
    seen_keys: Set[Tuple[str, int, float]] = set()

    matched_peaks = 0

    wl_step = _estimate_wl_step(wavelength)
    half_width_px = max(int((peak_width_nm / max(wl_step, 1e-9)) / 2), 1)

    for peak_idx, peak_wl in peaks:
        # Resolution-aware tolerance: use resolving_power if available
        tolerance = _get_tolerance(peak_wl, resolving_power, wavelength_tolerance_nm)

        transition = _match_transition(peak_wl, transitions, tolerance, reference_temperature_K)
        if transition is None:
            continue

        key = (transition.element, transition.ionization_stage, transition.wavelength_nm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        matched_peaks += 1

        # Integrate baseline-subtracted intensity
        start_idx = max(0, peak_idx - half_width_px)
        end_idx = min(len(intensity), peak_idx + half_width_px + 1)
        segment_wl = wavelength[start_idx:end_idx]
        segment_baseline = baseline[start_idx:end_idx]
        segment_corrected = intensity[start_idx:end_idx] - segment_baseline

        line_area = float(np.trapezoid(np.maximum(segment_corrected, 0.0), segment_wl))
        if line_area <= 0:
            continue

        # Poisson noise approximation on raw counts for uncertainty.
        # Uses rectangular sum (sqrt(sum(counts)) * wl_step) rather than
        # trapezoidal integration because the Poisson shot noise dominates
        # and the rectangular approximation is well within the overall
        # uncertainty budget for LIBS measurements.
        raw_counts = np.maximum(intensity[start_idx:end_idx], 1.0)
        line_unc = float(np.sqrt(np.sum(raw_counts)) * wl_step)

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


def _get_tolerance(
    peak_wl: float,
    resolving_power: Optional[float],
    fallback_nm: float,
) -> float:
    """Return the wavelength matching tolerance in nm.

    When ``resolving_power`` is provided, the tolerance is the resolution
    element at the peak wavelength.  Otherwise the fixed fallback is used.
    """
    if resolving_power is not None and resolving_power > 0:
        return peak_wl / resolving_power
    return fallback_nm


def _load_transitions(
    atomic_db: AtomicDatabase,
    elements: List[str],
    wavelength_min: float,
    wavelength_max: float,
    min_relative_intensity: Optional[float],
) -> List[Transition]:
    transitions: List[Transition] = []
    for element in elements:
        raw = atomic_db.get_transitions(
            element,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            min_relative_intensity=min_relative_intensity,
        )
        # Pre-filter: discard lines with negligible A_ki
        transitions.extend(t for t in raw if t.A_ki >= _MIN_AKI)
    return transitions


def _match_transition(
    peak_wavelength: float,
    transitions: List[Transition],
    tolerance_nm: float,
    reference_temperature_K: float = _REFERENCE_T_K,
) -> Optional[Transition]:
    """Match a peak to the best candidate transition using physics-aware scoring.

    Candidates within the wavelength tolerance are scored by an emissivity
    proxy (g_k * A_ki * exp(-E_k / kT)) weighted by proximity.  This
    favours strong, physically plausible lines over weak coincidences.
    """
    if tolerance_nm <= 0:
        return None

    kT = KB_EV * reference_temperature_K
    if kT <= 0:
        kT = KB_EV * _REFERENCE_T_K

    best_match: Optional[Transition] = None
    best_score = -1.0

    for transition in transitions:
        distance = abs(transition.wavelength_nm - peak_wavelength)
        if distance > tolerance_nm:
            continue

        # Emissivity proxy: g_k * A_ki * exp(-E_k / kT)
        emissivity = transition.g_k * transition.A_ki * np.exp(-transition.E_k_ev / kT)

        # Proximity weight: linearly penalise wavelength offset
        proximity = 1.0 - (distance / tolerance_nm)

        score = emissivity * proximity
        if score > best_score:
            best_score = score
            best_match = transition

    return best_match


def _estimate_wl_step(wavelength: np.ndarray) -> float:
    if wavelength.size < 2:
        return 1.0
    diffs = np.diff(wavelength)
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else 1.0
