"""Shared preprocessing for element identification algorithms.

Provides the canonical peak detection pipeline for all CF-LIBS modules:
baseline estimation, noise estimation, peak detection with cosmic-ray
rejection, and robust normalization.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import median_filter
from typing import List, Optional, Tuple


def estimate_baseline(
    wavelength: np.ndarray, intensity: np.ndarray, window_nm: float = 10.0
) -> np.ndarray:
    """Robust baseline estimation via median filter.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    window_nm : float
        Filter window width in nm (default 10.0)

    Returns
    -------
    np.ndarray
        Estimated baseline
    """
    if wavelength.size < 2:
        return intensity.copy()
    spacing = np.median(np.diff(wavelength))
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1e-10
    window_pts = max(3, int(window_nm / spacing))
    if window_pts % 2 == 0:
        window_pts += 1  # ensure odd
    return median_filter(intensity, size=window_pts)


def estimate_noise(intensity: np.ndarray, baseline: np.ndarray) -> float:
    """Iterative sigma-clipped MAD noise estimation.

    Uses 3 iterations of 3-sigma clipping to remove peak contributions
    before computing noise. Critical for LIBS spectra where raw MAD
    overestimates noise due to emission peaks and continuum.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array
    baseline : np.ndarray
        Baseline estimate

    Returns
    -------
    float
        Noise level (sigma)
    """
    residuals = intensity - baseline
    for _ in range(3):
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        sigma = mad * 1.4826
        if sigma < 1e-10:
            break
        mask = np.abs(residuals - med) < 3.0 * sigma
        if np.sum(mask) < 10:
            break
        residuals = residuals[mask]
    # Final estimate
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return mad * 1.4826


def detect_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    baseline: np.ndarray,
    noise: float,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
    min_distance_nm: Optional[float] = None,
    min_width_pts: int = 2,
) -> List[Tuple[int, float]]:
    """Unified peak detection above baseline.

    This is the canonical peak detection function for all CF-LIBS modules.
    It operates on baseline-subtracted intensity with noise-scaled thresholds,
    optional minimum peak separation, and cosmic-ray rejection via minimum
    peak width.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    baseline : np.ndarray
        Baseline estimate
    noise : float
        Noise level from estimate_noise
    threshold_factor : float
        Peak height threshold = noise * threshold_factor (default 4.0)
    prominence_factor : float
        Peak prominence threshold = noise * prominence_factor (default 1.5)
    min_distance_nm : float, optional
        Minimum distance between peaks in nm. If None, no distance
        constraint is applied. For resolution-aware detection, pass
        the instrument resolution element (wavelength / resolving_power).
    min_width_pts : int
        Minimum peak width in data points at half-maximum (default 2).
        Peaks narrower than this are rejected as cosmic-ray artifacts.
        Physical emission lines span at least the instrument resolution.

    Returns
    -------
    List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples for detected peaks
    """
    corrected = intensity - baseline
    threshold = noise * threshold_factor
    prominence = noise * prominence_factor

    # Convert min_distance_nm to pixels if provided
    distance = None
    if min_distance_nm is not None and len(wavelength) >= 2:
        spacing = float(np.median(np.diff(wavelength)))
        if spacing > 0:
            distance = max(1, int(min_distance_nm / spacing))

    kwargs = {"height": threshold, "prominence": prominence}
    if distance is not None:
        kwargs["distance"] = distance

    peak_indices, _ = find_peaks(corrected, **kwargs)

    if len(peak_indices) == 0:
        return []

    # Cosmic-ray rejection: filter out peaks narrower than min_width_pts
    if min_width_pts >= 2 and len(peak_indices) > 0:
        widths, _, _, _ = peak_widths(corrected, peak_indices, rel_height=0.5)
        width_mask = widths >= min_width_pts
        peak_indices = peak_indices[width_mask]

    return [(int(idx), float(wavelength[idx])) for idx in peak_indices]


def detect_peaks_auto(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    threshold_factor: float = 4.0,
    prominence_factor: float = 1.5,
    baseline_window_nm: float = 10.0,
    resolving_power: Optional[float] = None,
    min_width_pts: int = 2,
) -> Tuple[List[Tuple[int, float]], np.ndarray, float]:
    """High-level peak detection with automatic baseline and noise estimation.

    Convenience wrapper that runs the full preprocessing pipeline:
    baseline estimation, noise estimation, and peak detection with
    optional resolution-aware minimum distance.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    threshold_factor : float
        Peak height threshold in noise units (default 4.0)
    prominence_factor : float
        Peak prominence threshold in noise units (default 1.5)
    baseline_window_nm : float
        Median filter window for baseline estimation (default 10.0)
    resolving_power : float, optional
        Instrument resolving power (lambda/delta_lambda). If provided,
        the minimum peak distance is set to the resolution element at
        the mean wavelength.
    min_width_pts : int
        Minimum peak width for cosmic-ray rejection (default 2)

    Returns
    -------
    peaks : List[Tuple[int, float]]
        List of (index, wavelength_nm) tuples for detected peaks
    baseline : np.ndarray
        Estimated baseline array
    noise : float
        Estimated noise level (sigma)
    """
    if wavelength.size < 2:
        return [], np.zeros_like(intensity), 0.0

    baseline = estimate_baseline(wavelength, intensity, window_nm=baseline_window_nm)
    noise = estimate_noise(intensity, baseline)

    min_distance_nm = None
    if resolving_power is not None and resolving_power > 0:
        mean_wl = float(np.mean(wavelength))
        min_distance_nm = mean_wl / resolving_power

    peaks = detect_peaks(
        wavelength,
        intensity,
        baseline,
        noise,
        threshold_factor=threshold_factor,
        prominence_factor=prominence_factor,
        min_distance_nm=min_distance_nm,
        min_width_pts=min_width_pts,
    )
    return peaks, baseline, noise


def robust_normalize(intensity: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Percentile-based normalization robust to cosmic ray artifacts.

    Uses the given percentile instead of max() to avoid sensitivity
    to single-pixel cosmic ray spikes.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity array
    percentile : float
        Normalization percentile (default 95.0)

    Returns
    -------
    np.ndarray
        Normalized intensity array
    """
    scale = np.percentile(intensity, percentile)
    if scale > 1e-10:
        return intensity / scale
    return intensity.copy()
