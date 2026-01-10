"""
Outlier detection methods for LIBS spectra.

This module provides methods for identifying and removing anomalous spectra
from replicate measurements, which is essential for reliable CF-LIBS analysis.

Methods implemented:
- **SAM (Spectral Angle Mapper)**: Angle-based similarity insensitive to intensity scaling
- **MAD (Median Absolute Deviation)**: Robust univariate outlier detection (planned)

SAM is particularly useful for LIBS because:
1. Shot-to-shot laser energy fluctuations cause intensity variations
2. SAM compares spectral *shape*, ignoring overall intensity scaling
3. Anomalous spectra (e.g., plasma didn't form, contamination) have different shapes

References:
    - Kruse et al. (1993), "The Spectral Image Processing System (SIPS)"
    - Zhang et al. (2017), "Spectral preprocessing for LIBS"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.outliers")


class OutlierMethod(Enum):
    """Method for outlier detection."""

    SAM = "sam"  # Spectral Angle Mapper
    # MAD = "mad"  # Median Absolute Deviation (future)


@dataclass
class SAMResult:
    """
    Results of SAM-based outlier detection.

    Attributes
    ----------
    angles : np.ndarray
        SAM angle (radians) from each spectrum to the reference.
        Range: [0, pi/2]. Lower = more similar.
    outlier_mask : np.ndarray
        Boolean mask where True indicates an outlier spectrum.
    threshold : float
        Threshold angle (radians) used for outlier detection.
    reference_spectrum : np.ndarray
        The reference spectrum used for comparison.
    n_outliers : int
        Number of outliers detected.
    inlier_indices : np.ndarray
        Indices of inlier (non-outlier) spectra.
    outlier_indices : np.ndarray
        Indices of outlier spectra.
    method : str
        Thresholding method used ('mad', 'percentile', 'fixed').
    """

    angles: np.ndarray
    outlier_mask: np.ndarray
    threshold: float
    reference_spectrum: np.ndarray
    n_outliers: int
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray
    method: str = "mad"

    @property
    def n_inliers(self) -> int:
        """Number of inlier spectra."""
        return len(self.inlier_indices)

    @property
    def outlier_fraction(self) -> float:
        """Fraction of spectra flagged as outliers."""
        total = len(self.angles)
        return self.n_outliers / total if total > 0 else 0.0

    def angles_degrees(self) -> np.ndarray:
        """SAM angles in degrees for easier interpretation."""
        return np.degrees(self.angles)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"SAM Outlier Detection Results:",
            f"  Total spectra: {len(self.angles)}",
            f"  Outliers: {self.n_outliers} ({self.outlier_fraction*100:.1f}%)",
            f"  Threshold: {np.degrees(self.threshold):.2f} degrees ({self.method})",
            f"  Angle range: {np.degrees(self.angles.min()):.2f} - {np.degrees(self.angles.max()):.2f} degrees",
        ]
        if self.n_outliers > 0:
            lines.append(f"  Outlier indices: {list(self.outlier_indices)}")
        return "\n".join(lines)


class SpectralAngleMapper:
    """
    Spectral Angle Mapper (SAM) for outlier detection in LIBS spectra.

    SAM calculates the angle between two spectra treated as vectors in
    n-dimensional space (n = number of wavelength channels):

        SAM(s1, s2) = arccos(s1 · s2 / (||s1|| × ||s2||))

    Key properties:
    - Range: [0, π/2] radians (0° to 90°)
    - Intensity-invariant: SAM(s, k*s) = 0 for any scalar k > 0
    - Small angles = similar spectral shapes
    - Large angles = different spectral shapes

    Parameters
    ----------
    threshold_method : str
        Method for automatic threshold selection:
        - 'mad': Median + k × MAD (default, robust)
        - 'percentile': Use percentile of angle distribution
        - 'fixed': Use fixed threshold value
    threshold_sigma : float
        For 'mad' method: number of MAD units above median (default: 3.0)
    threshold_percentile : float
        For 'percentile' method: percentile value (default: 95.0)
    threshold_fixed : float
        For 'fixed' method: angle threshold in radians (default: 0.1 ~ 5.7°)
    reference_method : str
        Method for computing reference spectrum:
        - 'mean': Use mean spectrum (default)
        - 'median': Use element-wise median (more robust)

    Examples
    --------
    >>> sam = SpectralAngleMapper(threshold_sigma=3.0)
    >>> result = sam.detect_outliers(spectra)
    >>> clean_spectra = spectra[~result.outlier_mask]
    >>> print(result.summary())
    """

    def __init__(
        self,
        threshold_method: str = "mad",
        threshold_sigma: float = 3.0,
        threshold_percentile: float = 95.0,
        threshold_fixed: float = 0.1,
        reference_method: str = "mean",
    ):
        if threshold_method not in ("mad", "percentile", "fixed"):
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
        if reference_method not in ("mean", "median"):
            raise ValueError(f"Unknown reference_method: {reference_method}")

        self.threshold_method = threshold_method
        self.threshold_sigma = threshold_sigma
        self.threshold_percentile = threshold_percentile
        self.threshold_fixed = threshold_fixed
        self.reference_method = reference_method

    def spectral_angle(
        self,
        spectrum1: np.ndarray,
        spectrum2: np.ndarray,
    ) -> float:
        """
        Calculate SAM angle between two spectra.

        Parameters
        ----------
        spectrum1 : np.ndarray
            First spectrum (1D array of intensities)
        spectrum2 : np.ndarray
            Second spectrum (1D array, same length as spectrum1)

        Returns
        -------
        float
            Angle in radians [0, π/2]

        Raises
        ------
        ValueError
            If spectra have different lengths or are zero vectors
        """
        s1 = np.asarray(spectrum1, dtype=np.float64)
        s2 = np.asarray(spectrum2, dtype=np.float64)

        if s1.shape != s2.shape:
            raise ValueError(f"Spectrum shape mismatch: {s1.shape} vs {s2.shape}")

        # Compute norms
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            raise ValueError("Cannot compute SAM for zero or near-zero spectrum")

        # Compute cosine of angle
        cos_angle = np.dot(s1, s2) / (norm1 * norm2)

        # Clamp to [-1, 1] for numerical stability
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.arccos(cos_angle))

    def pairwise_angles(
        self,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise SAM angles between all spectra.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)

        Returns
        -------
        np.ndarray
            Symmetric matrix of shape (n_spectra, n_spectra) with SAM angles.
            Diagonal elements are 0.
        """
        spectra = np.asarray(spectra, dtype=np.float64)
        # Normalize all spectra
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        normalized = spectra / norms

        # Compute cosine similarity matrix
        cos_matrix = normalized @ normalized.T

        # Clamp for numerical stability
        cos_matrix = np.clip(cos_matrix, -1.0, 1.0)

        # Convert to angles
        angles = np.arccos(cos_matrix)

        return angles

    def angles_from_reference(
        self,
        spectra: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SAM angles from each spectrum to a reference.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        reference : np.ndarray, optional
            Reference spectrum. If None, computed from spectra using
            reference_method.

        Returns
        -------
        angles : np.ndarray
            SAM angle from each spectrum to reference
        reference : np.ndarray
            The reference spectrum used
        """
        spectra = np.asarray(spectra, dtype=np.float64)

        if reference is None:
            reference = self._compute_reference(spectra)
        else:
            reference = np.asarray(reference, dtype=np.float64)

        # Normalize reference
        ref_norm = np.linalg.norm(reference)
        if ref_norm < 1e-10:
            raise ValueError("Reference spectrum is zero or near-zero")
        ref_normalized = reference / ref_norm

        # Normalize all spectra
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = spectra / norms

        # Compute cosine similarities
        cos_angles = normalized @ ref_normalized

        # Clamp and convert to angles
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)

        return angles, reference

    def detect_outliers(
        self,
        spectra: np.ndarray,
        reference: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> SAMResult:
        """
        Detect outlier spectra using SAM distance from reference.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        reference : np.ndarray, optional
            Reference spectrum. If None, computed from spectra.
        threshold : float, optional
            Override threshold (in radians). If None, computed automatically.

        Returns
        -------
        SAMResult
            Detection results including outlier mask and statistics
        """
        spectra = np.asarray(spectra, dtype=np.float64)

        if spectra.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {spectra.shape}")

        n_spectra = spectra.shape[0]

        if n_spectra < 2:
            logger.warning("Need at least 2 spectra for outlier detection")
            return SAMResult(
                angles=np.array([0.0]),
                outlier_mask=np.array([False]),
                threshold=0.0,
                reference_spectrum=spectra[0] if n_spectra > 0 else np.array([]),
                n_outliers=0,
                inlier_indices=np.array([0]) if n_spectra > 0 else np.array([]),
                outlier_indices=np.array([], dtype=int),
                method=self.threshold_method,
            )

        # Compute angles from reference
        angles, ref_spectrum = self.angles_from_reference(spectra, reference)

        # Determine threshold
        if threshold is not None:
            thresh = threshold
            method_used = "fixed"
        else:
            thresh = self._compute_threshold(angles)
            method_used = self.threshold_method

        # Identify outliers
        outlier_mask = angles > thresh
        outlier_indices = np.where(outlier_mask)[0]
        inlier_indices = np.where(~outlier_mask)[0]

        n_outliers = int(np.sum(outlier_mask))

        if n_outliers > 0:
            logger.info(
                f"SAM detected {n_outliers}/{n_spectra} outliers "
                f"(threshold: {np.degrees(thresh):.2f}°)"
            )

        return SAMResult(
            angles=angles,
            outlier_mask=outlier_mask,
            threshold=thresh,
            reference_spectrum=ref_spectrum,
            n_outliers=n_outliers,
            inlier_indices=inlier_indices,
            outlier_indices=outlier_indices,
            method=method_used,
        )

    def filter_spectra(
        self,
        spectra: np.ndarray,
        reference: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, SAMResult]:
        """
        Filter out outlier spectra, returning only inliers.

        Convenience method that combines detection and filtering.

        Parameters
        ----------
        spectra : np.ndarray
            2D array of shape (n_spectra, n_wavelengths)
        reference : np.ndarray, optional
            Reference spectrum
        threshold : float, optional
            Override threshold (radians)

        Returns
        -------
        filtered_spectra : np.ndarray
            Spectra with outliers removed
        result : SAMResult
            Detection results for diagnostics
        """
        result = self.detect_outliers(spectra, reference, threshold)
        filtered = spectra[~result.outlier_mask]
        return filtered, result

    def _compute_reference(self, spectra: np.ndarray) -> np.ndarray:
        """Compute reference spectrum from collection."""
        if self.reference_method == "median":
            return np.median(spectra, axis=0)
        else:  # mean
            return np.mean(spectra, axis=0)

    def _compute_threshold(self, angles: np.ndarray) -> float:
        """Compute threshold based on angle distribution."""
        if self.threshold_method == "fixed":
            return self.threshold_fixed

        elif self.threshold_method == "percentile":
            return float(np.percentile(angles, self.threshold_percentile))

        else:  # mad
            median_angle = float(np.median(angles))
            # MAD = median(|x - median(x)|)
            mad = float(np.median(np.abs(angles - median_angle)))

            # Scale MAD to approximate standard deviation for normal distribution
            # For normal distribution: MAD ≈ 0.6745 × σ
            mad_scaled = mad * 1.4826

            threshold = median_angle + self.threshold_sigma * mad_scaled

            # Ensure threshold is at least median + small margin
            threshold = max(threshold, median_angle * 1.1)

            return threshold


def sam_distance(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
) -> float:
    """
    Convenience function to compute SAM angle between two spectra.

    Parameters
    ----------
    spectrum1 : np.ndarray
        First spectrum
    spectrum2 : np.ndarray
        Second spectrum

    Returns
    -------
    float
        SAM angle in radians
    """
    mapper = SpectralAngleMapper()
    return float(mapper.spectral_angle(spectrum1, spectrum2))


def detect_outlier_spectra(
    spectra: np.ndarray,
    threshold_sigma: float = 3.0,
    reference_method: str = "mean",
) -> SAMResult:
    """
    Convenience function for SAM-based outlier detection.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of shape (n_spectra, n_wavelengths)
    threshold_sigma : float
        Number of MAD units for threshold (default: 3.0)
    reference_method : str
        'mean' or 'median' for computing reference

    Returns
    -------
    SAMResult
        Detection results
    """
    mapper = SpectralAngleMapper(
        threshold_method="mad",
        threshold_sigma=threshold_sigma,
        reference_method=reference_method,
    )
    return mapper.detect_outliers(spectra)
