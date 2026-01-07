"""
Boltzmann plot generation and fitting for CF-LIBS.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.boltzmann")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = object  # Dummy for type hinting if missing


@dataclass
class LineObservation:
    """
    Represents a single spectral line observation.
    """

    wavelength_nm: float
    intensity: float  # measured intensity (integrated area)
    intensity_uncertainty: float
    element: str
    ionization_stage: int
    E_k_ev: float  # upper level energy in eV
    g_k: int  # statistical weight of upper level
    A_ki: float  # Einstein coefficient in s^-1

    @property
    def y_value(self) -> float:
        """Calculate y-axis value: ln(I * lambda / (g * A))."""
        # Note: Intensity units are arbitrary, but consistent relative to each other.
        # wavelength in nm is fine as long as consistent.
        # Term inside log must be dimensionless or units handled in intercept.
        # Standard usage: ln( I * lambda[nm] / (g * A[s^-1]) )
        if self.intensity <= 0:
            return -np.inf
        return np.log(self.intensity * self.wavelength_nm / (self.g_k * self.A_ki))

    @property
    def y_uncertainty(self) -> float:
        """
        Calculate uncertainty in y-axis value.
        dy = d(ln x) = dx / x
        Here x = I * ...
        dx/x = dI/I (assuming errors in lambda, g, A are negligible)
        """
        if self.intensity == 0:
            return 0.0
        return self.intensity_uncertainty / self.intensity


@dataclass
class BoltzmannFitResult:
    """
    Results of a Boltzmann plot fit.
    """

    temperature_K: float
    temperature_uncertainty_K: float
    intercept: float
    intercept_uncertainty: float
    r_squared: float
    n_points: int
    rejected_points: List[int]  # Indices of rejected points
    slope: float
    slope_uncertainty: float


class BoltzmannPlotFitter:
    """
    Fitter for Boltzmann plots to determine excitation temperature.
    """

    def __init__(self, outlier_sigma: float = 3.0, max_iterations: int = 5):
        """
        Initialize fitter.

        Parameters
        ----------
        outlier_sigma : float
            Sigma threshold for outlier rejection
        max_iterations : int
            Maximum iterations for sigma clipping
        """
        self.outlier_sigma = outlier_sigma
        self.max_iterations = max_iterations

    def fit(self, observations: List[LineObservation]) -> BoltzmannFitResult:
        """
        Perform weighted linear regression on Boltzmann plot data.

        Parameters
        ----------
        observations : List[LineObservation]
            List of line observations

        Returns
        -------
        BoltzmannFitResult
            Fit results
        """
        if len(observations) < 2:
            raise ValueError("Need at least 2 points for a fit")

        # Prepare arrays
        x_all = np.array([obs.E_k_ev for obs in observations])
        y_all = np.array([obs.y_value for obs in observations])
        y_err_all = np.array([obs.y_uncertainty for obs in observations])

        # Handle cases where y calculation failed (e.g. negative intensity)
        valid_mask = np.isfinite(y_all)
        if not np.all(valid_mask):
            logger.warning(f"Excluding {np.sum(~valid_mask)} points with invalid Y values")

        indices = np.arange(len(observations))
        mask = valid_mask.copy()

        slope = 0.0
        intercept = 0.0
        slope_err = 0.0
        intercept_err = 0.0
        r_squared = 0.0

        # Iterative fitting with sigma clipping
        for iteration in range(self.max_iterations):
            x = x_all[mask]
            y = y_all[mask]
            y_err = y_err_all[mask]

            if len(x) < 2:
                logger.warning("Too few points remaining after rejection")
                break

            # Weighted least squares
            # Weights w = 1 / sigma^2
            # If errors are zero/unknown, use uniform weights
            if np.all(y_err == 0):
                weights = np.ones_like(y)
            else:
                # Avoid division by zero
                safe_err = np.where(y_err > 0, y_err, np.inf)
                weights = 1.0 / safe_err**2

            # Weighted polyfit (degree 1)
            # numpy.polyfit cov=True returns covariance matrix
            # V = [[Var(slope), Cov], [Cov, Var(intercept)]]
            try:
                (m, c), cov = np.polyfit(x, y, 1, w=weights, cov=True)
            except np.linalg.LinAlgError:
                logger.error("Linear regression failed")
                return self._empty_result()

            slope = m
            intercept = c
            slope_err = np.sqrt(cov[0, 0])
            intercept_err = np.sqrt(cov[1, 1])

            # Calculate R^2
            y_pred = m * x + c
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Check for outliers
            residuals = y - y_pred
            # Standardized residuals (approximate)
            std_res = np.std(residuals)
            if std_res == 0:
                break

            # Identify points outside sigma
            # Note: Using simple residual std dev for clipping
            bad_indices = np.abs(residuals) > self.outlier_sigma * std_res

            # If no new outliers, stop
            if not np.any(bad_indices):
                break

            # Update mask (remove bad points from CURRENT set)
            # We need to map back to original indices.
            # The 'mask' is boolean array of length 'len(observations)'
            # 'bad_indices' corresponds to current 'x' array

            current_indices = indices[mask]
            outlier_global_indices = current_indices[bad_indices]

            mask[outlier_global_indices] = False

            logger.debug(f"Iteration {iteration}: Rejected {len(outlier_global_indices)} outliers")

        # Calculate Temperature
        # slope = -1 / (kB * T)
        # T = -1 / (kB * slope)
        # Check for non-physical slope (positive slope -> negative temp, or T=infinity)

        if slope >= 0:
            logger.warning(
                f"Positive or zero slope ({slope}) detected. Population inversion or error. T set to infinity."
            )
            temperature_K = float("inf")
            temp_err_K = float("inf")
        else:
            temperature_K = -1.0 / (KB_EV * slope)
            # Uncertainty: dT = |dT/dm| * dm = |1/(kB * m^2)| * dm = |T / m| * dm = T^2 * kB * dm
            temp_err_K = (temperature_K**2) * KB_EV * slope_err

        rejected_points = list(indices[~mask])

        return BoltzmannFitResult(
            temperature_K=temperature_K,
            temperature_uncertainty_K=temp_err_K,
            intercept=intercept,
            intercept_uncertainty=intercept_err,
            r_squared=r_squared,
            n_points=np.sum(mask),
            rejected_points=rejected_points,
            slope=slope,
            slope_uncertainty=slope_err,
        )

    def _empty_result(self) -> BoltzmannFitResult:
        return BoltzmannFitResult(
            temperature_K=0.0,
            temperature_uncertainty_K=0.0,
            intercept=0.0,
            intercept_uncertainty=0.0,
            r_squared=0.0,
            n_points=0,
            rejected_points=[],
            slope=0.0,
            slope_uncertainty=0.0,
        )

    def plot(
        self, observations: List[LineObservation], result: BoltzmannFitResult, ax=None
    ) -> Optional[object]:
        """
        Plot the Boltzmann plot.

        Parameters
        ----------
        observations : List[LineObservation]
            Data points
        result : BoltzmannFitResult
            Fit result
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        Figure or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not installed, cannot plot.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Extract data
        x_all = np.array([obs.E_k_ev for obs in observations])
        y_all = np.array([obs.y_value for obs in observations])
        y_err_all = np.array([obs.y_uncertainty for obs in observations])

        # Split into accepted and rejected
        rejected_set = set(result.rejected_points)
        accepted_mask = np.array([i not in rejected_set for i in range(len(observations))])

        # Plot accepted
        ax.errorbar(
            x_all[accepted_mask],
            y_all[accepted_mask],
            yerr=y_err_all[accepted_mask],
            fmt="o",
            color="blue",
            label="Accepted",
            alpha=0.7,
            capsize=3,
        )

        # Plot rejected
        if np.any(~accepted_mask):
            ax.scatter(
                x_all[~accepted_mask],
                y_all[~accepted_mask],
                marker="x",
                color="red",
                label="Rejected",
                zorder=5,
            )

        # Plot fit line
        # Use range of x
        x_min, x_max = np.min(x_all), np.max(x_all)
        x_range = np.linspace(x_min, x_max, 100)
        y_fit = result.slope * x_range + result.intercept

        label_fit = f"Fit: T = {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K\n$R^2$={result.r_squared:.3f}"
        ax.plot(x_range, y_fit, "k--", label=label_fit)

        ax.set_xlabel("Upper Level Energy $E_k$ (eV)")
        ax.set_ylabel("$\\ln(I \\lambda / g_k A_{ki})$")
        ax.set_title(
            f"Boltzmann Plot - {observations[0].element} {observations[0].ionization_stage}"
        )
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        return fig
