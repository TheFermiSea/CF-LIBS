"""
Tests for Boltzmann plot generation and fitting.
"""

import numpy as np
import pytest
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter
from cflibs.core.constants import KB_EV


def create_synthetic_lines(
    T_K: float, n_points: int = 10, noise_level: float = 0.05
) -> list[LineObservation]:
    """Generate synthetic spectral lines following Boltzmann distribution."""
    T_eV = T_K * KB_EV

    # Random upper energies between 2 and 6 eV
    energies = np.linspace(2.0, 6.0, n_points)

    # Constants (F*C/U) - arbitrary
    intercept_const = 10.0

    obs = []
    for Ek in energies:
        # ln(I*lam/gA) = ln(const) - Ek/kT
        # y = ln_const - Ek/T_eV
        expected_y = np.log(intercept_const) - Ek / T_eV

        # Add noise
        y_noisy = expected_y + np.random.normal(0, noise_level)

        # Back-calculate Intensity (assuming lam=1, g=1, A=1 for simplicity)
        # y = ln(I) -> I = exp(y)
        intensity = np.exp(y_noisy)

        # Estimate uncertainty (say 5%)
        I_err = intensity * 0.05

        obs.append(
            LineObservation(
                wavelength_nm=1.0,
                intensity=intensity,
                intensity_uncertainty=I_err,
                element="Fe",
                ionization_stage=1,
                E_k_ev=Ek,
                g_k=1,
                A_ki=1.0,
            )
        )

    return obs


def test_boltzmann_fit_perfect():
    """Test fitting on perfect data."""
    T_target = 10000.0
    # Very low noise
    lines = create_synthetic_lines(T_target, n_points=20, noise_level=0.0001)

    fitter = BoltzmannPlotFitter()
    result = fitter.fit(lines)

    assert result.n_points == 20
    # Tolerance 1%
    assert abs(result.temperature_K - T_target) < 100.0
    assert result.r_squared > 0.99


def test_outlier_rejection():
    """Test that outliers are correctly rejected."""
    T_target = 8000.0
    lines = create_synthetic_lines(T_target, n_points=10, noise_level=0.01)

    # Add an outlier (e.g. self-absorbed line, intensity lower than expected)
    # y = ln(I...) - Ek/kT. Lower I -> lower y.
    # Let's manually modify the last point
    outlier = lines[-1]
    # Reduce intensity significantly (e.g. factor of 10)
    outlier.intensity /= 10.0

    fitter = BoltzmannPlotFitter(outlier_sigma=2.0)
    result = fitter.fit(lines)

    assert len(result.rejected_points) >= 1
    assert 9 in result.rejected_points  # Last index

    # Temperature should still be reasonably close
    assert abs(result.temperature_K - T_target) < 500.0


def test_insufficient_points():
    """Test error handling for too few points."""
    lines = create_synthetic_lines(5000, n_points=1)
    fitter = BoltzmannPlotFitter()
    with pytest.raises(ValueError):
        fitter.fit(lines)


def test_y_value_calculation():
    """Test the y-axis calculation logic."""
    obs = LineObservation(
        wavelength_nm=500.0,
        intensity=100.0,
        intensity_uncertainty=10.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=3.0,
        g_k=2,
        A_ki=1.0e6,
    )
    # y = ln(I * lam / (g * A))
    # y = ln(100 * 500 / (2 * 1e6)) = ln(50000 / 2e6) = ln(0.025) ≈ -3.688
    expected = np.log(100.0 * 500.0 / (2.0 * 1.0e6))
    assert abs(obs.y_value - expected) < 1e-6

    # Uncertainty: dy = dI/I = 10/100 = 0.1
    assert abs(obs.y_uncertainty - 0.1) < 1e-6
