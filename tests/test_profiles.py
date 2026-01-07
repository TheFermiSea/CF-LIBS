"""
Tests for line profile functions.
"""

import numpy as np
import pytest
from cflibs.radiation.profiles import (
    gaussian_profile,
    lorentzian_profile,
    voigt_profile,
    voigt_fwhm,
    doppler_width,
    HAS_JAX,
)


def test_gaussian_integral():
    """Test Gaussian profile integrates to amplitude."""
    x = np.linspace(-10, 10, 1000)
    sigma = 1.0
    amp = 5.0
    y = gaussian_profile(x, 0.0, sigma, amp)
    integral = np.trapezoid(y, x)
    assert np.isclose(integral, amp, rtol=1e-3)


def test_lorentzian_integral():
    """Test Lorentzian profile integrates to amplitude."""
    # Lorentzian has heavy tails, need wide integration range
    x = np.linspace(-100, 100, 10000)
    gamma = 1.0
    amp = 5.0
    y = lorentzian_profile(x, 0.0, gamma, amp)

    integral = np.trapezoid(y, x)
    # 1e-2 tolerance due to finite integration range
    assert np.isclose(integral, amp, rtol=1e-2)


def test_voigt_limits():
    """Test Voigt profile reduces to Gaussian or Lorentzian in limits."""
    x = np.linspace(-5, 5, 100)
    sigma = 1.0
    gamma = 1e-9  # limit -> 0

    # Gaussian limit
    v = voigt_profile(x, 0.0, sigma, gamma, 1.0)
    g = gaussian_profile(x, 0.0, sigma, 1.0)
    assert np.allclose(v, g, atol=1e-4)

    # Lorentzian limit
    sigma = 1e-9
    gamma = 1.0
    v = voigt_profile(x, 0.0, sigma, gamma, 1.0)
    lorentzian = lorentzian_profile(x, 0.0, gamma, 1.0)
    assert np.allclose(v, lorentzian, atol=1e-4)


def test_voigt_fwhm():
    """Test Voigt FWHM approximation."""
    sigma = 1.0
    gamma = 1.0

    # Calculate expected FWHM
    # fG = 2.355 * 1.0 = 2.355
    # fL = 2.0 * 1.0 = 2.0
    # fV approx 0.5346*2 + sqrt(0.2166*4 + 2.355^2)
    # = 1.0692 + sqrt(0.8664 + 5.546)
    # = 1.0692 + sqrt(6.4124)
    # = 1.0692 + 2.532
    # = 3.601

    fwhm = voigt_fwhm(sigma, gamma)
    assert 3.5 < fwhm < 3.7


def test_doppler_width():
    """Test Doppler width calculation."""
    # T = 1 eV (~11600 K), Mass = 1 amu (H)
    # sigma = lambda * sqrt(2kT/mc^2)
    # v_th = sqrt(2kT/m) ~ sqrt(2 * 1.6e-19 / 1.67e-27) ~ sqrt(1.9e8) ~ 1.38e4 m/s
    # c = 3e8
    # sigma/lambda = v/c ~ 1.38e4 / 3e8 ~ 4.6e-5
    # lambda = 500 nm -> sigma ~ 0.023 nm
    # FWHM ~ 2.355 * 0.023 ~ 0.054 nm

    dw = doppler_width(500.0, 1.0, 1.0)
    assert 0.04 < dw < 0.07


def test_jax_import():
    """Test if JAX-compatible function is importable (even if JAX missing)."""
    from cflibs.radiation.profiles import voigt_profile_jax

    if not HAS_JAX:
        with pytest.raises(ImportError):
            voigt_profile_jax(0.0, 0.0, 1.0, 1.0)
