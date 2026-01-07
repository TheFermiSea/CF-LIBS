"""
Tests for Stark broadening calculations.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from cflibs.radiation.stark import (
    stark_hwhm,
    stark_width,
    stark_shift,
    estimate_stark_parameter,
    StarkBroadeningCalculator,
    stark_hwhm_jax,
)
from cflibs.core.constants import EV_TO_K


class TestStarkCalculation:
    def test_stark_scaling(self):
        """Test Stark width scaling with density and temperature."""
        w_ref = 0.1  # nm (HWHM)
        n_e = 2.0e16
        T = 40000.0

        # Expected: 0.1 * (2/1) * (40000/10000)^(-0.5)
        # = 0.2 * 4^(-0.5) = 0.2 * 0.5 = 0.1

        hwhm = stark_hwhm(n_e, T, w_ref, stark_alpha=0.5)
        assert np.isclose(hwhm, 0.1)

        # FWHM = 2 * HWHM
        fwhm = stark_width(n_e, T, w_ref, stark_alpha=0.5)
        assert np.isclose(fwhm, 0.2)

    def test_stark_shift(self):
        """Test Stark shift scaling."""
        d_ref = 0.05
        n_e = 0.5e16

        shift = stark_shift(n_e, d_ref)
        # 0.05 * 0.5 = 0.025
        assert np.isclose(shift, 0.025)

    def test_estimation(self):
        """Test fallback estimation."""
        # Visible line, neutral
        wl = 500.0
        E_up = 10.0
        IP = 13.6
        stage = 1

        w_est = estimate_stark_parameter(wl, E_up, IP, stage)
        assert 0.0001 <= w_est <= 0.5

        # Check scaling
        # Lower binding energy (closer to continuum) -> larger width
        w_est_high = estimate_stark_parameter(wl, 13.0, IP, stage)  # E_up=13
        w_est_low = estimate_stark_parameter(wl, 5.0, IP, stage)  # E_up=5
        assert w_est_high > w_est_low

    def test_calculator(self):
        """Test Calculator with mock DB."""
        db = MagicMock()
        # Mock get_stark_parameters returning (w, alpha, d)
        db.get_stark_parameters.return_value = (0.01, 0.5, 0.002)

        calc = StarkBroadeningCalculator(db)

        fwhm = calc.get_stark_width("Fe", 1, 500.0, n_e_cm3=1e16, T_e_eV=1.0)
        # T_K ~ 11604. REF_T = 10000.
        # w = 0.01 * 1 * (11604/10000)^-0.5 ~ 0.01 * 0.928
        # FWHM = 2 * w

        assert fwhm > 0.0
        db.get_stark_parameters.assert_called_once()

    def test_calculator_fallback(self):
        """Test Calculator fallback when DB returns None."""
        db = MagicMock()
        db.get_stark_parameters.return_value = (None, None, None)
        db.get_ionization_potential.return_value = 10.0

        calc = StarkBroadeningCalculator(db)

        fwhm = calc.get_stark_width("Fe", 1, 500.0, n_e_cm3=1e16, T_e_eV=1.0, upper_energy_ev=8.0)
        # Should fallback to estimate
        assert fwhm > 0.0

    def test_jax_imports(self):
        """Test JAX implementation matches standard."""
        pytest.importorskip("jax")

        w_ref = 0.1
        n_e = 2.0e16
        T_eV = 1.0
        alpha = 0.5

        hwhm_jax = stark_hwhm_jax(n_e, T_eV, w_ref, alpha)

        # Compare with standard
        T_K = T_eV * EV_TO_K
        hwhm_std = stark_hwhm(n_e, T_K, w_ref, alpha)

        # Note: JAX uses REF_T_EV = 0.86173 vs T_K=10000
        # 10000 * KB_EV = 0.86173...
        assert np.isclose(float(hwhm_jax), hwhm_std, rtol=1e-3)
