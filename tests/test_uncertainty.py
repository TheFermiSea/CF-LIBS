"""
Tests for uncertainty propagation utilities.

Tests the `cflibs.inversion.uncertainty` module which provides automatic
correlation-aware uncertainty propagation using the `uncertainties` package.
"""

import pytest
import numpy as np


# Skip all tests if uncertainties not available
pytest.importorskip("uncertainties")


class TestCreateBoltzmannUncertainties:
    """Tests for create_boltzmann_uncertainties function."""

    def test_with_covariance_matrix(self):
        """Test creating correlated uncertainties from covariance matrix."""
        from cflibs.inversion.uncertainty import create_boltzmann_uncertainties

        slope = -1.0
        intercept = 5.0
        # Covariance matrix with correlation
        cov = np.array([[0.01, 0.005], [0.005, 0.04]])

        slope_u, intercept_u = create_boltzmann_uncertainties(slope, intercept, cov)

        assert slope_u.nominal_value == pytest.approx(slope)
        assert intercept_u.nominal_value == pytest.approx(intercept)
        assert slope_u.std_dev == pytest.approx(np.sqrt(0.01))
        assert intercept_u.std_dev == pytest.approx(np.sqrt(0.04))

    def test_without_covariance_matrix(self):
        """Test fallback to independent uncertainties."""
        from cflibs.inversion.uncertainty import create_boltzmann_uncertainties

        slope = -1.0
        intercept = 5.0
        slope_err = 0.1
        intercept_err = 0.2

        slope_u, intercept_u = create_boltzmann_uncertainties(
            slope, intercept, None, slope_err, intercept_err
        )

        assert slope_u.nominal_value == pytest.approx(slope)
        assert intercept_u.nominal_value == pytest.approx(intercept)
        assert slope_u.std_dev == pytest.approx(slope_err)
        assert intercept_u.std_dev == pytest.approx(intercept_err)

    def test_correlation_preserved(self):
        """Test that correlations are preserved for correlated operations."""
        from cflibs.inversion.uncertainty import create_boltzmann_uncertainties

        slope = -1.0
        intercept = 5.0
        # Perfectly correlated
        cov = np.array([[0.01, 0.01], [0.01, 0.01]])

        slope_u, intercept_u = create_boltzmann_uncertainties(slope, intercept, cov)

        # For perfectly correlated variables, slope - slope = 0 exactly
        diff = slope_u - slope_u
        assert diff.nominal_value == pytest.approx(0.0)
        assert diff.std_dev == pytest.approx(0.0)


class TestTemperatureFromSlope:
    """Tests for temperature_from_slope function."""

    def test_basic_conversion(self):
        """Test temperature calculation from slope."""
        from cflibs.inversion.uncertainty import temperature_from_slope
        from uncertainties import ufloat
        from cflibs.core.constants import KB_EV, EV_TO_K

        # slope = -1/(kB * T_eV)
        # For T = 10000 K = 0.862 eV, slope = -1.16
        T_expected_K = 10000.0
        T_expected_eV = T_expected_K / EV_TO_K
        slope = -1.0 / (KB_EV * T_expected_eV)

        slope_u = ufloat(slope, 0.01)
        T_K_u = temperature_from_slope(slope_u)

        assert T_K_u.nominal_value == pytest.approx(T_expected_K, rel=0.01)
        # Uncertainty should be non-zero
        assert T_K_u.std_dev > 0


class TestClosurePropagation:
    """Tests for closure equation uncertainty propagation."""

    def test_standard_closure_sum_to_one(self):
        """Test that concentrations sum to 1 regardless of uncertainties."""
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.3),
            "Al": ufloat(7.0, 0.4),
        }
        partition_funcs = {"Fe": 25.0, "Si": 15.0, "Al": 10.0}

        concentrations_u = propagate_through_closure_standard(intercepts_u, partition_funcs)

        # Sum of concentrations should be exactly 1
        total = sum(c.nominal_value for c in concentrations_u.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_standard_closure_propagates_uncertainty(self):
        """Test that uncertainties are propagated through closure."""
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.3),
        }
        partition_funcs = {"Fe": 25.0, "Si": 15.0}

        concentrations_u = propagate_through_closure_standard(intercepts_u, partition_funcs)

        # All uncertainties should be positive (propagated)
        for conc_u in concentrations_u.values():
            assert conc_u.std_dev > 0

    def test_matrix_closure_fixed_element(self):
        """Test matrix mode closure with fixed element concentration."""
        from cflibs.inversion.uncertainty import propagate_through_closure_matrix
        from uncertainties import ufloat

        intercepts_u = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.3),
        }
        partition_funcs = {"Fe": 25.0, "Si": 15.0}

        concentrations_u = propagate_through_closure_matrix(
            intercepts_u, partition_funcs, matrix_element="Fe", matrix_fraction=0.9
        )

        # Matrix element should have concentration = matrix_fraction
        assert concentrations_u["Fe"].nominal_value == pytest.approx(0.9)

    def test_correlation_in_closure(self):
        """Test that correlations affect closure uncertainty properly.

        When intercepts are correlated (from same Boltzmann fit), the
        relative uncertainties in concentrations should be smaller than
        with independent errors.
        """
        from cflibs.inversion.uncertainty import propagate_through_closure_standard
        from uncertainties import ufloat, correlated_values

        # Case 1: Independent uncertainties
        intercepts_indep = {
            "Fe": ufloat(10.0, 0.5),
            "Si": ufloat(8.0, 0.5),
        }
        partition_funcs = {"Fe": 25.0, "Si": 25.0}

        conc_indep = propagate_through_closure_standard(intercepts_indep, partition_funcs)
        fe_uncert_indep = conc_indep["Fe"].std_dev

        # Case 2: Perfectly correlated (both from same measurement)
        # When q_Fe and q_Si move together, ratio stays more constant
        cov = np.array([[0.25, 0.25], [0.25, 0.25]])  # Perfect correlation
        q_fe, q_si = correlated_values([10.0, 8.0], cov)
        intercepts_corr = {"Fe": q_fe, "Si": q_si}

        conc_corr = propagate_through_closure_standard(intercepts_corr, partition_funcs)
        fe_uncert_corr = conc_corr["Fe"].std_dev

        # Correlated case should have smaller relative uncertainty
        # because common-mode errors cancel in the ratio
        assert fe_uncert_corr < fe_uncert_indep


class TestExtractValuesAndUncertainties:
    """Tests for extract_values_and_uncertainties function."""

    def test_extraction(self):
        """Test extracting nominal values and uncertainties from dict."""
        from cflibs.inversion.uncertainty import extract_values_and_uncertainties
        from uncertainties import ufloat

        data = {
            "Fe": ufloat(0.6, 0.05),
            "Si": ufloat(0.3, 0.03),
            "Al": ufloat(0.1, 0.02),
        }

        nominal, uncert = extract_values_and_uncertainties(data)

        assert nominal["Fe"] == pytest.approx(0.6)
        assert nominal["Si"] == pytest.approx(0.3)
        assert nominal["Al"] == pytest.approx(0.1)
        assert uncert["Fe"] == pytest.approx(0.05)
        assert uncert["Si"] == pytest.approx(0.03)
        assert uncert["Al"] == pytest.approx(0.02)


class TestBoltzmannFitResultCovariance:
    """Tests for covariance_matrix attribute in BoltzmannFitResult."""

    def test_sigma_clip_produces_covariance(self):
        """Test that sigma_clip fitting produces covariance matrix."""
        from cflibs.inversion.boltzmann import (
            BoltzmannPlotFitter,
            LineObservation,
            FitMethod,
        )

        # Create synthetic data for Fe I with known slope
        observations = [
            LineObservation(
                wavelength_nm=300.0 + i * 10,
                intensity=1000.0 * np.exp(-E / 0.8),  # T ~ 9300 K
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E,
                g_k=5,
                A_ki=1e8,
            )
            for i, E in enumerate([2.0, 3.0, 4.0, 5.0, 6.0])  # Need >2 points
        ]

        fitter = BoltzmannPlotFitter(method=FitMethod.SIGMA_CLIP)
        result = fitter.fit(observations)

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)
        # Diagonal elements should be positive
        assert result.covariance_matrix[0, 0] > 0
        assert result.covariance_matrix[1, 1] > 0

    def test_ransac_produces_covariance(self):
        """Test that RANSAC fitting produces covariance matrix."""
        from cflibs.inversion.boltzmann import (
            BoltzmannPlotFitter,
            LineObservation,
            FitMethod,
        )

        observations = [
            LineObservation(
                wavelength_nm=300.0 + i * 10,
                intensity=1000.0 * np.exp(-E / 0.8),
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E,
                g_k=5,
                A_ki=1e8,
            )
            for i, E in enumerate([2.0, 3.0, 4.0, 5.0, 6.0])
        ]

        fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
        result = fitter.fit(observations)

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)

    def test_huber_produces_covariance(self):
        """Test that Huber fitting produces covariance matrix."""
        from cflibs.inversion.boltzmann import (
            BoltzmannPlotFitter,
            LineObservation,
            FitMethod,
        )

        observations = [
            LineObservation(
                wavelength_nm=300.0 + i * 10,
                intensity=1000.0 * np.exp(-E / 0.8),
                intensity_uncertainty=50.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=E,
                g_k=5,
                A_ki=1e8,
            )
            for i, E in enumerate([2.0, 3.0, 4.0, 5.0, 6.0])
        ]

        fitter = BoltzmannPlotFitter(method=FitMethod.HUBER)
        result = fitter.fit(observations)

        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)


class TestCFLIBSResultFields:
    """Tests for new CFLIBSResult fields."""

    def test_result_has_new_fields(self):
        """Test that CFLIBSResult has the new uncertainty fields."""
        from cflibs.inversion.solver import CFLIBSResult

        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.9, "Si": 0.1},
            concentration_uncertainties={"Fe": 0.05, "Si": 0.02},
            iterations=5,
            converged=True,
            quality_metrics={"r_squared_last": 0.98},
            electron_density_uncertainty_cm3=1e15,
            boltzmann_covariance=np.array([[0.01, 0.005], [0.005, 0.04]]),
        )

        assert result.electron_density_uncertainty_cm3 == 1e15
        assert result.boltzmann_covariance is not None
        assert result.boltzmann_covariance.shape == (2, 2)

    def test_result_default_values(self):
        """Test that new fields have sensible defaults."""
        from cflibs.inversion.solver import CFLIBSResult

        result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=0.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 1.0},
            concentration_uncertainties={},
            iterations=1,
            converged=True,
        )

        assert result.electron_density_uncertainty_cm3 == 0.0
        assert result.boltzmann_covariance is None
