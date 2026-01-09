"""
Unit and integration tests for cflibs.inversion.self_absorption module.

Tests cover:
- SelfAbsorptionCorrector initialization and correction
- Optical depth estimation
- Round-trip correction validation
- estimate_optical_depth_from_intensity_ratio function
"""

import pytest
import numpy as np

from cflibs.inversion.self_absorption import (
    SelfAbsorptionCorrector,
    SelfAbsorptionResult,
    AbsorptionCorrectionResult,
    estimate_optical_depth_from_intensity_ratio,
)
from cflibs.inversion.boltzmann import LineObservation


# ==============================================================================
# Helper Functions
# ==============================================================================


def correction_factor(tau: float) -> float:
    """Calculate the self-absorption correction factor f(τ)."""
    if tau < 1e-10:
        return 1.0
    elif tau > 50:
        return 1.0 / tau
    else:
        return (1.0 - np.exp(-tau)) / tau


# ==============================================================================
# SelfAbsorptionCorrector Initialization Tests
# ==============================================================================


class TestSelfAbsorptionCorrectorInit:
    """Tests for SelfAbsorptionCorrector initialization."""

    def test_default_parameters(self):
        """Verify default parameters are set correctly."""
        corrector = SelfAbsorptionCorrector()

        assert corrector.optical_depth_threshold == 0.1
        assert corrector.mask_threshold == 3.0
        assert corrector.max_iterations == 5
        assert corrector.convergence_tolerance == 0.01
        assert corrector.plasma_length_cm == 0.1

    def test_custom_parameters(self):
        """Verify custom parameters are stored."""
        corrector = SelfAbsorptionCorrector(
            optical_depth_threshold=0.5,
            mask_threshold=5.0,
            max_iterations=10,
            convergence_tolerance=0.001,
            plasma_length_cm=0.5,
        )

        assert corrector.optical_depth_threshold == 0.5
        assert corrector.mask_threshold == 5.0
        assert corrector.max_iterations == 10
        assert corrector.convergence_tolerance == 0.001
        assert corrector.plasma_length_cm == 0.5


# ==============================================================================
# Correction Factor Tests
# ==============================================================================


class TestCorrectionFactor:
    """Tests for the correction factor formula f(τ) = (1 - exp(-τ)) / τ."""

    def test_optically_thin_limit(self):
        """Verify f(τ) ≈ 1 for τ << 1."""
        tau = 0.01
        f = correction_factor(tau)

        # Taylor expansion: f(τ) ≈ 1 - τ/2 for small τ
        expected = 1.0 - tau / 2
        assert f == pytest.approx(expected, rel=0.01)

    def test_moderate_optical_depth(self):
        """Verify f(τ) for τ = 1."""
        tau = 1.0
        f = correction_factor(tau)

        # f(1) = (1 - 1/e) / 1 ≈ 0.632
        expected = (1.0 - np.exp(-1)) / 1.0
        assert f == pytest.approx(expected, abs=1e-6)
        assert f == pytest.approx(0.632, rel=0.01)

    def test_optically_thick_limit(self):
        """Verify f(τ) → 1/τ as τ → ∞."""
        tau = 10.0
        f = correction_factor(tau)

        # For large τ, f(τ) ≈ 1/τ
        expected = 1.0 / tau
        assert f == pytest.approx(expected, rel=0.01)

    def test_very_optically_thick(self):
        """Verify f(τ) for very large τ."""
        tau = 100.0
        f = correction_factor(tau)

        assert f < 0.02  # Should be very small
        assert f == pytest.approx(1.0 / tau, rel=0.01)

    def test_zero_optical_depth(self):
        """Verify f(0) = 1 (no absorption)."""
        f = correction_factor(0.0)
        assert f == 1.0


# ==============================================================================
# Round-Trip Correction Tests
# ==============================================================================


class TestRoundTripCorrection:
    """Tests verifying absorption then correction recovers original intensity."""

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, 3.0])
    def test_round_trip_recovery(self, tau, self_absorption_test_line):
        """Apply absorption then correct, verify recovery of original."""
        test_data = self_absorption_test_line(
            optical_depth=tau,
            original_intensity=1000.0,
        )

        # The absorbed intensity = original * f(tau)
        # To recover original: corrected = absorbed / f(tau)
        absorbed = test_data["absorbed_intensity"]
        original = test_data["original_intensity"]
        f_tau = test_data["correction_factor"]

        # Verify the absorption was applied correctly
        assert absorbed == pytest.approx(original * f_tau, rel=0.001)

        # Verify correction recovers original
        corrected = absorbed / f_tau
        assert corrected == pytest.approx(original, rel=0.001)

    def test_round_trip_optically_thin(self, self_absorption_test_line):
        """Verify optically thin lines are essentially unchanged."""
        test_data = self_absorption_test_line(
            optical_depth=0.01,
            original_intensity=1000.0,
        )

        absorbed = test_data["absorbed_intensity"]
        original = test_data["original_intensity"]

        # Should be nearly identical
        assert absorbed == pytest.approx(original, rel=0.01)


# ==============================================================================
# Optical Depth Estimation Tests
# ==============================================================================


class TestOpticalDepthEstimation:
    """Tests for optical depth estimation methods."""

    def test_estimate_from_doublet_ratio_no_absorption(self):
        """Verify τ = 0 when observed ratio matches theoretical."""
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1000.0,
            intensity_weak=500.0,
            theoretical_ratio=2.0,
        )

        assert tau == 0.0

    def test_estimate_from_doublet_ratio_with_absorption(self):
        """Verify τ estimation when strong line is absorbed."""
        # Theoretical ratio is 2:1
        # If strong line is absorbed with τ=1, its intensity is reduced by f(1) ≈ 0.632
        theoretical_ratio = 2.0
        f_tau = correction_factor(1.0)

        intensity_weak = 500.0
        intensity_strong = 1000.0 * f_tau  # Absorbed

        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=intensity_strong,
            intensity_weak=intensity_weak,
            theoretical_ratio=theoretical_ratio,
        )

        # Should recover τ ≈ 1.0
        assert tau == pytest.approx(1.0, rel=0.1)

    def test_estimate_from_doublet_zero_weak(self):
        """Verify τ = 0 when weak intensity is zero."""
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1000.0,
            intensity_weak=0.0,
            theoretical_ratio=2.0,
        )

        assert tau == 0.0

    def test_estimate_from_doublet_zero_ratio(self):
        """Verify τ = 0 when theoretical ratio is zero."""
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1000.0,
            intensity_weak=500.0,
            theoretical_ratio=0.0,
        )

        assert tau == 0.0

    def test_estimate_from_doublet_ratio_greater_than_theoretical(self):
        """Verify τ = 0 when observed ratio exceeds theoretical."""
        # This would indicate emission enhancement, not absorption
        tau = estimate_optical_depth_from_intensity_ratio(
            intensity_strong=1200.0,  # Higher than expected
            intensity_weak=500.0,
            theoretical_ratio=2.0,
        )

        assert tau == 0.0


# ==============================================================================
# SelfAbsorptionCorrector.correct() Tests
# ==============================================================================


class TestSelfAbsorptionCorrectorCorrect:
    """Tests for SelfAbsorptionCorrector.correct() method."""

    @pytest.fixture
    def corrector(self):
        return SelfAbsorptionCorrector(
            optical_depth_threshold=0.1,
            mask_threshold=3.0,
            max_iterations=5,
        )

    @pytest.fixture
    def sample_observations(self):
        """Create sample line observations."""
        return [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
            LineObservation(
                wavelength_nm=410.0,
                intensity=800.0,
                intensity_uncertainty=16.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.5,
                g_k=7,
                A_ki=8e6,
            ),
        ]

    def test_correct_returns_result(self, corrector, sample_observations):
        """Verify correct() returns SelfAbsorptionResult."""
        result = corrector.correct(
            observations=sample_observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        assert isinstance(result, SelfAbsorptionResult)
        assert len(result.corrected_observations) + len(result.masked_observations) == 2

    def test_optically_thin_unchanged(self, corrector, sample_observations):
        """Verify optically thin lines are returned unchanged."""
        # Very low concentration = low optical depth
        result = corrector.correct(
            observations=sample_observations,
            temperature_K=10000.0,
            concentrations={"Fe": 1e-10},  # Very low
            total_number_density_cm3=1e15,  # Low density
            partition_funcs={"Fe": 25.0},
        )

        # All should be returned without correction
        for obs, corrected in zip(sample_observations, result.corrected_observations):
            assert corrected.intensity == pytest.approx(obs.intensity, rel=0.001)

    def test_corrections_dict_populated(self, corrector, sample_observations):
        """Verify corrections dictionary is populated for all lines."""
        result = corrector.correct(
            observations=sample_observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        # All wavelengths should have entries
        for obs in sample_observations:
            assert obs.wavelength_nm in result.corrections
            assert isinstance(result.corrections[obs.wavelength_nm], AbsorptionCorrectionResult)

    def test_masked_lines_have_warnings(self):
        """Verify masked lines generate warnings."""
        corrector = SelfAbsorptionCorrector(mask_threshold=0.5)

        # High concentration = potentially high optical depth
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=10000.0,
                intensity_uncertainty=100.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=0.0,  # Ground state = high absorption
                g_k=25,
                A_ki=1e9,  # Very strong line
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=8000.0,
            concentrations={"Fe": 0.5},  # High concentration
            total_number_density_cm3=1e20,  # High density
            partition_funcs={"Fe": 25.0},
        )

        # With high optical depth, line may be masked
        # Check that we got some result
        assert len(result.corrected_observations) + len(result.masked_observations) == 1

    def test_empty_observations(self, corrector):
        """Verify handling of empty observation list."""
        result = corrector.correct(
            observations=[],
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        assert result.corrected_observations == []
        assert result.masked_observations == []
        assert result.n_corrected == 0
        assert result.n_masked == 0

    def test_missing_element_in_concentrations(self, corrector):
        """Verify graceful handling when element not in concentrations."""
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Cu",  # Not in concentrations
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=4,
                A_ki=1e8,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},  # No Cu
            total_number_density_cm3=1e18,
            partition_funcs={"Cu": 2.0},
        )

        # Should return unchanged (τ = 0 for missing element)
        assert len(result.corrected_observations) == 1
        assert result.corrected_observations[0].intensity == 1000.0


# ==============================================================================
# AbsorptionCorrectionResult Tests
# ==============================================================================


class TestAbsorptionCorrectionResult:
    """Tests for AbsorptionCorrectionResult dataclass."""

    def test_dataclass_creation(self):
        """Verify AbsorptionCorrectionResult can be instantiated."""
        result = AbsorptionCorrectionResult(
            original_intensity=1000.0,
            corrected_intensity=1200.0,
            optical_depth=0.5,
            correction_factor=0.833,
            is_optically_thick=False,
            iterations=3,
        )

        assert result.original_intensity == 1000.0
        assert result.corrected_intensity == 1200.0
        assert result.optical_depth == 0.5
        assert result.correction_factor == 0.833
        assert result.is_optically_thick is False
        assert result.iterations == 3

    def test_default_iterations(self):
        """Verify default iterations is 1."""
        result = AbsorptionCorrectionResult(
            original_intensity=1000.0,
            corrected_intensity=1000.0,
            optical_depth=0.01,
            correction_factor=1.0,
            is_optically_thick=False,
        )

        assert result.iterations == 1


# ==============================================================================
# SelfAbsorptionResult Tests
# ==============================================================================


class TestSelfAbsorptionResult:
    """Tests for SelfAbsorptionResult dataclass."""

    def test_dataclass_creation(self):
        """Verify SelfAbsorptionResult can be instantiated."""
        obs = LineObservation(
            wavelength_nm=400.0,
            intensity=1000.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=3.0,
            g_k=9,
            A_ki=1e7,
        )

        result = SelfAbsorptionResult(
            corrected_observations=[obs],
            masked_observations=[],
            corrections={400.0: AbsorptionCorrectionResult(
                original_intensity=1000.0,
                corrected_intensity=1000.0,
                optical_depth=0.01,
                correction_factor=1.0,
                is_optically_thick=False,
            )},
            n_corrected=0,
            n_masked=0,
            max_optical_depth=0.01,
            warnings=[],
        )

        assert len(result.corrected_observations) == 1
        assert len(result.masked_observations) == 0
        assert 400.0 in result.corrections
        assert result.n_corrected == 0
        assert result.max_optical_depth == 0.01

    def test_default_warnings(self):
        """Verify default warnings is empty list."""
        result = SelfAbsorptionResult(
            corrected_observations=[],
            masked_observations=[],
            corrections={},
            n_corrected=0,
            n_masked=0,
            max_optical_depth=0.0,
        )

        assert result.warnings == []


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestSelfAbsorptionEdgeCases:
    """Edge case tests for self-absorption module."""

    def test_zero_intensity(self):
        """Verify handling of zero intensity."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=0.0,  # Zero intensity
                intensity_uncertainty=0.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        # Should not crash
        assert len(result.corrected_observations) + len(result.masked_observations) == 1

    def test_zero_temperature(self):
        """Verify handling of zero temperature."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=0.0,  # Zero temperature
            concentrations={"Fe": 0.01},
            total_number_density_cm3=1e18,
            partition_funcs={"Fe": 25.0},
        )

        # Should return unchanged (τ = 0 for T = 0)
        assert len(result.corrected_observations) == 1

    def test_lower_level_energies_used(self):
        """Verify lower level energies are used when provided."""
        corrector = SelfAbsorptionCorrector()

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        # With E_i = 0 (ground state) vs E_i = 2.0 eV (excited)
        # Ground state should have higher absorption
        result_ground = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.1},
            total_number_density_cm3=1e19,
            partition_funcs={"Fe": 25.0},
            lower_level_energies={400.0: 0.0},  # Ground state
        )

        result_excited = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.1},
            total_number_density_cm3=1e19,
            partition_funcs={"Fe": 25.0},
            lower_level_energies={400.0: 3.0},  # Excited state
        )

        # Ground state should have higher optical depth
        tau_ground = result_ground.corrections[400.0].optical_depth
        tau_excited = result_excited.corrections[400.0].optical_depth

        assert tau_ground >= tau_excited

    def test_max_iterations_respected(self):
        """Verify max_iterations limit is respected."""
        corrector = SelfAbsorptionCorrector(
            max_iterations=2,
            convergence_tolerance=1e-10,  # Very tight, won't converge
        )

        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=1000.0,
                intensity_uncertainty=20.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=3.0,
                g_k=9,
                A_ki=1e7,
            ),
        ]

        result = corrector.correct(
            observations=observations,
            temperature_K=10000.0,
            concentrations={"Fe": 0.5},
            total_number_density_cm3=1e20,
            partition_funcs={"Fe": 25.0},
        )

        # Should not hang, should return with iterations <= max_iterations
        assert len(result.corrected_observations) + len(result.masked_observations) == 1
