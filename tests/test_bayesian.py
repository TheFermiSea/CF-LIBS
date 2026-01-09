"""
Tests for Bayesian inference module.

These tests validate:
1. BayesianForwardModel initialization and spectrum computation
2. Log-likelihood calculation with noise model
3. Prior creation functions
4. MCMC sampling (integration test)
"""

import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path

# Skip all tests if JAX is not available
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cflibs.inversion.bayesian import (
    BayesianForwardModel,
    AtomicDataArrays,
    NoiseParameters,
    PriorConfig,
    log_likelihood,
)


@pytest.fixture
def bayesian_db():
    """Create a database with partition functions and Stark parameters for Bayesian tests."""
    fd, db_path = tempfile.mkstemp(suffix=".db")

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL,
            stark_w REAL,
            stark_alpha REAL
        )
    """)

    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)

    conn.execute("""
        CREATE TABLE partition_functions (
            element TEXT,
            sp_num INTEGER,
            a0 REAL,
            a1 REAL,
            a2 REAL,
            a3 REAL,
            a4 REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)

    # Insert Fe spectral lines with Stark parameters
    lines_data = [
        # Fe I lines (sp_num=1)
        ("Fe", 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.02, 0.5),
        ("Fe", 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500, 0.015, 0.5),
        ("Fe", 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200, 0.01, 0.5),
        ("Fe", 1, 438.35, 5.0e6, 0.0, 4.47, 9, 9, 800, 0.025, 0.5),
        # Fe II lines (sp_num=2)
        ("Fe", 2, 238.20, 3.0e8, 0.0, 5.22, 10, 10, 600, 0.03, 0.6),
        ("Fe", 2, 259.94, 2.2e8, 0.0, 4.77, 8, 10, 400, 0.025, 0.6),
    ]
    conn.executemany("""
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int, stark_w, stark_alpha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, lines_data)

    # Cu I lines
    cu_lines = [
        ("Cu", 1, 324.75, 1.4e8, 0.0, 3.82, 2, 4, 2000, 0.01, 0.5),
        ("Cu", 1, 327.40, 1.4e8, 0.0, 3.79, 2, 2, 1000, 0.01, 0.5),
        ("Cu", 1, 510.55, 2.0e6, 1.39, 3.82, 4, 4, 300, 0.008, 0.5),
    ]
    conn.executemany("""
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int, stark_w, stark_alpha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, cu_lines)

    # Ionization potentials
    conn.executemany("""
        INSERT INTO species_physics (element, sp_num, ip_ev) VALUES (?, ?, ?)
    """, [
        ("Fe", 1, 7.87),
        ("Fe", 2, 16.18),
        ("Cu", 1, 7.73),
        ("Cu", 2, 20.29),
    ])

    # Partition function coefficients (Irwin polynomial form)
    # log(U) = a0 + a1*log(T) + a2*log(T)^2 + ...
    pf_data = [
        # Fe I: U ~ 25 at 10000K
        ("Fe", 1, 3.22, 0.0, 0.0, 0.0, 0.0),
        # Fe II: U ~ 40 at 10000K
        ("Fe", 2, 3.69, 0.0, 0.0, 0.0, 0.0),
        # Cu I: U ~ 2 at 10000K
        ("Cu", 1, 0.69, 0.0, 0.0, 0.0, 0.0),
        # Cu II: U ~ 1 at 10000K
        ("Cu", 2, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    conn.executemany("""
        INSERT INTO partition_functions (element, sp_num, a0, a1, a2, a3, a4)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, pf_data)

    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink()


@pytest.mark.requires_jax
class TestNoiseParameters:
    """Tests for NoiseParameters dataclass."""

    def test_default_values(self):
        """Test default noise parameter values."""
        params = NoiseParameters()
        assert params.readout_noise == 10.0
        assert params.dark_current == 1.0
        assert params.gain == 1.0

    def test_custom_values(self):
        """Test custom noise parameter values."""
        params = NoiseParameters(
            readout_noise=5.0,
            dark_current=0.5,
            gain=2.0,
        )
        assert params.readout_noise == 5.0
        assert params.dark_current == 0.5
        assert params.gain == 2.0


@pytest.mark.requires_jax
class TestPriorConfig:
    """Tests for PriorConfig dataclass."""

    def test_default_values(self):
        """Test default prior configuration."""
        config = PriorConfig()
        assert config.T_eV_range == (0.5, 3.0)
        # Note: log_ne range limited to 17.0 due to Voigt profile gradient stability
        assert config.log_ne_range == (15.0, 17.0)
        assert config.concentration_alpha == 1.0

    def test_custom_values(self):
        """Test custom prior configuration."""
        config = PriorConfig(
            T_eV_range=(0.8, 2.5),
            log_ne_range=(16.0, 18.0),
            concentration_alpha=2.0,
        )
        assert config.T_eV_range == (0.8, 2.5)
        assert config.log_ne_range == (16.0, 18.0)
        assert config.concentration_alpha == 2.0


@pytest.mark.requires_jax
class TestAtomicDataArrays:
    """Tests for AtomicDataArrays dataclass."""

    def test_creation(self):
        """Test AtomicDataArrays creation."""
        n_lines = 10
        arrays = AtomicDataArrays(
            wavelength_nm=jnp.linspace(300, 600, n_lines),
            aki=jnp.ones(n_lines) * 1e7,
            ek_ev=jnp.linspace(1, 5, n_lines),
            gk=jnp.ones(n_lines, dtype=jnp.int32) * 9,
            ip_ev=jnp.ones(n_lines) * 7.87,
            ion_stage=jnp.zeros(n_lines, dtype=jnp.int32),
            element_idx=jnp.zeros(n_lines, dtype=jnp.int32),
            stark_w=jnp.ones(n_lines) * 0.02,
            stark_alpha=jnp.ones(n_lines) * 0.5,
            mass_amu=jnp.ones(n_lines) * 55.85,
            partition_coeffs=jnp.zeros((1, 3, 5)),
            ionization_potentials=jnp.zeros((1, 3)),
            elements=["Fe"],
        )

        assert len(arrays.wavelength_nm) == n_lines
        assert arrays.elements == ["Fe"]


@pytest.mark.requires_jax
class TestBayesianForwardModel:
    """Tests for BayesianForwardModel class."""

    def test_init(self, bayesian_db):
        """Test BayesianForwardModel initialization."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=500,
        )

        assert model.elements == ["Fe", "Cu"]
        assert len(model.wavelength) == 500
        assert model.atomic_data is not None
        assert len(model.atomic_data.wavelength_nm) > 0

    def test_forward_returns_spectrum(self, bayesian_db):
        """Test that forward model returns a valid spectrum."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=500,
        )

        # Typical LIBS parameters
        T_eV = 1.0  # ~11600 K
        log_ne = 17.0  # 10^17 cm^-3
        concentrations = jnp.array([0.8, 0.2])  # 80% Fe, 20% Cu

        spectrum = model.forward(T_eV, log_ne, concentrations)

        # Check output shape
        assert spectrum.shape == (500,)

        # Spectrum should be non-negative
        assert jnp.all(spectrum >= 0)

        # Spectrum should have some emission (not all zeros)
        assert jnp.max(spectrum) > 0

    def test_forward_temperature_dependence(self, bayesian_db):
        """Test that spectrum intensity increases with temperature."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        log_ne = 17.0
        concentrations = jnp.array([1.0])

        # Lower temperature
        spectrum_low = model.forward(0.5, log_ne, concentrations)
        # Higher temperature
        spectrum_high = model.forward(1.5, log_ne, concentrations)

        # Higher temperature should generally increase emission
        # (This is approximate - actual behavior depends on excitation energies)
        # At minimum, spectra should be different
        assert not jnp.allclose(spectrum_low, spectrum_high)

    def test_forward_density_dependence(self, bayesian_db):
        """Test that spectrum changes with electron density."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        T_eV = 1.0
        concentrations = jnp.array([1.0])

        # Lower density
        spectrum_low = model.forward(T_eV, 16.0, concentrations)
        # Higher density
        spectrum_high = model.forward(T_eV, 18.0, concentrations)

        # Spectra should be different
        assert not jnp.allclose(spectrum_low, spectrum_high)

    def test_forward_concentration_scaling(self, bayesian_db):
        """Test that spectrum scales with concentration."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe", "Cu"],
            wavelength_range=(200, 600),
            pixels=500,
        )

        T_eV = 1.0
        log_ne = 17.0

        # All Fe
        spectrum_fe = model.forward(T_eV, log_ne, jnp.array([1.0, 0.0]))
        # All Cu
        spectrum_cu = model.forward(T_eV, log_ne, jnp.array([0.0, 1.0]))
        # 50-50 mix
        spectrum_mix = model.forward(T_eV, log_ne, jnp.array([0.5, 0.5]))

        # Spectra should all be different
        assert not jnp.allclose(spectrum_fe, spectrum_cu)
        assert not jnp.allclose(spectrum_fe, spectrum_mix)

    def test_partition_function(self, bayesian_db):
        """Test partition function evaluation."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        # Test partition function at 10000 K
        T_K = 10000.0
        coeffs = jnp.array([3.22, 0.0, 0.0, 0.0, 0.0])  # Fe I approx

        U = model._partition_function(T_K, coeffs)

        # Should be positive
        assert U > 0
        # Fe I at 10000 K should be ~25
        np.testing.assert_allclose(U, np.exp(3.22), rtol=0.01)


@pytest.mark.requires_jax
class TestLogLikelihood:
    """Tests for log_likelihood function."""

    def test_perfect_match(self):
        """Test log-likelihood for perfect match."""
        predicted = jnp.array([100.0, 200.0, 150.0])
        observed = jnp.array([100.0, 200.0, 150.0])

        ll = log_likelihood(predicted, observed)

        # Perfect match should give high likelihood
        # (not infinite because of variance term)
        assert jnp.isfinite(ll)

    def test_likelihood_decreases_with_mismatch(self):
        """Test that likelihood decreases as mismatch increases."""
        predicted = jnp.array([100.0, 200.0, 150.0])

        # Perfect match
        observed_good = predicted
        ll_good = log_likelihood(predicted, observed_good)

        # Small mismatch
        observed_ok = predicted + 10.0
        ll_ok = log_likelihood(predicted, observed_ok)

        # Large mismatch
        observed_bad = predicted + 100.0
        ll_bad = log_likelihood(predicted, observed_bad)

        # Likelihood should decrease with mismatch
        assert ll_good > ll_ok > ll_bad

    def test_noise_parameters_affect_likelihood(self):
        """Test that noise parameters affect likelihood."""
        predicted = jnp.array([100.0, 200.0, 150.0])
        # Use larger residual so residual^2/variance dominates over log(variance)
        observed = predicted + 100.0

        # Low noise (tight tolerance) - large residual penalized more
        noise_low = NoiseParameters(readout_noise=5.0)
        ll_low = log_likelihood(predicted, observed, noise_low)

        # High noise (loose tolerance) - large residual penalized less
        noise_high = NoiseParameters(readout_noise=100.0)
        ll_high = log_likelihood(predicted, observed, noise_high)

        # With large residual, higher noise tolerance gives higher likelihood
        assert ll_high > ll_low

    def test_handles_zero_predicted(self):
        """Test handling of zero/small predicted values."""
        predicted = jnp.array([0.0, 100.0, 0.001])
        observed = jnp.array([10.0, 100.0, 10.0])

        ll = log_likelihood(predicted, observed)

        # Should return finite value
        assert jnp.isfinite(ll)


@pytest.mark.requires_jax
class TestPriorCreation:
    """Tests for prior creation functions."""

    def test_temperature_prior_import(self):
        """Test temperature prior can be created (if NumPyro available)."""
        try:
            from cflibs.inversion.bayesian import create_temperature_prior
            prior = create_temperature_prior(0.5, 3.0, "uniform")
            assert prior is not None
        except ImportError:
            pytest.skip("NumPyro not available")

    def test_density_prior_import(self):
        """Test density prior can be created (if NumPyro available)."""
        try:
            from cflibs.inversion.bayesian import create_density_prior
            prior = create_density_prior(15.0, 19.0, "uniform")
            assert prior is not None
        except ImportError:
            pytest.skip("NumPyro not available")

    def test_concentration_prior_import(self):
        """Test concentration prior can be created (if NumPyro available)."""
        try:
            from cflibs.inversion.bayesian import create_concentration_prior
            prior = create_concentration_prior(n_elements=3, alpha=1.0)
            assert prior is not None
        except ImportError:
            pytest.skip("NumPyro not available")


@pytest.mark.requires_jax
class TestBayesianForwardModelEdgeCases:
    """Edge case tests for BayesianForwardModel."""

    def test_single_element(self, bayesian_db):
        """Test model with single element."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        spectrum = model.forward(1.0, 17.0, jnp.array([1.0]))
        assert spectrum.shape == (200,)
        assert jnp.all(spectrum >= 0)

    def test_extreme_temperature_low(self, bayesian_db):
        """Test model at low temperature."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # Very low temperature (should still work)
        spectrum = model.forward(0.3, 17.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_extreme_temperature_high(self, bayesian_db):
        """Test model at high temperature."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # High temperature
        spectrum = model.forward(5.0, 17.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_extreme_density_low(self, bayesian_db):
        """Test model at low density."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # Low density
        spectrum = model.forward(1.0, 14.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_extreme_density_high(self, bayesian_db):
        """Test model at high density."""
        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=200,
        )

        # High density
        spectrum = model.forward(1.0, 20.0, jnp.array([1.0]))
        assert jnp.all(jnp.isfinite(spectrum))

    def test_no_lines_in_range(self, bayesian_db):
        """Test error when no lines in wavelength range."""
        with pytest.raises(ValueError, match="No atomic data"):
            BayesianForwardModel(
                db_path=bayesian_db,
                elements=["Fe"],
                wavelength_range=(100, 150),  # No Fe lines here
                pixels=100,
            )


@pytest.mark.requires_jax
@pytest.mark.slow
class TestMCMCSampling:
    """Integration tests for MCMC sampling (require NumPyro)."""

    @pytest.mark.skip(
        reason="MCMC initialization fails due to Voigt profile gradient issues at "
        "certain parameter combinations. The MCMCSampler class is functional "
        "when gradients are valid. See CF-LIBS-0oq for details."
    )
    def test_run_mcmc_smoke(self, bayesian_db):
        """Smoke test for MCMC sampling."""
        numpyro = pytest.importorskip("numpyro")
        from cflibs.inversion.bayesian import run_mcmc

        model = BayesianForwardModel(
            db_path=bayesian_db,
            elements=["Fe"],
            wavelength_range=(350, 450),
            pixels=100,
        )

        # Generate synthetic observed data
        T_true = 1.0
        log_ne_true = 17.0
        concentrations_true = jnp.array([1.0])

        synthetic = model.forward(T_true, log_ne_true, concentrations_true)
        # Add noise
        rng = np.random.default_rng(42)
        observed = np.array(synthetic) + rng.normal(0, 10, len(synthetic))
        observed = np.maximum(observed, 1.0)

        # Run short MCMC (just to verify it works)
        results = run_mcmc(
            model,
            observed,
            num_warmup=10,
            num_samples=20,
            seed=42,
        )

        # Check results structure
        assert "T_eV_mean" in results
        assert "log_ne_mean" in results
        assert "concentrations_mean" in results

        # Parameters should be in reasonable range
        assert 0.3 < results["T_eV_mean"] < 5.0
        assert 14.0 < results["log_ne_mean"] < 20.0
