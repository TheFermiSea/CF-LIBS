"""
Bayesian inference for CF-LIBS analysis.

This module implements Bayesian forward modeling and inference for CF-LIBS,
including:
- JAX-compatible forward model with full physics (Saha-Boltzmann, Voigt, Stark)
- Physically motivated priors for plasma parameters
- Log-likelihood function with realistic noise model (Poisson + Gaussian)

The implementation is designed to work with NumPyro for MCMC sampling.

References:
- Tognoni et al., "CF-LIBS: State of the art" (2010)
- Ciucci et al., "New procedure for quantitative elemental analysis by LIBS" (1999)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from cflibs.core.constants import (
    SAHA_CONST_CM3,
    C_LIGHT,
    EV_TO_K,
    EV_TO_J,
    KB_EV,
)
from cflibs.core.logging_config import get_logger
from cflibs.atomic.database import AtomicDatabase

logger = get_logger("inversion.bayesian")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jit = lambda f: f

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False
    numpyro = None
    dist = None


# Physical constants
H_PLANCK = 6.626e-34  # Planck constant [J·s]
M_PROTON = 1.6726219e-27  # Proton mass [kg]

# Standard atomic masses for fallback [amu]
STANDARD_MASSES = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95, "K": 39.10, "Ca": 40.08,
    "Sc": 44.96, "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Mn": 54.94,
    "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
}


@dataclass
class AtomicDataArrays:
    """
    Atomic data stored as JAX arrays for efficient computation.

    All arrays are indexed by line number (n_lines,).
    """

    wavelength_nm: Any  # Line wavelengths [nm]
    aki: Any  # Einstein A coefficients [s^-1]
    ek_ev: Any  # Upper level energy [eV]
    gk: Any  # Upper level degeneracy
    ip_ev: Any  # Ionization potential of parent species [eV]
    ion_stage: Any  # Ionization stage (0=neutral, 1=singly ionized)
    element_idx: Any  # Element index
    stark_w: Any  # Stark width reference [nm]
    stark_alpha: Any  # Stark temperature exponent
    mass_amu: Any  # Atomic mass [amu]
    partition_coeffs: Any  # Partition function coefficients (n_elements, n_stages, 5)
    ionization_potentials: Any  # Ionization potentials (n_elements, n_stages)
    elements: List[str] = field(default_factory=list)


@dataclass
class NoiseParameters:
    """
    Noise model parameters for LIBS spectra.

    The noise model combines:
    - Poisson shot noise: sqrt(I) (signal-dependent)
    - Gaussian readout noise: constant (signal-independent)
    - Background noise: additive offset

    Attributes
    ----------
    readout_noise : float
        RMS readout noise in counts (default: 10)
    dark_current : float
        Dark current per pixel in counts (default: 1)
    gain : float
        Detector gain (counts/photon) (default: 1)
    """

    readout_noise: float = 10.0
    dark_current: float = 1.0
    gain: float = 1.0


@dataclass
class PriorConfig:
    """
    Configuration for Bayesian priors on plasma parameters.

    Attributes
    ----------
    T_eV_range : Tuple[float, float]
        Temperature range in eV (default: 0.5-3.0 eV, typical LIBS)
    log_ne_range : Tuple[float, float]
        Log10(electron density) range (default: 15-19, i.e., 10^15-10^19 cm^-3)
    concentration_alpha : float
        Dirichlet prior concentration parameter (default: 1.0, uniform on simplex)
    """

    T_eV_range: Tuple[float, float] = (0.5, 3.0)
    log_ne_range: Tuple[float, float] = (15.0, 19.0)
    concentration_alpha: float = 1.0


class BayesianForwardModel:
    """
    Bayesian forward model for CF-LIBS spectra.

    This class provides a JAX-compatible forward model that maps plasma
    parameters (T, n_e, concentrations) to synthetic spectra. The physics
    includes:
    - Saha-Boltzmann population distribution
    - Voigt line profiles (Humlicek W4 Faddeeva approximation)
    - Stark broadening with temperature scaling
    - Proper Doppler broadening with mass dependence

    Parameters
    ----------
    db_path : str
        Path to atomic database
    elements : List[str]
        Elements to include
    wavelength_range : Tuple[float, float]
        Wavelength range [nm]
    wavelength_grid : np.ndarray, optional
        Custom wavelength grid; if None, auto-generated
    instrument_fwhm_nm : float
        Instrument FWHM in nm (default: 0.05)
    """

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        wavelength_range: Tuple[float, float],
        wavelength_grid: Optional[np.ndarray] = None,
        pixels: int = 2048,
        instrument_fwhm_nm: float = 0.05,
    ):
        if not HAS_JAX:
            raise ImportError("JAX required. Install with: pip install jax jaxlib")

        self.elements = elements
        self.wavelength_range = wavelength_range
        self.instrument_fwhm_nm = instrument_fwhm_nm

        # Create wavelength grid
        if wavelength_grid is not None:
            self.wavelength = jnp.array(wavelength_grid)
        else:
            self.wavelength = jnp.linspace(
                wavelength_range[0], wavelength_range[1], pixels
            )

        # Load atomic data
        self.atomic_data = self._load_atomic_data(db_path)

        logger.info(
            f"BayesianForwardModel: {len(elements)} elements, "
            f"{len(self.wavelength)} wavelengths, "
            f"{len(self.atomic_data.wavelength_nm)} lines"
        )

    def _load_atomic_data(self, db_path: str) -> AtomicDataArrays:
        """Load atomic data from database into JAX arrays."""
        import pandas as pd
        import sqlite3

        # Open direct connection for data loading
        conn = sqlite3.connect(db_path)
        db = AtomicDatabase(db_path)

        # Query spectral lines
        placeholders = ",".join(["?"] * len(self.elements))
        query = f"""
            SELECT
                l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk,
                sp.ip_ev, l.stark_w, l.stark_alpha
            FROM lines l
            JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
            WHERE l.wavelength_nm BETWEEN ? AND ?
            AND l.element IN ({placeholders})
            ORDER BY l.wavelength_nm
        """
        params = [self.wavelength_range[0], self.wavelength_range[1]] + self.elements
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            raise ValueError(
                f"No atomic data for elements {self.elements} in "
                f"range {self.wavelength_range}"
            )

        # Map elements to indices
        el_map = {el: i for i, el in enumerate(self.elements)}
        df["el_idx"] = df["element"].map(el_map)

        # Get atomic masses
        element_masses = {}
        for el in self.elements:
            db_mass = db.get_atomic_mass(el)
            if db_mass is not None:
                element_masses[el] = db_mass
            elif el in STANDARD_MASSES:
                element_masses[el] = STANDARD_MASSES[el]
            else:
                element_masses[el] = 50.0
                logger.warning(f"No mass for {el}, using fallback 50 amu")
        df["mass_amu"] = df["element"].map(element_masses)

        # Load partition function coefficients
        max_stages = 3
        n_elements = len(self.elements)
        coeffs = np.zeros((n_elements, max_stages, 5), dtype=np.float32)
        ips = np.zeros((n_elements, max_stages), dtype=np.float32)

        # Default coefficients
        coeffs[:, 0, 0] = np.log(25.0)
        coeffs[:, 1, 0] = np.log(15.0)
        coeffs[:, 2, 0] = np.log(10.0)

        try:
            cursor = conn.cursor()

            # Load ionization potentials
            cursor.execute(
                f"SELECT element, sp_num, ip_ev FROM species_physics "
                f"WHERE element IN ({placeholders})",
                self.elements,
            )
            for row in cursor.fetchall():
                el, sp_num, ip_ev = row
                if el in el_map and ip_ev is not None:
                    el_idx = el_map[el]
                    stage_idx = sp_num - 1
                    if 0 <= stage_idx < max_stages:
                        ips[el_idx, stage_idx] = ip_ev

            # Load partition function coefficients
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='partition_functions'"
            )
            if cursor.fetchone():
                cursor.execute(
                    f"SELECT element, sp_num, a0, a1, a2, a3, a4 "
                    f"FROM partition_functions WHERE element IN ({placeholders})",
                    self.elements,
                )
                for row in cursor.fetchall():
                    el, sp_num, a0, a1, a2, a3, a4 = row
                    if el in el_map:
                        el_idx = el_map[el]
                        stage_idx = sp_num - 1
                        if 0 <= stage_idx < max_stages:
                            coeffs[el_idx, stage_idx] = [a0, a1, a2, a3, a4]
        except Exception as e:
            logger.warning(f"Failed to load physics data: {e}")
        finally:
            conn.close()

        # Handle missing Stark parameters
        stark_w_raw = df["stark_w"].fillna(float("nan")).values
        stark_alpha_raw = df["stark_alpha"].fillna(0.5).values

        return AtomicDataArrays(
            wavelength_nm=jnp.array(df["wavelength_nm"].values, dtype=jnp.float32),
            aki=jnp.array(df["aki"].values, dtype=jnp.float32),
            ek_ev=jnp.array(df["ek_ev"].values, dtype=jnp.float32),
            gk=jnp.array(df["gk"].values, dtype=jnp.float32),
            ip_ev=jnp.array(df["ip_ev"].values, dtype=jnp.float32),
            ion_stage=jnp.array(df["sp_num"].values - 1, dtype=jnp.int32),
            element_idx=jnp.array(df["el_idx"].values, dtype=jnp.int32),
            stark_w=jnp.array(stark_w_raw, dtype=jnp.float32),
            stark_alpha=jnp.array(stark_alpha_raw, dtype=jnp.float32),
            mass_amu=jnp.array(df["mass_amu"].values, dtype=jnp.float32),
            partition_coeffs=jnp.array(coeffs, dtype=jnp.float32),
            ionization_potentials=jnp.array(ips, dtype=jnp.float32),
            elements=self.elements,
        )

    def forward(
        self,
        T_eV: float,
        log_ne: float,
        concentrations: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute synthetic spectrum for given plasma parameters.

        Parameters
        ----------
        T_eV : float
            Temperature in eV
        log_ne : float
            Log10 of electron density [cm^-3]
        concentrations : array
            Element concentrations (must sum to 1)

        Returns
        -------
        array
            Synthetic spectrum intensity
        """
        n_e = 10.0 ** log_ne
        return self._compute_spectrum(T_eV, n_e, concentrations)

    @staticmethod
    def _partition_function(T_K: float, coeffs: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate polynomial partition function.

        log(U) = sum_i a_i * (log(T))^i  (Irwin form)
        """
        log_T = jnp.log(T_K)
        powers = jnp.array([1.0, log_T, log_T**2, log_T**3, log_T**4])
        log_U = jnp.sum(coeffs * powers, axis=-1)
        return jnp.exp(log_U)

    def _compute_spectrum(
        self,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute spectrum with full physics.

        Uses Saha-Boltzmann populations, Voigt profiles, and Stark broadening.
        """
        data = self.atomic_data
        T_K = T_eV * EV_TO_K

        # Partition functions for all elements and stages
        U0 = self._partition_function(T_K, data.partition_coeffs[:, 0])
        U1 = self._partition_function(T_K, data.partition_coeffs[:, 1])

        # Ionization potentials for neutral -> ion transition
        IP_I = data.ionization_potentials[:, 0]

        # Saha ratio: n_ion / n_neutral
        saha_factor = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5)
        ratio_ion_neutral = (
            2.0 * saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV)
        )

        # Population fractions
        frac_neutral = 1.0 / (1.0 + ratio_ion_neutral)
        frac_ion = ratio_ion_neutral / (1.0 + ratio_ion_neutral)

        # Per-line quantities
        el_idx = data.element_idx
        ion_stage = data.ion_stage

        pop_fraction = jnp.where(ion_stage == 0, frac_neutral[el_idx], frac_ion[el_idx])
        U_val = jnp.where(ion_stage == 0, U0[el_idx], U1[el_idx])

        # Species number density
        element_conc = concentrations[el_idx]
        N_species_total = element_conc * n_e
        N_species = N_species_total * pop_fraction

        # Boltzmann upper level population
        n_upper = N_species * (data.gk / U_val) * jnp.exp(-data.ek_ev / T_eV)

        # Line emissivity: epsilon = (hc / 4pi * lambda) * A * n_upper
        epsilon = (H_PLANCK * C_LIGHT / (4 * jnp.pi * data.wavelength_nm * 1e-9)) * data.aki * n_upper

        # --- Line Broadening ---
        # Doppler width
        mass_kg = data.mass_amu * M_PROTON
        sigma_doppler = data.wavelength_nm * jnp.sqrt(
            2.0 * T_eV * EV_TO_J / (mass_kg * C_LIGHT**2)
        )

        # Instrument broadening
        sigma_inst = self.instrument_fwhm_nm / 2.355
        sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

        # Stark broadening (HWHM)
        REF_NE = 1.0e16
        REF_T_EV = 0.86173  # 10000 K

        # Estimate Stark width for missing values
        binding_energy = jnp.maximum(data.ip_ev - data.ek_ev, 0.1)
        n_eff = (ion_stage + 1) * jnp.sqrt(13.605 / binding_energy)
        w_est = 2.0e-5 * (data.wavelength_nm / 500.0) ** 2 * (n_eff**4)
        w_est = jnp.clip(w_est, 0.0001, 0.5)
        w_ref = jnp.where(jnp.isnan(data.stark_w), w_est, data.stark_w)

        factor_ne = n_e / REF_NE
        factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -data.stark_alpha)
        gamma_stark = w_ref * factor_ne * factor_T

        # --- Voigt Profile (Humlicek W4 approximation) ---
        diff = self.wavelength[:, None] - data.wavelength_nm[None, :]
        z = (diff + 1j * gamma_stark) / (sigma_total * jnp.sqrt(2.0))

        x_h = jnp.real(z)
        y_h = jnp.abs(jnp.imag(z))
        s = jnp.abs(x_h) + y_h
        t = y_h - 1j * x_h

        # Region 1: s >= 15
        w_r1 = t * 0.5641896 / (0.5 + t * t)

        # Region 2: 5.5 <= s < 15
        u = t * t
        w_r2 = t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))

        # Region 3: s < 5.5 and y >= 0.195*|x| - 0.176
        w_r3 = (
            16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))
        ) / (
            16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )

        # Region 4: s < 5.5 and y < 0.195*|x| - 0.176
        w_r4 = jnp.exp(u) - t * (
            36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419)))))
        ) / (
            32066.6 - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u))))))
        )

        w_z = jnp.where(
            s >= 15.0,
            w_r1,
            jnp.where(
                s >= 5.5,
                w_r2,
                jnp.where(
                    y_h >= 0.195 * jnp.abs(x_h) - 0.176,
                    w_r3,
                    w_r4,
                ),
            ),
        )

        profile = jnp.real(w_z) / (sigma_total * jnp.sqrt(2.0 * jnp.pi))

        # Sum line contributions
        intensity = jnp.sum(epsilon * profile, axis=1)

        return intensity


def log_likelihood(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    noise_params: NoiseParameters = NoiseParameters(),
) -> float:
    """
    Compute log-likelihood for observed spectrum given predicted.

    The noise model combines Poisson shot noise and Gaussian readout noise:
        variance = predicted / gain + readout_noise^2 + dark_current

    This is the standard CCD noise model for spectroscopy.

    Parameters
    ----------
    predicted : array
        Predicted spectrum from forward model
    observed : array
        Observed spectrum (counts)
    noise_params : NoiseParameters
        Noise model parameters

    Returns
    -------
    float
        Log-likelihood value
    """
    # Ensure positive predicted values
    pred_safe = jnp.maximum(predicted, 1e-10)

    # Variance: Poisson (shot) + Gaussian (readout) + dark current
    variance = (
        pred_safe / noise_params.gain
        + noise_params.readout_noise**2
        + noise_params.dark_current
    )

    # Gaussian log-likelihood
    residual = observed - pred_safe
    log_lik = -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi * variance) + residual**2 / variance
    )

    return log_lik


def bayesian_model(
    forward_model: BayesianForwardModel,
    observed: jnp.ndarray,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
):
    """
    NumPyro probabilistic model for CF-LIBS Bayesian inference.

    This defines the full Bayesian model with priors and likelihood.
    Use with MCMC or variational inference.

    Parameters
    ----------
    forward_model : BayesianForwardModel
        Forward model instance
    observed : array
        Observed spectrum
    prior_config : PriorConfig
        Prior configuration
    noise_params : NoiseParameters
        Noise model parameters
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro")

    n_elements = len(forward_model.elements)

    # --- Priors ---
    # Temperature: uniform on physically realistic range
    T_eV = numpyro.sample(
        "T_eV",
        dist.Uniform(prior_config.T_eV_range[0], prior_config.T_eV_range[1]),
    )

    # Electron density: log-uniform (Jeffreys prior for scale parameter)
    log_ne = numpyro.sample(
        "log_ne",
        dist.Uniform(prior_config.log_ne_range[0], prior_config.log_ne_range[1]),
    )

    # Concentrations: Dirichlet prior (ensures sum to 1)
    alpha = jnp.ones(n_elements) * prior_config.concentration_alpha
    concentrations = numpyro.sample("concentrations", dist.Dirichlet(alpha))

    # --- Forward Model ---
    predicted = forward_model.forward(T_eV, log_ne, concentrations)

    # --- Likelihood ---
    # Variance model: Poisson + readout noise
    pred_safe = jnp.maximum(predicted, 1e-10)
    variance = (
        pred_safe / noise_params.gain
        + noise_params.readout_noise**2
        + noise_params.dark_current
    )
    sigma = jnp.sqrt(variance)

    # Observe data
    numpyro.sample("obs", dist.Normal(pred_safe, sigma), obs=observed)


def run_mcmc(
    forward_model: BayesianForwardModel,
    observed: np.ndarray,
    prior_config: PriorConfig = PriorConfig(),
    noise_params: NoiseParameters = NoiseParameters(),
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run MCMC sampling for Bayesian CF-LIBS inference.

    Parameters
    ----------
    forward_model : BayesianForwardModel
        Forward model instance
    observed : array
        Observed spectrum
    prior_config : PriorConfig
        Prior configuration
    noise_params : NoiseParameters
        Noise model parameters
    num_warmup : int
        Number of warmup samples
    num_samples : int
        Number of posterior samples
    num_chains : int
        Number of MCMC chains
    seed : int
        Random seed

    Returns
    -------
    dict
        MCMC results including posterior samples
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro")

    import jax.random as random

    observed_jax = jnp.array(observed)

    # Create NUTS sampler
    kernel = NUTS(
        lambda obs: bayesian_model(forward_model, obs, prior_config, noise_params)
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    # Run sampling
    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, observed_jax)

    # Get results
    samples = mcmc.get_samples()

    # Compute summary statistics
    results = {
        "samples": samples,
        "T_eV_mean": float(jnp.mean(samples["T_eV"])),
        "T_eV_std": float(jnp.std(samples["T_eV"])),
        "log_ne_mean": float(jnp.mean(samples["log_ne"])),
        "log_ne_std": float(jnp.std(samples["log_ne"])),
        "concentrations_mean": {
            el: float(jnp.mean(samples["concentrations"][:, i]))
            for i, el in enumerate(forward_model.elements)
        },
        "concentrations_std": {
            el: float(jnp.std(samples["concentrations"][:, i]))
            for i, el in enumerate(forward_model.elements)
        },
    }

    # Add derived quantities
    results["n_e_mean"] = 10.0 ** results["log_ne_mean"]
    results["T_K_mean"] = results["T_eV_mean"] * EV_TO_K

    logger.info(
        f"MCMC complete: T = {results['T_eV_mean']:.3f} +/- {results['T_eV_std']:.3f} eV, "
        f"n_e = {results['n_e_mean']:.2e} cm^-3"
    )

    return results


# --- Convenience functions for priors (CF-LIBS-zbs) ---


def create_temperature_prior(
    T_min_eV: float = 0.5,
    T_max_eV: float = 3.0,
    prior_type: str = "uniform",
) -> Any:
    """
    Create temperature prior distribution.

    Parameters
    ----------
    T_min_eV : float
        Minimum temperature [eV]
    T_max_eV : float
        Maximum temperature [eV]
    prior_type : str
        Prior type: 'uniform', 'normal', 'truncnorm'

    Returns
    -------
    numpyro.distribution
        Prior distribution
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required")

    if prior_type == "uniform":
        return dist.Uniform(T_min_eV, T_max_eV)
    elif prior_type == "normal":
        # Centered on typical LIBS temperature
        mean = (T_min_eV + T_max_eV) / 2
        std = (T_max_eV - T_min_eV) / 4
        return dist.TruncatedNormal(mean, std, low=T_min_eV, high=T_max_eV)
    else:
        return dist.Uniform(T_min_eV, T_max_eV)


def create_density_prior(
    log_ne_min: float = 15.0,
    log_ne_max: float = 19.0,
    prior_type: str = "uniform",
) -> Any:
    """
    Create electron density prior distribution.

    Log-uniform (Jeffreys) prior is appropriate for scale parameters.

    Parameters
    ----------
    log_ne_min : float
        Log10 of minimum density [cm^-3]
    log_ne_max : float
        Log10 of maximum density [cm^-3]
    prior_type : str
        Prior type: 'uniform' (log-uniform), 'normal'

    Returns
    -------
    numpyro.distribution
        Prior distribution for log10(n_e)
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required")

    if prior_type == "uniform":
        # Log-uniform = uniform on log scale = Jeffreys prior
        return dist.Uniform(log_ne_min, log_ne_max)
    elif prior_type == "normal":
        mean = (log_ne_min + log_ne_max) / 2
        std = (log_ne_max - log_ne_min) / 4
        return dist.TruncatedNormal(mean, std, low=log_ne_min, high=log_ne_max)
    else:
        return dist.Uniform(log_ne_min, log_ne_max)


def create_concentration_prior(
    n_elements: int,
    alpha: float = 1.0,
    known_concentrations: Optional[Dict[int, float]] = None,
) -> Any:
    """
    Create concentration prior distribution.

    Uses Dirichlet distribution which naturally enforces:
    - All concentrations positive
    - Concentrations sum to 1

    Parameters
    ----------
    n_elements : int
        Number of elements
    alpha : float
        Dirichlet concentration parameter:
        - alpha = 1: Uniform on simplex
        - alpha > 1: Peaked at center (equal concentrations)
        - alpha < 1: Peaked at corners (sparse compositions)
    known_concentrations : dict, optional
        Known concentration constraints {element_idx: value}

    Returns
    -------
    numpyro.distribution
        Prior distribution
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required")

    alphas = jnp.ones(n_elements) * alpha

    # Adjust for known concentrations (informative prior)
    if known_concentrations:
        for idx, value in known_concentrations.items():
            # Increase alpha for known elements to peak near their value
            alphas = alphas.at[idx].set(alpha * (1 + 10 * value))

    return dist.Dirichlet(alphas)
