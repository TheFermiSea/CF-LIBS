"""
Hybrid inversion combining manifold lookup with gradient descent.

This module implements a two-stage inversion strategy:
1. Coarse search: Manifold nearest-neighbor (cosine similarity) for initial guess
2. Fine tuning: JAX autodiff + L-BFGS optimization from coarse guess

This approach combines the global search capability of the manifold lookup
(avoiding local minima) with the precision of gradient-based optimization.

References:
- Tognoni et al., "CF-LIBS: State of the art" (2010)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from cflibs.core.constants import KB_EV, EV_TO_J, EV_TO_K, C_LIGHT
from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.hybrid")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad
    from jax.scipy.optimize import minimize as jax_minimize

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


@dataclass
class HybridInversionResult:
    """
    Result of hybrid inversion.

    Attributes
    ----------
    temperature_eV : float
        Recovered temperature in eV
    electron_density_cm3 : float
        Recovered electron density in cm^-3
    concentrations : Dict[str, float]
        Recovered elemental concentrations
    coarse_temperature_eV : float
        Initial temperature from manifold lookup
    coarse_electron_density_cm3 : float
        Initial electron density from manifold lookup
    coarse_concentrations : Dict[str, float]
        Initial concentrations from manifold lookup
    coarse_similarity : float
        Similarity score from manifold lookup
    final_residual : float
        Final chi-squared residual
    converged : bool
        Whether optimization converged
    iterations : int
        Number of optimization iterations
    method : str
        Optimization method used
    """

    temperature_eV: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    coarse_temperature_eV: float
    coarse_electron_density_cm3: float
    coarse_concentrations: Dict[str, float]
    coarse_similarity: float
    final_residual: float
    converged: bool
    iterations: int
    method: str = "L-BFGS-B"
    metadata: Dict = field(default_factory=dict)

    @property
    def temperature_K(self) -> float:
        """Temperature in Kelvin."""
        return self.temperature_eV * EV_TO_K

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Hybrid Inversion Result",
            f"  Coarse (manifold): T={self.coarse_temperature_eV:.3f} eV, "
            f"n_e={self.coarse_electron_density_cm3:.2e} cm^-3, "
            f"similarity={self.coarse_similarity:.4f}",
            f"  Fine (optimized):  T={self.temperature_eV:.3f} eV, "
            f"n_e={self.electron_density_cm3:.2e} cm^-3",
            f"  Concentrations:",
        ]
        for el, c in self.concentrations.items():
            lines.append(f"    {el}: {c:.4f}")
        lines.append(
            f"  Converged: {self.converged} ({self.iterations} iterations, "
            f"residual={self.final_residual:.2e})"
        )
        return "\n".join(lines)


class HybridInverter:
    """
    Hybrid inversion combining manifold lookup with gradient descent.

    Algorithm:
    1. Use manifold cosine similarity to find initial guess (T0, ne0, C0)
    2. Define loss function: L = sum((measured - forward(T, ne, C))^2 / sigma^2)
    3. Use JAX autodiff + L-BFGS to minimize loss from initial guess

    This approach:
    - Avoids local minima (manifold provides global search)
    - Achieves high precision (gradient descent refines)
    - Is faster than pure optimization (good starting point)
    - Is more robust than pure manifold (not limited to grid points)

    Parameters
    ----------
    manifold : ManifoldLoader
        Pre-computed spectral manifold
    forward_model : callable, optional
        Forward model function: (T_eV, n_e, concentrations) -> spectrum
        If not provided, uses simple Gaussian model
    """

    def __init__(
        self,
        manifold,
        forward_model: Optional[Callable] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for hybrid inversion. Install with: pip install jax jaxlib"
            )

        self.manifold = manifold
        self.forward_model = forward_model or self._default_forward_model
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Extract manifold info
        self.wavelength = jnp.array(manifold.wavelength)
        self.elements = list(manifold.elements)
        self.n_elements = len(self.elements)

        logger.info(
            f"HybridInverter initialized: {self.n_elements} elements, "
            f"{len(self.wavelength)} wavelengths"
        )

    def invert(
        self,
        measured_spectrum: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        use_manifold_init: bool = True,
        initial_guess: Optional[Dict] = None,
        bounds: Optional[Dict] = None,
    ) -> HybridInversionResult:
        """
        Perform hybrid inversion on measured spectrum.

        Parameters
        ----------
        measured_spectrum : array
            Measured spectrum (must match manifold wavelength grid)
        uncertainties : array, optional
            Spectral uncertainties (defaults to sqrt(spectrum))
        method : str
            Optimization method: 'L-BFGS-B', 'BFGS', 'CG'
        use_manifold_init : bool
            Use manifold lookup for initial guess
        initial_guess : dict, optional
            Override initial guess (T_eV, n_e, concentrations)
        bounds : dict, optional
            Parameter bounds

        Returns
        -------
        HybridInversionResult
            Inversion results
        """
        measured = jnp.array(measured_spectrum)

        # Set default uncertainties
        if uncertainties is None:
            uncertainties = jnp.sqrt(jnp.maximum(measured, 1.0))
        else:
            uncertainties = jnp.array(uncertainties)

        # Set default bounds
        if bounds is None:
            bounds = {
                "T_eV": (0.3, 3.0),  # Typical LIBS range
                "n_e": (1e15, 1e19),
                "concentration": (0.0, 1.0),
            }

        # Stage 1: Coarse search via manifold
        if use_manifold_init and initial_guess is None:
            coarse_idx, coarse_similarity, coarse_params = self.manifold.find_nearest_spectrum(
                np.array(measured), method="cosine", use_jax=True
            )

            coarse_T = coarse_params["T_eV"]
            coarse_ne = coarse_params["n_e_cm3"]
            coarse_conc = {el: coarse_params.get(el, 0.0) for el in self.elements}

            logger.info(
                f"Coarse search: T={coarse_T:.3f} eV, n_e={coarse_ne:.2e}, "
                f"similarity={coarse_similarity:.4f}"
            )
        elif initial_guess is not None:
            coarse_T = initial_guess.get("T_eV", 1.0)
            coarse_ne = initial_guess.get("n_e", 1e17)
            coarse_conc = {
                el: initial_guess.get(el, 1.0 / self.n_elements) for el in self.elements
            }
            coarse_similarity = 0.0
        else:
            # Default initial guess
            coarse_T = 1.0
            coarse_ne = 1e17
            coarse_conc = {el: 1.0 / self.n_elements for el in self.elements}
            coarse_similarity = 0.0

        # Stage 2: Fine tuning via gradient descent
        # Pack parameters into array for optimization
        # [log(T), log(n_e), softmax(concentrations)]
        x0 = self._pack_params(coarse_T, coarse_ne, coarse_conc)

        # Define loss function
        def loss_fn(x):
            T_eV, n_e, conc_arr = self._unpack_params(x)
            predicted = self.forward_model(T_eV, n_e, conc_arr, self.wavelength)
            residuals = (measured - predicted) / uncertainties
            return jnp.sum(residuals**2)

        # Run optimization
        try:
            result = jax_minimize(
                loss_fn,
                x0,
                method=method.lower().replace("-", ""),  # JAX uses 'bfgs' not 'L-BFGS-B'
                options={"maxiter": self.max_iterations},
            )

            final_x = result.x
            final_loss = float(result.fun)
            converged = result.success
            iterations = result.nit if hasattr(result, "nit") else self.max_iterations
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using coarse result")
            final_x = x0
            final_loss = float(loss_fn(x0))
            converged = False
            iterations = 0

        # Unpack final parameters
        final_T, final_ne, final_conc_arr = self._unpack_params(final_x)
        final_concentrations = {el: float(final_conc_arr[i]) for i, el in enumerate(self.elements)}

        logger.info(
            f"Fine tuning: T={final_T:.3f} eV, n_e={final_ne:.2e}, "
            f"residual={final_loss:.2e}, converged={converged}"
        )

        return HybridInversionResult(
            temperature_eV=float(final_T),
            electron_density_cm3=float(final_ne),
            concentrations=final_concentrations,
            coarse_temperature_eV=coarse_T,
            coarse_electron_density_cm3=coarse_ne,
            coarse_concentrations=coarse_conc,
            coarse_similarity=coarse_similarity,
            final_residual=final_loss,
            converged=converged,
            iterations=iterations,
            method=method,
        )

    def _pack_params(
        self, T_eV: float, n_e: float, concentrations: Dict[str, float]
    ) -> jnp.ndarray:
        """Pack parameters into optimization vector."""
        # Use log scale for T and n_e (ensures positivity)
        log_T = jnp.log(T_eV)
        log_ne = jnp.log(n_e)

        # Use log-softmax for concentrations (ensures sum to 1, all positive)
        conc_arr = jnp.array([concentrations.get(el, 0.01) for el in self.elements])
        conc_arr = jnp.maximum(conc_arr, 1e-6)  # Avoid log(0)
        log_conc = jnp.log(conc_arr)  # Will be softmax'ed during unpack

        return jnp.concatenate([jnp.array([log_T, log_ne]), log_conc])

    def _unpack_params(self, x: jnp.ndarray) -> Tuple[float, float, jnp.ndarray]:
        """Unpack optimization vector to parameters."""
        log_T = x[0]
        log_ne = x[1]
        log_conc = x[2:]

        T_eV = jnp.exp(log_T)
        n_e = jnp.exp(log_ne)

        # Apply softmax to ensure concentrations sum to 1
        conc_arr = jax.nn.softmax(log_conc)

        return T_eV, n_e, conc_arr

    def _default_forward_model(
        self,
        T_eV: float,
        n_e: float,
        concentrations: jnp.ndarray,
        wavelength: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Simple forward model for testing.

        This is a simplified Gaussian emission model. For production use,
        provide a full physics-based forward model.

        Parameters
        ----------
        T_eV : float
            Temperature in eV
        n_e : float
            Electron density
        concentrations : array
            Element concentrations
        wavelength : array
            Wavelength grid

        Returns
        -------
        array
            Predicted spectrum
        """
        # This is a placeholder - in production, use the ManifoldGenerator's
        # _compute_spectrum_snapshot or a similar physics-based model

        # Generate synthetic lines for demonstration
        n_wl = len(wavelength)
        wl_min = float(wavelength[0])
        wl_max = float(wavelength[-1])

        spectrum = jnp.zeros(n_wl)

        # Add Gaussian emission lines for each element
        for i, el in enumerate(self.elements):
            c = concentrations[i]

            # Generate some synthetic lines per element
            n_lines = 5
            for j in range(n_lines):
                # Line position depends on element index
                center = wl_min + (wl_max - wl_min) * (i * n_lines + j + 0.5) / (
                    len(self.elements) * n_lines
                )

                # Intensity from Boltzmann distribution
                E_k = 2.0 + j * 0.5  # Fake upper energy
                boltzmann = jnp.exp(-E_k / (KB_EV * T_eV * EV_TO_K))
                intensity = c * boltzmann * 1000.0

                # Doppler width
                sigma = center * jnp.sqrt(2.0 * T_eV * EV_TO_J / (50.0 * 1.67e-27 * C_LIGHT**2))
                sigma = jnp.maximum(sigma, 0.01)

                # Add Gaussian
                profile = jnp.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
                spectrum = spectrum + intensity * profile / (sigma * jnp.sqrt(2 * jnp.pi))

        return spectrum


class SpectralFitter:
    """
    Gradient-based spectral fitting using JAX autodiff.

    This class provides fine-grained control over spectral fitting without
    requiring a pre-computed manifold. Use when you have a forward model
    and want to fit plasma parameters directly.

    Parameters
    ----------
    forward_model : callable
        Forward model: (T_eV, n_e, concentrations, wavelength) -> spectrum
    elements : List[str]
        Element names
    wavelength : array
        Wavelength grid
    """

    def __init__(
        self,
        forward_model: Callable,
        elements: List[str],
        wavelength: np.ndarray,
    ):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for spectral fitting. Install with: pip install jax jaxlib"
            )

        self.forward_model = forward_model
        self.elements = elements
        self.wavelength = jnp.array(wavelength)
        self.n_elements = len(elements)

    def fit(
        self,
        measured_spectrum: np.ndarray,
        initial_T_eV: float = 1.0,
        initial_n_e: float = 1e17,
        initial_concentrations: Optional[Dict[str, float]] = None,
        uncertainties: Optional[np.ndarray] = None,
        method: str = "BFGS",
        max_iterations: int = 100,
    ) -> HybridInversionResult:
        """
        Fit spectrum to data using gradient descent.

        Parameters
        ----------
        measured_spectrum : array
            Measured spectrum
        initial_T_eV : float
            Initial temperature guess
        initial_n_e : float
            Initial electron density guess
        initial_concentrations : dict, optional
            Initial concentration guess per element
        uncertainties : array, optional
            Spectral uncertainties
        method : str
            Optimization method
        max_iterations : int
            Maximum iterations

        Returns
        -------
        HybridInversionResult
            Fitting results
        """
        measured = jnp.array(measured_spectrum)

        if uncertainties is None:
            uncertainties = jnp.sqrt(jnp.maximum(measured, 1.0))
        else:
            uncertainties = jnp.array(uncertainties)

        if initial_concentrations is None:
            initial_concentrations = {el: 1.0 / self.n_elements for el in self.elements}

        # Pack initial guess
        x0 = self._pack(initial_T_eV, initial_n_e, initial_concentrations)

        def loss_fn(x):
            T, ne, conc = self._unpack(x)
            predicted = self.forward_model(T, ne, conc, self.wavelength)
            residuals = (measured - predicted) / uncertainties
            return jnp.sum(residuals**2)

        try:
            result = jax_minimize(
                loss_fn,
                x0,
                method=method.lower(),
                options={"maxiter": max_iterations},
            )

            final_T, final_ne, final_conc = self._unpack(result.x)
            converged = result.success
            iterations = result.nit if hasattr(result, "nit") else max_iterations
            final_loss = float(result.fun)
        except Exception as e:
            logger.warning(f"Fitting failed: {e}")
            final_T = initial_T_eV
            final_ne = initial_n_e
            final_conc = jnp.array(list(initial_concentrations.values()))
            converged = False
            iterations = 0
            final_loss = float(loss_fn(x0))

        return HybridInversionResult(
            temperature_eV=float(final_T),
            electron_density_cm3=float(final_ne),
            concentrations={
                el: float(final_conc[i]) for i, el in enumerate(self.elements)
            },
            coarse_temperature_eV=initial_T_eV,
            coarse_electron_density_cm3=initial_n_e,
            coarse_concentrations=initial_concentrations,
            coarse_similarity=0.0,
            final_residual=final_loss,
            converged=converged,
            iterations=iterations,
            method=method,
        )

    def _pack(
        self, T_eV: float, n_e: float, concentrations: Dict[str, float]
    ) -> jnp.ndarray:
        """Pack parameters."""
        log_T = jnp.log(T_eV)
        log_ne = jnp.log(n_e)
        conc_arr = jnp.array([concentrations.get(el, 0.01) for el in self.elements])
        conc_arr = jnp.maximum(conc_arr, 1e-6)
        log_conc = jnp.log(conc_arr)
        return jnp.concatenate([jnp.array([log_T, log_ne]), log_conc])

    def _unpack(self, x: jnp.ndarray) -> Tuple[float, float, jnp.ndarray]:
        """Unpack parameters."""
        T_eV = jnp.exp(x[0])
        n_e = jnp.exp(x[1])
        conc = jax.nn.softmax(x[2:])
        return T_eV, n_e, conc
