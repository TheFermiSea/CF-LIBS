"""
JAX-based manifold generator for high-throughput CF-LIBS.

This module implements GPU-accelerated generation of pre-computed spectral
manifolds using JAX. The manifold enables fast inference by pre-calculating
spectra for all parameter combinations of interest.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import time
from pathlib import Path

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

    # Define dummy decorators to allow class definition
    def jit(func):
        return func

    def vmap(func, *args, **kwargs):
        return func


from cflibs.core.constants import SAHA_CONST_CM3, C_LIGHT, EV_TO_K
from cflibs.atomic.database import AtomicDatabase
from cflibs.manifold.config import ManifoldConfig
from cflibs.core.logging_config import get_logger
from cflibs.plasma.partition import polynomial_partition_function_jax

logger = get_logger("manifold.generator")


class ManifoldGenerator:
    """
    Generator for pre-computed spectral manifolds.

    This class generates a high-dimensional lookup table of synthetic spectra
    using JAX for GPU acceleration. The manifold covers a parameter space
    defined by temperature, electron density, and element concentrations.

    The generated manifold can be used for fast inference by finding the
    nearest matching spectrum rather than solving physics equations at runtime.
    """

    def __init__(self, config: ManifoldConfig):
        """
        Initialize manifold generator.

        Parameters
        ----------
        config : ManifoldConfig
            Configuration for manifold generation

        Raises
        ------
        ImportError
            If JAX is not installed
        """
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for manifold generation. " "Install with: pip install jax jaxlib"
            )

        config.validate()
        self.config = config

        # Load atomic database
        self.atomic_db = AtomicDatabase(config.db_path)

        # Load atomic data into JAX arrays
        self.atomic_data = self._load_atomic_data()

        logger.info(
            f"Initialized ManifoldGenerator: {len(config.elements)} elements, "
            f"λ=[{config.wavelength_range[0]:.1f}, {config.wavelength_range[1]:.1f}] nm"
        )

    def _load_atomic_data(self) -> Tuple:
        """
        Load atomic data from database and convert to JAX arrays.

        Returns
        -------
        Tuple
            Atomic data as JAX arrays:
            (lines_wl, lines_aki, lines_ek, lines_gk, lines_ip, lines_z, lines_el_idx,
             partition_coeffs, ionization_potentials)
        """
        import pandas as pd

        logger.info("Loading atomic data from database...")

        # Build query for all elements
        placeholders = ",".join(["?"] * len(self.config.elements))
        query = f"""
            SELECT 
                l.element, l.sp_num, l.wavelength_nm, l.aki, l.ek_ev, l.gk, 
                sp.ip_ev
            FROM lines l
            JOIN species_physics sp ON l.element = sp.element AND l.sp_num = sp.sp_num
            WHERE l.wavelength_nm BETWEEN ? AND ?
            AND l.element IN ({placeholders})
            ORDER BY l.wavelength_nm
        """
        params = [
            self.config.wavelength_range[0],
            self.config.wavelength_range[1],
        ] + self.config.elements

        df = pd.read_sql_query(query, self.atomic_db.conn, params=params)

        if df.empty:
            raise ValueError(
                f"No atomic data found for elements {self.config.elements} "
                f"in wavelength range {self.config.wavelength_range}"
            )

        # Map element names to indices
        el_map = {el: i for i, el in enumerate(self.config.elements)}
        df["el_idx"] = df["element"].map(el_map)

        # Convert to JAX arrays
        lines_wl = jnp.array(df["wavelength_nm"].values, dtype=jnp.float32)
        lines_aki = jnp.array(df["aki"].values, dtype=jnp.float32)
        lines_ek = jnp.array(df["ek_ev"].values, dtype=jnp.float32)
        lines_gk = jnp.array(df["gk"].values, dtype=jnp.float32)
        lines_ip = jnp.array(df["ip_ev"].values, dtype=jnp.float32)
        lines_z = jnp.array(df["sp_num"].values - 1, dtype=jnp.int32)  # 0=neutral, 1=ion
        lines_el_idx = jnp.array(df["el_idx"].values, dtype=jnp.int32)

        logger.info(f"Loaded {len(df)} spectral lines")

        # --- Load Partition Function Coefficients & Ionization Potentials ---
        # Shapes:
        # partition_coeffs: (num_elements, max_stages, 5)
        # ionization_potentials: (num_elements, max_stages)
        max_stages = 3
        num_elements = len(self.config.elements)

        coeffs = np.zeros((num_elements, max_stages, 5), dtype=np.float32)
        ips = np.zeros((num_elements, max_stages), dtype=np.float32)

        # Set defaults for coeffs (approximate log(U))
        coeffs[:, 0, 0] = np.log(25.0)
        coeffs[:, 1, 0] = np.log(15.0)
        coeffs[:, 2, 0] = np.log(10.0)

        # Load Physics Data (IPs and Coeffs)
        try:
            cursor = self.atomic_db.conn.cursor()

            # Load IPs
            ip_query = f"""
                SELECT element, sp_num, ip_ev
                FROM species_physics
                WHERE element IN ({placeholders})
            """
            cursor.execute(ip_query, self.config.elements)
            for row in cursor.fetchall():
                el, sp_num, ip_ev = row
                if el in el_map and ip_ev is not None:
                    el_idx = el_map[el]
                    stage_idx = sp_num - 1
                    if 0 <= stage_idx < max_stages:
                        ips[el_idx, stage_idx] = ip_ev

            # Load Partition Coeffs
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='partition_functions'"
            )
            if cursor.fetchone():
                pf_query = f"""
                    SELECT element, sp_num, a0, a1, a2, a3, a4
                    FROM partition_functions
                    WHERE element IN ({placeholders})
                """
                cursor.execute(pf_query, self.config.elements)
                count = 0
                for row in cursor.fetchall():
                    el, sp_num, a0, a1, a2, a3, a4 = row
                    if el in el_map:
                        el_idx = el_map[el]
                        stage_idx = sp_num - 1
                        if 0 <= stage_idx < max_stages:
                            coeffs[el_idx, stage_idx] = [a0, a1, a2, a3, a4]
                            count += 1
                logger.info(f"Loaded partition coefficients for {count} species")

        except Exception as e:
            logger.warning(f"Failed to load physics data: {e}")

        partition_coeffs = jnp.array(coeffs, dtype=jnp.float32)
        ionization_potentials = jnp.array(ips, dtype=jnp.float32)

        return (
            lines_wl,
            lines_aki,
            lines_ek,
            lines_gk,
            lines_ip,
            lines_z,
            lines_el_idx,
            partition_coeffs,
            ionization_potentials,
        )

        @staticmethod
        @jit
        def _saha_eggert_solver(
            T_eV: float,
            n_e: float,
            concentration_map: jnp.ndarray,
            lines_ip: jnp.ndarray,
            lines_z: jnp.ndarray,
            lines_el_idx: jnp.ndarray,
            lines_ek: jnp.ndarray,
            lines_gk: jnp.ndarray,
            partition_coeffs: jnp.ndarray,
            ionization_potentials: jnp.ndarray,
        ) -> jnp.ndarray:
            """

            Vectorized Saha-Eggert solver for JAX.



            Calculates upper level populations for all lines simultaneously.



            Parameters

            ----------

            T_eV : float

                Electron temperature in eV

            n_e : float

                Electron density in cm^-3

            concentration_map : array

                Element concentrations

            lines_* : arrays

                Atomic data arrays

            partition_coeffs : array

                Partition function coefficients (num_elements, max_stages, 5)

            ionization_potentials : array

                Ionization potentials (num_elements, max_stages)



            Returns

            -------

            array

                Upper level populations

            """

            # Calculate T in Kelvin

            T_K = T_eV * EV_TO_K

            # Retrieve Partition Functions for all lines' elements

            # U0: Neutral (stage 0), U1: Ion (stage 1)

            # We only support I/II balance for now in this fast solver

            coeffs_0 = partition_coeffs[lines_el_idx, 0]

            coeffs_1 = partition_coeffs[lines_el_idx, 1]

            U0 = polynomial_partition_function_jax(T_K, coeffs_0)

            U1 = polynomial_partition_function_jax(T_K, coeffs_1)

            # Retrieve Ionization Potential (I -> II) for all lines' elements

            # We need the IP of the neutral species (stage 0) to balance I <-> II

            IP_I = ionization_potentials[lines_el_idx, 0]

            # Saha equation: n1 / n0 = 2/ne * (2pi mkT/h^2)^1.5 * (U1/U0) * exp(-IP/kT)

            # SAHA_CONST_CM3 is (2pi mk/h^2)^1.5 approx 6e21

            # We need the factor 2 for electron spin

            saha_factor = (SAHA_CONST_CM3 / n_e) * (T_eV**1.5)

            ratio_n1_n0 = 2.0 * saha_factor * (U1 / U0) * jnp.exp(-IP_I / T_eV)

            # Calculate population fractions

            # frac0 = n0 / (n0 + n1)

            # frac1 = n1 / (n0 + n1)

            frac0 = 1.0 / (1.0 + ratio_n1_n0)

            frac1 = ratio_n1_n0 / (1.0 + ratio_n1_n0)

            # Select fraction based on line's ionization stage

            # lines_z=0 -> use frac0, lines_z=1 -> use frac1

            pop_fraction = jnp.where(lines_z == 0, frac0, frac1)

            # Select appropriate partition function for Boltzmann

            # n_upper = n_species * (g / U) * exp(-E / T)

            U_val = jnp.where(lines_z == 0, U0, U1)

            # Total element density

            # N_total_element = element_conc * n_e

            element_conc = concentration_map[lines_el_idx]

            N_species_total = element_conc * n_e

            # Species density (n0 or n1)

            N_species = N_species_total * pop_fraction

            # Boltzmann level population

            n_upper = N_species * (lines_gk / U_val) * jnp.exp(-lines_ek / T_eV)

            return n_upper

        @staticmethod
        @jit
        def _compute_spectrum_snapshot(
            wl_grid: jnp.ndarray,
            T_eV: float,
            n_e: float,
            concentrations: jnp.ndarray,
            atomic_data: Tuple,
        ) -> jnp.ndarray:
            """

            Compute spectrum for a single time snapshot.



            Parameters

            ----------

            wl_grid : array

                Wavelength grid

            T_eV : float

                Temperature in eV

            n_e : float

                Electron density

            concentrations : array

                Element concentrations

            atomic_data : Tuple

                Atomic data arrays



            Returns

            -------

            array

                Spectral intensity

            """

            (
                l_wl,
                l_aki,
                l_ek,
                l_gk,
                l_ip,
                l_z,
                l_el_idx,
                partition_coeffs,
                ionization_potentials,
            ) = atomic_data

            # Solve populations

            n_upper = ManifoldGenerator._saha_eggert_solver(
                T_eV,
                n_e,
                concentrations,
                l_ip,
                l_z,
                l_el_idx,
                l_ek,
                l_gk,
                partition_coeffs,
                ionization_potentials,
            )

            # Line emissivity: epsilon = (hc / 4pi lambda) * A * n_upper

            H_J = 6.626e-34  # Planck constant in J·s

            epsilon = (H_J * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)) * l_aki * n_upper

            # Line broadening (Gaussian approximation for Phase 1)

            # Doppler width

            sigma_doppler = l_wl * 7.16e-7 * jnp.sqrt(T_eV * 11604)

            # Instrument function (simplified)

            sigma_inst = 0.05 / 2.355  # Assuming 0.05 nm FWHM

            sigma_total = jnp.sqrt(sigma_doppler**2 + sigma_inst**2)

            # Render to grid

            diff = wl_grid[:, None] - l_wl[None, :]

            profile = jnp.exp(-0.5 * (diff / sigma_total) ** 2) / (
                sigma_total * jnp.sqrt(2 * jnp.pi)
            )

            # Sum contributions

            intensity = jnp.sum(epsilon * profile, axis=1)

            return intensity

    @staticmethod
    @jit
    def _time_integrated_spectrum(
        wl_grid: jnp.ndarray,
        params: jnp.ndarray,
        atomic_data: Tuple,
        gate_width_s: float,
        time_steps: int,
    ) -> jnp.ndarray:
        """
        Compute time-integrated spectrum for cooling plasma.

        Parameters
        ----------
        wl_grid : array
            Wavelength grid
        params : array
            [T_max, ne_max, C_el1, C_el2, ...]
        atomic_data : Tuple
            Atomic data arrays
        gate_width_s : float
            Gate width in seconds
        time_steps : int
            Number of integration steps

        Returns
        -------
        array
            Time-integrated spectral intensity
        """
        T_max = params[0]
        ne_max = params[1]
        concs = params[2:]

        # Time grid
        times = jnp.linspace(0, gate_width_s, time_steps)
        dt = times[1] - times[0]

        # Cooling laws (power law decay)
        t0 = 1e-6
        T_trail = T_max * (1 + times / t0) ** (-0.5)
        ne_trail = ne_max * (1 + times / t0) ** (-1.0)

        # Integrate over time
        def step_fn(carry, inputs):
            T, ne = inputs
            intensity = jnp.where(
                T > 0.4,  # Only if T > 0.4 eV
                ManifoldGenerator._compute_spectrum_snapshot(wl_grid, T, ne, concs, atomic_data),
                jnp.zeros_like(wl_grid),
            )
            return carry + intensity * dt, None

        spectrum_accum = jnp.zeros_like(wl_grid)
        spectrum_accum, _ = jax.lax.scan(step_fn, spectrum_accum, (T_trail, ne_trail))

        return spectrum_accum

    def generate_manifold(self, progress_callback: Optional[callable] = None) -> None:
        """
        Generate the complete spectral manifold.

        Parameters
        ----------
        progress_callback : callable, optional
            Callback function(completed, total, percentage) for progress updates
        """
        logger.info("Starting manifold generation...")

        # Build parameter grid
        T_grid = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            self.config.temperature_steps,
        )
        ne_grid = np.geomspace(
            self.config.density_range[0], self.config.density_range[1], self.config.density_steps
        )

        # Build concentration grid (simplex for multi-element)
        # For now, simple grid for Ti-Al-V system
        params_list = []

        if len(self.config.elements) == 4:  # Ti-Al-V-Fe system
            al_range = np.linspace(0, 0.12, self.config.concentration_steps)
            v_range = np.linspace(0, 0.12, self.config.concentration_steps)

            for T in T_grid:
                for ne in ne_grid:
                    for al in al_range:
                        for v in v_range:
                            ti = 1.0 - (al + v)
                            if ti < 0:
                                continue
                            # [T, ne, Ti, Al, V, Fe]
                            params_list.append([T, ne, ti, al, v, 0.002])
        else:
            # Generic: vary first element concentration
            conc_range = np.linspace(0.5, 1.0, self.config.concentration_steps)
            for T in T_grid:
                for ne in ne_grid:
                    for c1 in conc_range:
                        # Simple case: one varying element
                        concs = [c1] + [0.0] * (len(self.config.elements) - 1)
                        params_list.append([T, ne] + concs)

        params_arr = np.array(params_list, dtype=np.float32)
        n_samples = len(params_arr)

        logger.info(f"Parameter grid: {n_samples} spectra to generate")
        logger.info(f"  Temperature: {len(T_grid)} points")
        logger.info(f"  Density: {len(ne_grid)} points")
        logger.info(
            f"  Concentrations: {len(params_list) // (len(T_grid) * len(ne_grid))} combinations"
        )

        # Create wavelength grid
        wl_grid = jnp.linspace(
            self.config.wavelength_range[0], self.config.wavelength_range[1], self.config.pixels
        )

        # Move atomic data to device
        atomic_data = tuple(jax.device_put(x) for x in self.atomic_data)

        # Vectorized function
        @jit
        def batch_spectrum(batch_params):
            return vmap(
                lambda p: ManifoldGenerator._time_integrated_spectrum(
                    wl_grid, p, atomic_data, self.config.gate_width_s, self.config.time_steps
                ),
                in_axes=0,
            )(batch_params)

        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for manifold generation. " "Install with: pip install h5py"
            )

        # Open HDF5 file
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        with h5py.File(output_path, "w") as f:
            # Create datasets
            dset_spec = f.create_dataset(
                "spectra",
                (n_samples, self.config.pixels),
                dtype="f4",
                compression="gzip",
                compression_opts=4,
            )
            dset_param = f.create_dataset(
                "params",
                (n_samples, len(self.config.elements) + 2),  # T, ne, concentrations
                dtype="f4",
            )
            f.create_dataset("wavelength", data=np.array(wl_grid), dtype="f4")

            # Store metadata
            f.attrs["elements"] = self.config.elements
            f.attrs["wavelength_range"] = self.config.wavelength_range
            f.attrs["temperature_range"] = self.config.temperature_range
            f.attrs["density_range"] = self.config.density_range

            # Process in batches
            for i in range(0, n_samples, self.config.batch_size):
                batch = params_arr[i : i + self.config.batch_size]
                batch_jax = jnp.array(batch)

                # Compute spectra
                spectra = batch_spectrum(batch_jax)

                # Convert to numpy
                spectra_np = np.array(spectra)

                # Save
                end_idx = min(i + self.config.batch_size, n_samples)
                dset_spec[i:end_idx] = spectra_np
                dset_param[i:end_idx] = batch

                # Progress update
                if progress_callback:
                    progress_callback(i + len(batch), n_samples, (i + len(batch)) / n_samples)
                elif i % (self.config.batch_size * 10) == 0:
                    logger.info(f"Generated {i}/{n_samples} ({i/n_samples:.1%})")

        total_time = time.time() - start_time
        logger.info(
            f"Manifold generation complete: {n_samples} spectra in {total_time:.2f}s "
            f"({n_samples/total_time:.0f} spectra/sec)"
        )
        logger.info(f"Output saved to: {output_path}")
