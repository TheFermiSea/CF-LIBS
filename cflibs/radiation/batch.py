"""
Batch processing utilities for multiple spectra.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from cflibs.radiation.spectrum_model import SpectrumModel
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.batch")


def compute_spectrum_batch(
    models: List[SpectrumModel], n_workers: Optional[int] = None, use_processes: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute multiple spectra in parallel.

    Parameters
    ----------
    models : List[SpectrumModel]
        List of spectrum models to compute
    n_workers : int, optional
        Number of worker threads/processes. If None, uses CPU count.
    use_processes : bool
        If True, use processes instead of threads (for CPU-bound work)

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (wavelength, intensity) tuples
    """
    if not models:
        return []

    if n_workers is None:
        import os

        n_workers = os.cpu_count() or 1

    logger.info(f"Computing {len(models)} spectra with {n_workers} workers")

    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results = []
    with executor_class(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(model.compute_spectrum): i for i, model in enumerate(models)}

        # Collect results in order
        completed = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                completed[idx] = result
            except Exception as e:
                logger.error(f"Error computing spectrum {idx}: {e}")
                completed[idx] = (None, None)

        # Return in original order
        results = [completed[i] for i in range(len(models))]

    logger.info(f"Completed batch computation of {len(results)} spectra")
    return results



def _apply_params_to_plasma(plasma, params: dict) -> None:
    """
    Apply parameter values to a SingleZoneLTEPlasma object in-place.
    
    Recognizes the keys "T_e_eV" and "n_e" to set plasma.T_e_eV and plasma.n_e respectively; any other keys that match entries in plasma.species will update those species values.
    
    Parameters:
        plasma: SingleZoneLTEPlasma
            Plasma state to modify in-place.
        params: dict
            Mapping of parameter names to values. Expected keys include "T_e_eV", "n_e", or species names present in plasma.species.
    """
    if "T_e_eV" in params:
        plasma.T_e_eV = params["T_e_eV"]
    if "n_e" in params:
        plasma.n_e = params["n_e"]

    for key, value in params.items():
        if key not in ("T_e_eV", "n_e") and key in plasma.species:
            plasma.species[key] = value


def compute_spectrum_grid(
    base_model: SpectrumModel,
    parameter_grid: Dict[str, List[float]],
    n_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute spectra for every combination of parameter values in `parameter_grid` using `base_model` as a template.
    
    Parameters:
        base_model (SpectrumModel): Template model whose plasma will be copied and updated for each parameter combination.
        parameter_grid (Dict[str, List[float]]): Mapping from parameter names to lists of values to sweep. Supported keys include 'T_e_eV', 'n_e', and species names present in the model's plasma.
        n_workers (Optional[int]): Number of worker threads/processes to use for parallel computation (defaults to automatic selection).
    
    Returns:
        parameters (List[Dict[str, float]]): List of parameter dictionaries corresponding to each computed spectrum, in the same order as the returned spectra.
        spectra (List[Tuple[np.ndarray, np.ndarray]]): List of (wavelength, intensity) tuples for each parameter combination, preserving the input order.
    """
    from itertools import product
    from copy import deepcopy

    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())

    all_combinations = list(product(*param_values))

    logger.info(f"Computing grid with {len(all_combinations)} parameter combinations")

    # Create models for each combination
    models = []
    parameter_list = []

    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        parameter_list.append(params)

        # Create new model with modified plasma
        model = deepcopy(base_model)
        plasma = model.plasma

        _apply_params_to_plasma(plasma, params)

        models.append(model)

    # Compute all spectra
    spectra = compute_spectrum_batch(models, n_workers=n_workers)

    return parameter_list, spectra


def compute_spectrum_ensemble(
    base_model: SpectrumModel,
    n_samples: int,
    parameter_distributions: Dict[str, callable],
    n_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute spectra for an ensemble of parameter sets sampled from the provided distributions.
    
    Parameters:
        base_model (SpectrumModel): Template model whose deep copies are used for each sample.
        n_samples (int): Number of parameter samples to generate.
        parameter_distributions (Dict[str, callable]): Mapping of parameter names to zero-argument callables that return a float for each sample.
        n_workers (Optional[int]): Number of worker threads/processes to use for spectrum computation (optional).
    
    Returns:
        parameters (List[Dict[str, float]]): List of sampled parameter dictionaries in the same order spectra were computed.
        spectra (List[Tuple[np.ndarray, np.ndarray]]): List of (wavelength, intensity) tuples corresponding to each sampled parameter set.
    """
    from copy import deepcopy

    logger.info(f"Generating ensemble with {n_samples} samples")

    # Generate parameter samples
    parameter_list = []
    models = []

    for i in range(n_samples):
        params = {}
        for name, dist_func in parameter_distributions.items():
            params[name] = dist_func()
        parameter_list.append(params)

        # Create model with sampled parameters
        model = deepcopy(base_model)
        plasma = model.plasma

        _apply_params_to_plasma(plasma, params)

        models.append(model)

    # Compute all spectra
    spectra = compute_spectrum_batch(models, n_workers=n_workers)

    return parameter_list, spectra