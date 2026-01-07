"""
Radiation and spectral calculations.

This module provides:
- Line emissivity and opacity calculations
- Radiative transfer solvers (single zone, multi-zone)
- Continuum emission (Bremsstrahlung, recombination, etc.)
"""

from cflibs.radiation.emissivity import calculate_line_emissivity, calculate_spectrum_emissivity
from cflibs.radiation.profiles import gaussian_profile, apply_gaussian_broadening
from cflibs.radiation.spectrum_model import SpectrumModel
from cflibs.radiation.batch import (
    compute_spectrum_batch,
    compute_spectrum_grid,
    compute_spectrum_ensemble,
)

__all__ = [
    "calculate_line_emissivity",
    "calculate_spectrum_emissivity",
    "gaussian_profile",
    "apply_gaussian_broadening",
    "SpectrumModel",
    "compute_spectrum_batch",
    "compute_spectrum_grid",
    "compute_spectrum_ensemble",
]
