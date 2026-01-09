"""
Validation utilities for CF-LIBS.

This module provides tools for validating the CF-LIBS pipeline through
round-trip testing: generate synthetic spectra with known parameters,
add noise, run inversion, and verify parameter recovery.
"""

from cflibs.validation.round_trip import (
    GoldenSpectrumGenerator,
    NoiseModel,
    RoundTripValidator,
    RoundTripResult,
    GoldenSpectrum,
)

__all__ = [
    "GoldenSpectrumGenerator",
    "NoiseModel",
    "RoundTripValidator",
    "RoundTripResult",
    "GoldenSpectrum",
]
