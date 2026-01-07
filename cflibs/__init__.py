"""
CF-LIBS: Computational Framework for Laser-Induced Breakdown Spectroscopy

A production-grade Python library for forward modeling, inversion, and analysis
of LIBS plasmas with emphasis on rigorous physics, high-performance numerics,
and reproducible workflows.
"""

__version__ = "0.1.0"
__author__ = "TheFermiSea"

# Core imports for convenience
from cflibs.core import constants
from cflibs.core import units

__all__ = [
    "constants",
    "units",
]
