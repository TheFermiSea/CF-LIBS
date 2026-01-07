"""
Input/output utilities.

This module provides:
- Standardized file formats for spectra
- Plasma configuration I/O
- Atomic data snapshot I/O
- YAML/JSON config loading
"""

from cflibs.io.spectrum import load_spectrum, save_spectrum

__all__ = [
    "load_spectrum",
    "save_spectrum",
]
