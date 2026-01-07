"""
Manifold generation for high-throughput CF-LIBS analysis.

This module provides tools for generating pre-computed spectral manifolds
that enable fast inference without solving physics equations at runtime.

The manifold is a high-dimensional lookup table of synthetic spectra
generated from first principles using JAX for GPU acceleration.
"""

from cflibs.manifold.generator import ManifoldGenerator
from cflibs.manifold.config import ManifoldConfig
from cflibs.manifold.loader import ManifoldLoader

__all__ = [
    "ManifoldGenerator",
    "ManifoldConfig",
    "ManifoldLoader",
]
