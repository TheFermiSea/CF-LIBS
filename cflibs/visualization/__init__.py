"""
Interactive visualization widgets for CF-LIBS analysis.

This module provides Jupyter-compatible interactive widgets for visualizing
CF-LIBS spectra, Boltzmann plots, Bayesian posteriors, and quality metrics.

Optional Dependencies
---------------------
The widgets require ipywidgets and plotly:

    pip install cflibs[widgets]

Usage
-----
>>> from cflibs.visualization import SpectrumViewer, BoltzmannPlotWidget
>>> viewer = SpectrumViewer()
>>> viewer.add_spectrum(wavelength, intensity, label="Sample 1")
>>> viewer.show()
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from importlib.util import find_spec

# Check for optional dependencies
HAS_WIDGETS = find_spec("ipywidgets") is not None and find_spec("plotly") is not None

if HAS_WIDGETS:
    from cflibs.visualization.widgets import (
        SpectrumViewer,
        BoltzmannPlotWidget,
        PosteriorViewer,
        QualityDashboard,
    )

    __all__ = [
        "HAS_WIDGETS",
        "SpectrumViewer",
        "BoltzmannPlotWidget",
        "PosteriorViewer",
        "QualityDashboard",
    ]
else:
    __all__ = ["HAS_WIDGETS"]
