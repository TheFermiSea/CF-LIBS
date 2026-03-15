"""
Benchmark harness for CF-LIBS algorithm experiments.

This package provides infrastructure for systematically evaluating CF-LIBS
inversion pipelines: synthetic test corpus generation, compositional accuracy
metrics (Aitchison distance, CLR/ILR transforms), and an experiment runner
that collects timing and accuracy data.

Main Classes
------------
BenchmarkCorpus
    Generates synthetic LIBS spectra with known ground truth.
BenchmarkHarness
    Registers and runs named pipeline functions on a corpus.
BenchmarkReport
    Aggregated results with per-pipeline, per-spectrum accuracy and timing.

Key Functions
-------------
aitchison_distance
    Compositional distance in the simplex (scale-invariant).
clr_transform
    Centered log-ratio transform.
ilr_transform
    Isometric log-ratio transform (Helmert sub-composition).
rmse_composition
    Standard RMSE between two composition vectors.
per_element_error
    Per-element absolute and relative errors.
"""

from cflibs.benchmarks.metrics import (
    aitchison_distance,
    clr_transform,
    ilr_transform,
    rmse_composition,
    per_element_error,
)
from cflibs.benchmarks.corpus import BenchmarkCorpus, BenchmarkSpectrum
from cflibs.benchmarks.harness import (
    AccuracyTier,
    BenchmarkHarness,
    BenchmarkReport,
    PipelineResult,
    SpectrumResult,
)

__all__ = [
    # Metrics
    "aitchison_distance",
    "clr_transform",
    "ilr_transform",
    "rmse_composition",
    "per_element_error",
    # Corpus
    "BenchmarkCorpus",
    "BenchmarkSpectrum",
    # Harness
    "AccuracyTier",
    "BenchmarkHarness",
    "BenchmarkReport",
    "PipelineResult",
    "SpectrumResult",
]
