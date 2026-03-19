"""
Open LIBS Spectral Benchmark Database.

This module provides standardized benchmarks for comparing LIBS algorithm
performance, addressing the research gap of non-standardized cross-study
comparisons in the LIBS community.

Key Features
------------
- Certified reference material (CRM) spectra representation
- Standardized train/test splits for reproducible evaluation
- Community-accepted evaluation metrics (RMSEP, MAE, bias)
- Support for multiple instrumental conditions
- Synthetic benchmark generation for validation

Main Classes
------------
BenchmarkSpectrum
    Single spectrum with ground truth composition and metadata
BenchmarkDataset
    Collection of spectra with train/test splits
BenchmarkMetrics
    Evaluation metrics for algorithm comparison
SyntheticBenchmarkGenerator
    Generate synthetic benchmarks with known ground truth

Example
-------
>>> from cflibs.benchmark import BenchmarkDataset, BenchmarkMetrics
>>>
>>> # Load a benchmark dataset
>>> dataset = BenchmarkDataset.from_json("nist_steel_crm.json")
>>>
>>> # Get train/test splits
>>> train, test = dataset.get_split("default")
>>>
>>> # Evaluate predictions
>>> metrics = BenchmarkMetrics()
>>> results = metrics.evaluate(predictions, test.true_compositions)
>>> print(results.summary())

References
----------
- Hahn & Omenetto (2010) "Applied Spectroscopy Focal Point Review"
- Tognoni et al. (2010) "CF-LIBS: State of the art"
- NIST Standard Reference Materials program
"""

from importlib import import_module

_LAZY_SYNTHETIC_EXPORTS = {"CalibrationOptions", "compute_binary_metrics", "run_synthetic_benchmark"}
_LAZY_UNIFIED_EXPORTS = {
    "UnifiedBenchmarkContext",
    "UnifiedBenchmarkRunner",
    "build_composition_workflow_registry",
    "build_id_workflow_registry",
    "load_default_datasets",
}

__all__ = [
    # Core data structures
    "BenchmarkSpectrum",
    "InstrumentalConditions",
    "SampleMetadata",
    "BenchmarkDataset",
    "DataSplit",
    # Metrics
    "BenchmarkMetrics",
    "EvaluationResult",
    "ElementMetrics",
    "MetricType",
    # Synthetic generation
    "SyntheticBenchmarkGenerator",
    "CompositionRange",
    "ConditionVariation",
    "CorpusRecipe",
    "PerturbationAxes",
    "build_synthetic_id_corpus",
    "default_axes",
    "default_recipes",
    "UnifiedBenchmarkContext",
    "UnifiedBenchmarkRunner",
    "build_composition_workflow_registry",
    "build_id_workflow_registry",
    "load_default_datasets",
    "CalibrationOptions",
    "compute_binary_metrics",
    "run_synthetic_benchmark",
    # I/O
    "load_benchmark",
    "save_benchmark",
    "BenchmarkFormat",
]

_MODULE_EXPORTS = {
    "BenchmarkSpectrum": ("cflibs.benchmark.dataset", "BenchmarkSpectrum"),
    "InstrumentalConditions": ("cflibs.benchmark.dataset", "InstrumentalConditions"),
    "SampleMetadata": ("cflibs.benchmark.dataset", "SampleMetadata"),
    "BenchmarkDataset": ("cflibs.benchmark.dataset", "BenchmarkDataset"),
    "DataSplit": ("cflibs.benchmark.dataset", "DataSplit"),
    "BenchmarkMetrics": ("cflibs.benchmark.metrics", "BenchmarkMetrics"),
    "EvaluationResult": ("cflibs.benchmark.metrics", "EvaluationResult"),
    "ElementMetrics": ("cflibs.benchmark.metrics", "ElementMetrics"),
    "MetricType": ("cflibs.benchmark.metrics", "MetricType"),
    "SyntheticBenchmarkGenerator": ("cflibs.benchmark.synthetic", "SyntheticBenchmarkGenerator"),
    "CompositionRange": ("cflibs.benchmark.synthetic", "CompositionRange"),
    "ConditionVariation": ("cflibs.benchmark.synthetic", "ConditionVariation"),
    "CorpusRecipe": ("cflibs.benchmark.synthetic_corpus", "CorpusRecipe"),
    "PerturbationAxes": ("cflibs.benchmark.synthetic_corpus", "PerturbationAxes"),
    "build_synthetic_id_corpus": ("cflibs.benchmark.synthetic_corpus", "build_synthetic_id_corpus"),
    "default_axes": ("cflibs.benchmark.synthetic_corpus", "default_axes"),
    "default_recipes": ("cflibs.benchmark.synthetic_corpus", "default_recipes"),
    "load_benchmark": ("cflibs.benchmark.loaders", "load_benchmark"),
    "save_benchmark": ("cflibs.benchmark.loaders", "save_benchmark"),
    "BenchmarkFormat": ("cflibs.benchmark.loaders", "BenchmarkFormat"),
}


def __getattr__(name: str):
    """Lazy-load benchmark exports so lightweight submodules remain importable."""
    if name in _MODULE_EXPORTS:
        module_name, attr_name = _MODULE_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _LAZY_SYNTHETIC_EXPORTS:
        module = import_module("cflibs.benchmark.synthetic_eval")

        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_UNIFIED_EXPORTS:
        module = import_module("cflibs.benchmark.unified")

        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
