"""
Inversion algorithms for CF-LIBS.
"""

from cflibs.inversion.boltzmann import LineObservation, BoltzmannFitResult, BoltzmannPlotFitter

from cflibs.inversion.closure import ClosureEquation, ClosureResult

from cflibs.inversion.solver import IterativeCFLIBSSolver, CFLIBSResult

from cflibs.inversion.quality import (
    QualityMetrics,
    QualityAssessor,
    compute_reconstruction_chi_squared,
)

from cflibs.inversion.line_selection import (
    LineScore,
    LineSelectionResult,
    LineSelector,
    identify_resonance_lines,
)

from cflibs.inversion.self_absorption import (
    AbsorptionCorrectionResult,
    SelfAbsorptionResult,
    SelfAbsorptionCorrector,
    estimate_optical_depth_from_intensity_ratio,
)

__all__ = [
    "LineObservation",
    "BoltzmannFitResult",
    "BoltzmannPlotFitter",
    "ClosureEquation",
    "ClosureResult",
    "IterativeCFLIBSSolver",
    "CFLIBSResult",
    "QualityMetrics",
    "QualityAssessor",
    "compute_reconstruction_chi_squared",
    "LineScore",
    "LineSelectionResult",
    "LineSelector",
    "identify_resonance_lines",
    "AbsorptionCorrectionResult",
    "SelfAbsorptionResult",
    "SelfAbsorptionCorrector",
    "estimate_optical_depth_from_intensity_ratio",
]
