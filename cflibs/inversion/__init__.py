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

# Hybrid inversion (requires JAX)
try:
    from cflibs.inversion.hybrid import HybridInverter, HybridInversionResult, SpectralFitter

    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False

# Bayesian inference (requires JAX + NumPyro)
try:
    from cflibs.inversion.bayesian import (
        BayesianForwardModel,
        AtomicDataArrays,
        NoiseParameters,
        PriorConfig,
        MCMCResult,
        MCMCSampler,
        ConvergenceStatus,
        log_likelihood,
        bayesian_model,
        run_mcmc,
        create_temperature_prior,
        create_density_prior,
        create_concentration_prior,
    )

    _HAS_BAYESIAN = True
except ImportError:
    _HAS_BAYESIAN = False

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

# Add hybrid inversion exports if available
if _HAS_HYBRID:
    __all__.extend(["HybridInverter", "HybridInversionResult", "SpectralFitter"])

# Add Bayesian exports if available
if _HAS_BAYESIAN:
    __all__.extend([
        "BayesianForwardModel",
        "AtomicDataArrays",
        "NoiseParameters",
        "PriorConfig",
        "MCMCResult",
        "MCMCSampler",
        "ConvergenceStatus",
        "log_likelihood",
        "bayesian_model",
        "run_mcmc",
        "create_temperature_prior",
        "create_density_prior",
        "create_concentration_prior",
    ])
