"""
Inversion algorithms for CF-LIBS.
"""

from cflibs.inversion.boltzmann import (
    LineObservation,
    BoltzmannFitResult,
    BoltzmannPlotFitter,
    FitMethod,
)

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

from cflibs.inversion.cdsb import (
    CDSBPlotter,
    CDSBResult,
    CDSBLineObservation,
    CDSBConvergenceStatus,
    LineOpticalDepth,
    create_cdsb_observation,
    from_transition,
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

# Nested sampling (requires dynesty)
try:
    from cflibs.inversion.bayesian import (
        NestedSampler,
        NestedSamplingResult,
    )

    _HAS_NESTED = True
except ImportError:
    _HAS_NESTED = False

__all__ = [
    "LineObservation",
    "BoltzmannFitResult",
    "BoltzmannPlotFitter",
    "FitMethod",
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
    # CD-SB plotting
    "CDSBPlotter",
    "CDSBResult",
    "CDSBLineObservation",
    "CDSBConvergenceStatus",
    "LineOpticalDepth",
    "create_cdsb_observation",
    "from_transition",
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

# Add Nested sampling exports if available
if _HAS_NESTED:
    __all__.extend([
        "NestedSampler",
        "NestedSamplingResult",
    ])

# Uncertainty propagation (requires uncertainties package)
try:
    from cflibs.inversion.uncertainty import (
        HAS_UNCERTAINTIES,
        create_boltzmann_uncertainties,
        temperature_from_slope,
        saha_factor_with_uncertainty,
        propagate_through_closure_standard,
        propagate_through_closure_matrix,
        extract_values_and_uncertainties,
    )

    _HAS_UNCERTAINTY = True
except ImportError:
    _HAS_UNCERTAINTY = False
    HAS_UNCERTAINTIES = False

# Add uncertainty exports if available
if _HAS_UNCERTAINTY:
    __all__.extend([
        "HAS_UNCERTAINTIES",
        "create_boltzmann_uncertainties",
        "temperature_from_slope",
        "saha_factor_with_uncertainty",
        "propagate_through_closure_standard",
        "propagate_through_closure_matrix",
        "extract_values_and_uncertainties",
    ])

# Monte Carlo UQ (always available - no external dependencies beyond numpy)
from cflibs.inversion.uncertainty import (
    MonteCarloUQ,
    MonteCarloResult,
    PerturbationType,
    AtomicDataUncertainty,
    run_monte_carlo_uq,
    HAS_JOBLIB,
)

__all__.extend([
    "MonteCarloUQ",
    "MonteCarloResult",
    "PerturbationType",
    "AtomicDataUncertainty",
    "run_monte_carlo_uq",
    "HAS_JOBLIB",
])

# Outlier detection (always available - no external dependencies)
from cflibs.inversion.outliers import (
    OutlierMethod,
    SAMResult,
    SpectralAngleMapper,
    sam_distance,
    detect_outlier_spectra,
    MADResult,
    MADOutlierDetector,
    mad_outliers_1d,
    mad_outliers_spectra,
    mad_clean_channels,
)

__all__.extend([
    "OutlierMethod",
    "SAMResult",
    "SpectralAngleMapper",
    "sam_distance",
    "detect_outlier_spectra",
    "MADResult",
    "MADOutlierDetector",
    "mad_outliers_1d",
    "mad_outliers_spectra",
    "mad_clean_channels",
])

# Matrix effect correction (always available - no external dependencies)
from cflibs.inversion.matrix_effects import (
    MatrixType,
    MatrixClassificationResult,
    CorrectionFactor,
    CorrectionFactorDB,
    MatrixCorrectionResult,
    MatrixEffectCorrector,
    InternalStandardResult,
    InternalStandardizer,
    combine_corrections,
)

__all__.extend([
    "MatrixType",
    "MatrixClassificationResult",
    "CorrectionFactor",
    "CorrectionFactorDB",
    "MatrixCorrectionResult",
    "MatrixEffectCorrector",
    "InternalStandardResult",
    "InternalStandardizer",
    "combine_corrections",
])
