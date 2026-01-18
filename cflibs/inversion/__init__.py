"""
Inversion algorithms for CF-LIBS.

This module provides the core inversion algorithms for calibration-free
LIBS analysis, including Boltzmann plotting, closure equations, and
optional Bayesian inference and Monte Carlo uncertainty quantification.
"""

# --- Core (always available) ---
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
from cflibs.inversion.uncertainty import (
    MonteCarloUQ,
    MonteCarloResult,
    PerturbationType,
    AtomicDataUncertainty,
    run_monte_carlo_uq,
    HAS_JOBLIB,
)
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

# --- Optional availability flags ---
HAS_HYBRID = False
HAS_BAYESIAN = False
HAS_NESTED = False
HAS_UNCERTAINTIES = False

# --- Optional: Hybrid inversion (requires JAX) ---
try:
    from cflibs.inversion.hybrid import HybridInverter, HybridInversionResult, SpectralFitter
    HAS_HYBRID = True
except ImportError:
    pass

# --- Optional: Bayesian inference (requires JAX + NumPyro) ---
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
    HAS_BAYESIAN = True
except ImportError:
    pass

# --- Optional: Nested sampling (requires dynesty) ---
try:
    from cflibs.inversion.bayesian import NestedSampler, NestedSamplingResult
    HAS_NESTED = True
except ImportError:
    pass

# --- Optional: Uncertainty propagation (requires uncertainties package) ---
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
except ImportError:
    pass

# --- Public API ---
__all__ = [
    # Boltzmann plotting
    "LineObservation",
    "BoltzmannFitResult",
    "BoltzmannPlotFitter",
    "FitMethod",
    # Closure
    "ClosureEquation",
    "ClosureResult",
    # Solver
    "IterativeCFLIBSSolver",
    "CFLIBSResult",
    # Quality metrics
    "QualityMetrics",
    "QualityAssessor",
    "compute_reconstruction_chi_squared",
    # Line selection
    "LineScore",
    "LineSelectionResult",
    "LineSelector",
    "identify_resonance_lines",
    # Self-absorption
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
    # Monte Carlo UQ
    "MonteCarloUQ",
    "MonteCarloResult",
    "PerturbationType",
    "AtomicDataUncertainty",
    "run_monte_carlo_uq",
    "HAS_JOBLIB",
    # Outlier detection
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
    # Matrix effects
    "MatrixType",
    "MatrixClassificationResult",
    "CorrectionFactor",
    "CorrectionFactorDB",
    "MatrixCorrectionResult",
    "MatrixEffectCorrector",
    "InternalStandardResult",
    "InternalStandardizer",
    "combine_corrections",
    # Availability flags
    "HAS_HYBRID",
    "HAS_BAYESIAN",
    "HAS_NESTED",
    "HAS_UNCERTAINTIES",
]

# Extend __all__ with optional exports
if HAS_HYBRID:
    __all__.extend(["HybridInverter", "HybridInversionResult", "SpectralFitter"])

if HAS_BAYESIAN:
    __all__.extend([
        "BayesianForwardModel", "AtomicDataArrays", "NoiseParameters", "PriorConfig",
        "MCMCResult", "MCMCSampler", "ConvergenceStatus", "log_likelihood",
        "bayesian_model", "run_mcmc", "create_temperature_prior",
        "create_density_prior", "create_concentration_prior",
    ])

if HAS_NESTED:
    __all__.extend(["NestedSampler", "NestedSamplingResult"])

if HAS_UNCERTAINTIES:
    __all__.extend([
        "create_boltzmann_uncertainties", "temperature_from_slope",
        "saha_factor_with_uncertainty", "propagate_through_closure_standard",
        "propagate_through_closure_matrix", "extract_values_and_uncertainties",
    ])
