# CF-LIBS Development Roadmap

This document outlines the development plan for implementing a production-grade Calibration-Free LIBS system. The roadmap has been restructured based on analysis of CF-LIBS requirements and best practices from the scientific literature.

## Strategic Approach

**"Physics First, Then Inversion"**

The core insight driving this roadmap: building inversion algorithms on top of incorrect physics produces scientifically meaningless results (GIGO - Garbage In, Garbage Out). Therefore:

1. **Phase 2a**: Fix fundamental physics (partition functions, Stark, Voigt)
2. **Phase 2b**: Implement classic CF-LIBS algorithm
3. **Phase 2c**: Add quality metrics and diagnostics
4. **Phase 2d**: Advanced forward fitting (manifold-based)

## Phase Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Minimal viable physics engine |
| Phase 2a | ✅ Complete | Physics foundation (partition functions, Voigt, Stark) |
| Phase 2b | ✅ Complete | Classic CF-LIBS implementation (Boltzmann, solver, closure) |
| Phase 2c | ✅ Complete | Quality metrics and diagnostics (95 tests) |
| Phase 2d | ✅ Complete | Advanced forward fitting (manifold + hybrid inversion) |
| Phase 3 | ✅ Complete | Bayesian methods and uncertainty |
| Phase 4 | 📋 Future | Ecosystem and integrations |

---

## Phase 1: Minimal Viable Physics Engine ✅

**Status**: Complete

Established the foundation for LIBS spectral modeling:

- [x] Saha-Boltzmann solver (up to 3 ionization stages)
- [x] Gaussian line profiles with Doppler broadening
- [x] Forward modeling pipeline (plasma → spectrum)
- [x] Atomic database (SQLite with NIST data)
- [x] Instrument convolution
- [x] CLI interface
- [x] Basic test coverage

**Limitations identified**:
- Doppler broadening uses ad-hoc approximation
- No Lorentzian/Voigt profiles
- No Stark broadening (cannot determine n_e)
- Manifold uses constant partition functions
- No inversion capability

---

## Phase 2a: Physics Foundation ✅

**Status**: Complete
**Priority**: Critical (P0)
**Tracking**: `bd show CF-LIBS-i0q`

Without correct physics, all downstream work (manifolds, inversion) produces invalid results.

### Completed Tasks

#### Database Schema Upgrade
- [x] Add `stark_w`, `stark_alpha`, `stark_shift` columns to `lines` table
- [x] Add `is_resonance` flag for ground-state transitions
- [x] Create `partition_functions` table with polynomial coefficients
- [x] Migration script for existing databases

#### Temperature-Dependent Partition Functions
- [x] Polynomial evaluator: `log(U) = Σ a_n (log T)^n` (Irwin form)
- [x] JAX-compatible implementation (`polynomial_partition_function_jax`)
- [x] NumPy fallback using direct summation
- [x] Caching for performance

#### Voigt Line Profile
- [x] Lorentzian profile implementation
- [x] Voigt via JAX-compatible Humlicek W4 approximation
- [x] Unit tests against scipy.special.wofz

#### Stark Broadening
- [x] Stark width scaling: `w(n_e, T) = w_ref × (n_e/10^16) × (T/T_ref)^(-α)`
- [x] Database lookup for Stark parameters
- [x] Fallback estimation (`estimate_stark_parameter_jax`)
- [x] Integration with line profile calculations

#### Fix Doppler Broadening
- [x] Replace ad-hoc formula with proper: `σ = λ × sqrt(2kT/mc²)`
- [x] Retrieve atomic mass from database (with fallback table)
- [x] Fixed in both SpectrumModel and ManifoldGenerator

---

## Phase 2b: Classic CF-LIBS Implementation ✅

**Status**: Complete
**Priority**: Critical (P0)
**Tracking**: `bd show CF-LIBS-59b`

Implemented the standard CF-LIBS algorithm used in literature (Ciucci, Tognoni, et al.).

### Completed Tasks

#### Boltzmann Plot Generation
- [x] Data structure for line observations (`LineObservation`, `BoltzmannPlotData`)
- [x] Weighted linear regression with outlier rejection
- [x] Sigma-clipping for robust fitting
- [x] RANSAC robust fitting for gross outliers
- [x] Huber M-estimation for moderate outliers
- [x] `FitMethod` enum for method selection
- [x] `BoltzmannPlotFitter` in `cflibs/inversion/boltzmann.py`
- [x] 20 tests in `tests/test_boltzmann.py`

#### Iterative CF-LIBS Solver
- [x] `IterativeCFLIBSSolver` in `cflibs/inversion/solver.py`
- [x] Saha correction to neutral plane
- [x] Converges in <20 iterations
- [x] Handles multiple elements simultaneously

#### Closure Equation
- [x] `ClosureEquation` in `cflibs/inversion/closure.py`
- [x] Standard mode: ΣC_measured = 1.0
- [x] Matrix mode: User specifies balance element
- [x] Oxide mode: For geological samples

#### Self-Absorption Correction
- [x] `SelfAbsorptionCorrector` in `cflibs/inversion/self_absorption.py`
- [x] Recursive correction with optical depth estimation
- [x] Line masking as fallback
- [x] Algorithm limitations documented (homogeneous plasma, LTE, Gaussian profile assumptions)
- [x] 46 tests in `tests/test_self_absorption.py`

#### Automatic Line Selection
- [x] `LineSelector` in `cflibs/inversion/line_selection.py`
- [x] Quality scoring: SNR × (1/σ_atomic) × isolation factor
- [x] Energy spread requirement
- [x] Resonance line exclusion option
- [x] Integration tests with Boltzmann fitting
- [x] 42 tests in `tests/test_line_selection.py`

---

## Phase 2c: Quality Metrics and Diagnostics ✅

**Status**: Complete (204 tests in quality/selection/absorption/outliers modules)
**Priority**: High (P1)
**Tracking**: `bd show CF-LIBS-4xu`

### Completed Tasks

#### Quality Metrics
- [x] `QualityMetrics` dataclass with R², χ², consistency scores
- [x] `QualityAssessor` in `cflibs/inversion/quality.py`
- [x] Boltzmann fit R² and residual analysis
- [x] Saha-Boltzmann consistency check
- [x] Inter-element temperature agreement
- [x] 31 tests in `tests/test_quality.py`

#### Line Selection Quality
- [x] Isolation scoring for line interference
- [x] Energy spread requirements
- [x] Scoring algorithm documented (SNR × atomic data quality × isolation)
- [x] 42 tests in `tests/test_line_selection.py`

#### Self-Absorption Diagnostics
- [x] Optical depth estimation
- [x] Correction factor tracking
- [x] Algorithm limitations documented
- [x] 46 tests in `tests/test_self_absorption.py`

#### Outlier Detection ✅
`CF-LIBS-n95`, `CF-LIBS-gy2` | Complete

- [x] **SAM (Spectral Angle Mapper)**: `SpectralAngleMapper` in `cflibs/inversion/outliers.py`
  - Intensity-invariant outlier detection (robust to shot-to-shot laser fluctuations)
  - `sam_distance()`, `detect_outlier_spectra()` convenience functions
  - Configurable angular threshold with automatic calculation option
  - 35 tests in `tests/test_outliers.py`
- [x] **MAD (Median Absolute Deviation)**: `MADOutlierDetector` in `cflibs/inversion/outliers.py`
  - Robust outlier detection with 50% breakdown point
  - Three modes: `1d` (scalar), `spectrum` (entire spectra), `channel` (per-wavelength)
  - `mad_outliers_1d()`, `mad_outliers_spectra()`, `mad_clean_channels()` convenience functions
  - Channel cleaning with interpolation or median replacement for cosmic ray removal
  - 30 tests in `tests/test_outliers.py`
- [x] `OutlierMethod` enum for method selection
- [x] `SAMResult`, `MADResult` dataclasses with statistics and indices

#### Error Propagation
`CF-LIBS-0pb` | P2 | Remaining

- [ ] Analytical errors from regression covariance
- [ ] Monte Carlo propagation (perturb inputs, re-run)
- [ ] Report: T±σ_T, n_e±σ_ne, C±σ_C

---

## Phase 2d: Advanced Forward Fitting ✅

**Status**: Complete
**Priority**: Normal (P2)
**Tracking**: `bd show CF-LIBS-cxm`

For difficult cases (heavy interference, high opacity) where classic CF-LIBS struggles.

### Completed Tasks

#### Repair Manifold Generator Physics ✅
`CF-LIBS-1mb` | Complete

- [x] Voigt profiles with Humlicek W4 Faddeeva approximation
- [x] Stark broadening with scaling law and database lookup
- [x] Proper Doppler width with mass dependence
- [x] Atomic mass lookup with standard element fallbacks
- [x] 14 physics tests in `tests/test_manifold_physics.py`
- [x] Configuration options: `use_voigt_profile`, `use_stark_broadening`
- [x] Physics version metadata in HDF5 manifolds

#### Round-Trip Testing ✅
`CF-LIBS-4wl` | Complete

- [x] `GoldenSpectrumGenerator`: Synthetic spectra with known ground truth
- [x] `NoiseModel`: Realistic noise (Poisson shot + Gaussian readout + laser fluctuations)
- [x] `RoundTripValidator`: Forward-inverse pipeline validation
- [x] 17 tests in `tests/test_round_trip.py`

#### Hybrid Inversion Strategy ✅
`CF-LIBS-o7b` | Complete

- [x] `HybridInverter`: Manifold coarse search + JAX L-BFGS optimization
- [x] `SpectralFitter`: Standalone JAX-based spectral fitting
- [x] Parameter packing with log-transform (positivity) and softmax (simplex)
- [x] 14 tests in `tests/test_hybrid_inversion.py`

---

## Phase 3: Bayesian Methods and Uncertainty ✅

**Status**: Complete
**Priority**: Normal (P2)
**Tracking**: `bd show CF-LIBS-cjq`

Full uncertainty quantification via Bayesian inference. Critical for scientific credibility.

### Completed Tasks

#### Bayesian Forward Model ✅
`CF-LIBS-cxp` | Complete

- [x] `BayesianForwardModel` class with full physics (Saha-Boltzmann, Voigt, Stark)
- [x] Likelihood function: `log P(spectrum | T, n_e, C)` via `log_likelihood()`
- [x] Noise model: Poisson (shot) + Gaussian (readout) + dark current
- [x] JAX-compatible for autodiff (integrates with NumPyro)
- [x] 57 tests in `tests/test_bayesian.py`

#### Prior Specification ✅
`CF-LIBS-zbs` | Complete

- [x] `PriorConfig` dataclass with physical bounds
- [x] Temperature prior: uniform on 0.5-3 eV (typical LIBS range)
- [x] Electron density prior: log-uniform (Jeffreys) on 10^15-10^19 cm^-3
- [x] Concentration prior: Dirichlet (enforces simplex constraint)
- [x] `create_temperature_prior()`, `create_density_prior()`, `create_concentration_prior()` functions

#### MCMC Sampling ✅
`CF-LIBS-0oq` | Complete

- [x] `MCMCSampler` class wrapping NumPyro NUTS
- [x] `MCMCResult` dataclass with samples, summary statistics, and credible intervals
- [x] Convergence diagnostics: R-hat (Gelman-Rubin) and ESS via ArviZ
- [x] `summary_table()` for publication-ready output
- [x] ArviZ integration (`plot_trace()`, `plot_posterior()`)
- [x] `posterior_predictive_check()` for model validation (chi-squared, Bayesian p-value)
- [x] **Fixed**: Voigt profile gradients now stable via Weideman rational approximation (CF-LIBS-452)
- [x] User Guide with workflow examples in module docstring
- [x] Prior Selection Guide (temperature, density, concentration)
- [x] Prior Sensitivity Analysis documentation
- [x] Convergence Diagnostics guide

### Completed Tasks

#### Nested Sampling (Model Comparison) ✅
`CF-LIBS-nfo` | Complete

- [x] dynesty integration via `NestedSampler` class
- [x] Evidence calculation: `log_evidence`, `log_evidence_err` in `NestedSamplingResult`
- [x] Bayes factor model comparison: `NestedSamplingResult.compare_models()`
- [x] Unit cube prior transform with Dirichlet stick-breaking for concentrations
- [x] 8 tests in `tests/test_bayesian.py::TestNestedSampling`

#### Uncertainty Reporting ✅
`CF-LIBS-k7g` | Complete

- [x] MCMCResult with credible intervals: T±σ, n_e±σ, C±σ
- [x] Correlation analysis: `correlation_matrix()`, `correlation_table()`
- [x] ArviZ visualization: `plot_corner()`, `plot_forest()`

---

## Phase 4: Ecosystem and Integrations 📋

**Status**: Future
**Priority**: Low (P3)

Production deployment and ecosystem support.

### Planned Features

- Jupyter-friendly visualization
- Export tools for common formats
- Hardware driver integration (VISA, PyMoDAQ)
- Example notebooks
- Performance optimization for batch processing

---

## Issue Tracking

This project uses `bd` (beads) for issue tracking. View all issues:

```bash
bd list                    # All issues
bd ready                   # Ready to work on
bd blocked                 # Blocked issues
bd show CF-LIBS-xxx        # Issue details
bd dep tree CF-LIBS-y7o    # Dependency tree
```

## Literature Review

A systematic review of high-performance LIBS algorithms (2018-2025) was conducted to guide implementation priorities. Key findings are documented in `docs/literature/high_performance_libs_algorithms.md`.

**Tracking**: `bd show CF-LIBS-4eh` (Implementation Priorities), `bd show CF-LIBS-h1l` (Research Gaps)

### Implemented from Literature
- RANSAC/Huber robust fitting for outlier rejection (CF-LIBS-4eh.1)
- Self-absorption validation framework (CF-LIBS-4eh.2)
- Line quality scoring integration (CF-LIBS-4eh.3)
- Posterior predictive checks for Bayesian UQ (CF-LIBS-4eh.4)

### Future Directions (P3/P4)
- Multi-element joint optimization
- Transfer learning for instrument calibration
- Real-time analysis pipelines
- Physics-informed neural networks

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

Priority areas for contribution:
1. **Error Propagation** - Monte Carlo uncertainty (CF-LIBS-0pb)
2. **Benchmark Database** - Open LIBS spectral reference data (CF-LIBS-h1l.1)
3. **Matrix Effects** - Correction methods for complex samples (CF-LIBS-h1l.2)
4. **Documentation** - Examples, tutorials, Jupyter notebooks

## References

- Ciucci, A., et al. "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy." Applied Spectroscopy 53.8 (1999): 960-964.
- Tognoni, E., et al. "Calibration-free laser-induced breakdown spectroscopy: state of the art." Spectrochimica Acta Part B 65.1 (2010): 1-14.
- Aragón, C., and J. A. Aguilera. "Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods." Spectrochimica Acta Part B 63.9 (2008): 893-916.
