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
| Phase 2a | 🔄 In Progress | Physics foundation |
| Phase 2b | 📋 Planned | Classic CF-LIBS implementation |
| Phase 2c | 📋 Planned | Quality metrics and diagnostics |
| Phase 2d | 📋 Planned | Advanced forward fitting |
| Phase 3 | 📋 Future | Bayesian methods and uncertainty |
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

## Phase 2a: Physics Foundation 🔄

**Status**: In Progress
**Priority**: Critical (P0)
**Tracking**: `bd show CF-LIBS-i0q`

Without correct physics, all downstream work (manifolds, inversion) produces invalid results.

### Tasks

#### Database Schema Upgrade
`CF-LIBS-j1v` | P0 | Blocks all other Phase 2a work

- [ ] Add `stark_w`, `stark_alpha`, `stark_shift` columns to `lines` table
- [ ] Add `is_resonance` flag for ground-state transitions
- [ ] Create `partition_functions` table with polynomial coefficients
- [ ] Migration script for existing databases

#### Temperature-Dependent Partition Functions
`CF-LIBS-irv` | P0 | Blocked by database schema

- [ ] Polynomial evaluator: `log(U) = Σ a_n (log T)^n` (Irwin form)
- [ ] JAX-compatible implementation for manifold generator
- [ ] NumPy fallback using direct summation
- [ ] Caching for performance

#### Voigt Line Profile
`CF-LIBS-013` | P0

- [ ] Lorentzian profile implementation
- [ ] Voigt via JAX-compatible rational approximation (Humlicek)
- [ ] Unit tests against scipy.special.wofz

#### Stark Broadening
`CF-LIBS-k6h` | P0

- [ ] Stark width scaling: `w(n_e, T) = w_ref × (n_e/10^16) × (T/T_ref)^(-0.5)`
- [ ] Database lookup for Stark parameters
- [ ] Fallback approximation for missing data
- [ ] Integration with line profile calculations

#### Fix Doppler Broadening
`CF-LIBS-8o7` | P1 | Bug fix

- [ ] Replace ad-hoc formula with proper: `Δλ_D = λ_0 × sqrt(2kT/mc²)`
- [ ] Retrieve atomic mass from database
- [ ] Fix in both SpectrumModel and ManifoldGenerator

---

## Phase 2b: Classic CF-LIBS Implementation 📋

**Status**: Planned
**Priority**: Critical (P0)
**Tracking**: `bd show CF-LIBS-59b`
**Blocked by**: Phase 2a

Implement the standard CF-LIBS algorithm used in literature (Ciucci, Tognoni, et al.).

### Tasks

#### Boltzmann Plot Generation
`CF-LIBS-6vf` | P0

- [ ] Data structure for line observations
- [ ] Weighted linear regression
- [ ] Outlier rejection (RANSAC/sigma-clipping)
- [ ] Visualization output

#### Iterative CF-LIBS Solver
`CF-LIBS-970` | P0 | Core algorithm

Algorithm:
1. Initialize T₀, n_e₀
2. Calculate U(T) for all species
3. Correct ionic lines to neutral Boltzmann plane
4. Weighted linear regression for temperature
5. Extract species intercepts → concentrations
6. Apply closure equation
7. Update n_e via Saha
8. Iterate until convergence

Acceptance criteria:
- [ ] Converges in <20 iterations
- [ ] Matches literature values for benchmark spectra
- [ ] Handles multiple elements simultaneously

#### Closure Equation
`CF-LIBS-zau` | P1

- [ ] Standard mode: ΣC_measured = 1.0
- [ ] Matrix mode: User specifies balance element
- [ ] Oxide mode: For geological samples (cations → oxides)

#### Self-Absorption Correction
`CF-LIBS-n0a` | P1

- [ ] Recursive correction: `I_corr = I_meas / [(1 - exp(-τ)) / τ]`
- [ ] Optical depth estimation
- [ ] Line masking as fallback

#### Automatic Line Selection
`CF-LIBS-usv` | P1

- [ ] Quality scoring: SNR × (1/unc_atomic) × IsolationFactor
- [ ] Exclude resonance lines by default
- [ ] Energy spread requirement (>2 eV)

---

## Phase 2c: Quality Metrics and Diagnostics 📋

**Status**: Planned
**Priority**: High (P1)
**Tracking**: `bd show CF-LIBS-4xu`
**Blocked by**: Phase 2b iterative solver

### Tasks

#### Quality Metrics
`CF-LIBS-4jd` | P1

- [ ] R² of Boltzmann fit
- [ ] Saha-Boltzmann consistency
- [ ] Reconstruction residual χ²
- [ ] Inter-element temperature agreement

#### Error Propagation
`CF-LIBS-0pb` | P2

- [ ] Analytical errors from regression covariance
- [ ] Monte Carlo propagation (perturb inputs, re-run)
- [ ] Report: T±σ_T, n_e±σ_ne, C±σ_C

---

## Phase 2d: Advanced Forward Fitting 📋

**Status**: Planned
**Priority**: Normal (P2)
**Tracking**: `bd show CF-LIBS-cxm`

For difficult cases (heavy interference, high opacity) where classic CF-LIBS struggles.

### Tasks

#### Repair Manifold Generator Physics
`CF-LIBS-1mb` | P2 | Blocked by Voigt, Stark

- [ ] Replace hard-coded partition functions with polynomial evaluator
- [ ] Add Stark broadening to profiles
- [ ] Fix Doppler width calculation
- [ ] Validate against SpectrumModel output

#### Hybrid Inversion Strategy
`CF-LIBS-o7b` | P2

- [ ] Coarse search: Manifold nearest-neighbor (cosine similarity)
- [ ] Fine-tune: JAX autodiff + BFGS/L-M optimizer
- [ ] Combine global search + local precision

#### Round-Trip Testing
`CF-LIBS-4wl` | P2

- [ ] Generate "Golden Spectra" with known parameters
- [ ] Add realistic noise (Poisson + Gaussian)
- [ ] Verify parameter recovery: T±5%, n_e±20%

---

## Phase 3: Bayesian Methods and Uncertainty 📋

**Status**: Future
**Priority**: Low (P3)

Full uncertainty quantification via Bayesian inference.

### Planned Features

- MCMC sampling (emcee, PyMC)
- Nested sampling for model comparison
- Prior specification for plasma parameters
- Posterior diagnostics and credible intervals
- Propagation of atomic data uncertainties

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

Priority areas for contribution:
1. **Phase 2a physics** - Voigt profiles, Stark broadening
2. **Phase 2b classic solver** - Boltzmann plots, closure equation
3. **Testing** - Round-trip tests, benchmark spectra
4. **Documentation** - Examples, tutorials

## References

- Ciucci, A., et al. "New procedure for quantitative elemental analysis by laser-induced plasma spectroscopy." Applied Spectroscopy 53.8 (1999): 960-964.
- Tognoni, E., et al. "Calibration-free laser-induced breakdown spectroscopy: state of the art." Spectrochimica Acta Part B 65.1 (2010): 1-14.
- Aragón, C., and J. A. Aguilera. "Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods." Spectrochimica Acta Part B 63.9 (2008): 893-916.
