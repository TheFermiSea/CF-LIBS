# CF-LIBS

A Python library for **Calibration-Free Laser-Induced Breakdown Spectroscopy (CF-LIBS)**: quantitative elemental analysis without calibration standards, using plasma physics to calculate compositions directly from spectral line intensities.

## What is CF-LIBS?

Calibration-Free LIBS is an established quantitative spectroscopy technique where elemental concentrations are determined directly from spectral measurements using fundamental plasma physics equations, eliminating the need for matrix-matched calibration standards.

**Key principle**: The intensity of an emission line relates to elemental concentration through:

```
I_ki = F × C_s × (g_k × A_ki / U(T)) × exp(-E_k / kT)
```

By measuring multiple lines from multiple elements and applying the **closure equation** (ΣC = 1), both temperature and concentrations can be determined self-consistently.

## Current Status

**Phase 1 Complete** - Forward modeling works for simple LTE plasmas with Gaussian line profiles.

**Phase 2 In Progress** - Implementing full CF-LIBS workflow (see [ROADMAP.md](ROADMAP.md)).

### What Works Now

- **Forward Modeling Pipeline**: Plasma state → Saha-Boltzmann → Line emissivity → Spectrum
- **Saha-Boltzmann Solver**: Ionization equilibrium for up to 3 ionization stages
- **Gaussian Line Profiles**: Temperature-dependent Doppler broadening
- **Atomic Database**: SQLite interface with NIST data, connection pooling, caching
- **Instrument Convolution**: Gaussian instrument function, response curves
- **Echelle Spectrometer Support**: Order extraction, wavelength calibration
- **Manifold Generation**: JAX-accelerated pre-computed spectral lookup (simplified physics)

### What's Missing (Phase 2)

- **Classic CF-LIBS Algorithm**: Iterative Boltzmann/Saha solver with closure equation
- **Voigt/Lorentzian Profiles**: Currently Gaussian-only
- **Stark Broadening**: Required for electron density determination
- **Self-Absorption Correction**: Required for optically thick plasmas
- **Temperature-Dependent Partition Functions**: Manifold uses hard-coded constants

See [ROADMAP.md](ROADMAP.md) for the complete development plan.

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Pandas, PyYAML

### Install from Source

```bash
git clone https://github.com/yourusername/CF-LIBS.git
cd CF-LIBS
pip install -e ".[dev]"
```

### Generate Atomic Database

Before using CF-LIBS, generate the atomic database from NIST:

```bash
cflibs generate-db
# Or directly:
python datagen_v2.py
```

**Note**: Database generation requires internet access and can take several hours.

## Quick Start

### Forward Model (What Works Now)

```python
from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.state import SingleZoneLTEPlasma
from cflibs.instrument.model import InstrumentModel
from cflibs.radiation.spectrum_model import SpectrumModel

# Load atomic data
db = AtomicDatabase("libs_production.db")

# Define plasma
plasma = SingleZoneLTEPlasma(
    T_e=10000.0,  # K
    n_e=1e17,     # cm^-3
    species={"Fe": 1e15, "Ti": 5e14}
)

# Create spectrum model
model = SpectrumModel(
    plasma=plasma,
    atomic_db=db,
    instrument=InstrumentModel(resolution_fwhm_nm=0.05),
    lambda_min=300.0,
    lambda_max=500.0,
    delta_lambda=0.01
)

# Generate spectrum
wavelength, intensity = model.compute_spectrum()
```

### CLI Usage

```bash
# Generate spectrum from config file
cflibs forward examples/config_example.yaml --output spectrum.csv

# Generate manifold for fast inference
cflibs generate-manifold examples/manifold_config_example.yaml --progress
```

## Project Structure

```
cflibs/
├── core/        # Constants, units, caching, factories, ABCs
├── atomic/      # Atomic data structures, SQLite database
├── plasma/      # Plasma state, Saha-Boltzmann solver
├── radiation/   # Emissivity, profiles, spectrum generation
├── instrument/  # Instrument response, echelle support
├── manifold/    # Pre-computed spectral manifolds (JAX)
├── inversion/   # CF-LIBS inversion [PLACEHOLDER - Phase 2]
├── hardware/    # Hardware interfaces [STUBS]
├── io/          # File I/O utilities
└── cli/         # Command-line interface
```

## Documentation

- [ROADMAP.md](ROADMAP.md) - Development roadmap with detailed phases
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/User_Guide.md](docs/User_Guide.md) - User documentation
- [docs/API_Reference.md](docs/API_Reference.md) - API documentation
- [docs/Database_Generation.md](docs/Database_Generation.md) - Atomic database generation

## Development

### Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/test_plasma.py -v      # Specific module
pytest -m "not requires_db"         # Skip database tests
```

### Code Quality

```bash
black cflibs/ tests/                # Format code
ruff check cflibs/ tests/           # Lint
mypy cflibs/                        # Type check
```

### Issue Tracking

This project uses `bd` (beads) for issue tracking:

```bash
bd ready              # See available work
bd list               # All issues
bd show CF-LIBS-xxx   # Issue details
```

## Physics Background

CF-LIBS assumes:

1. **Local Thermodynamic Equilibrium (LTE)**: Boltzmann distribution for level populations
2. **Saha Ionization Balance**: Temperature-dependent ionization equilibrium
3. **Closure Constraint**: All concentrations sum to 100%

The iterative CF-LIBS algorithm:

1. Estimate initial T, n_e
2. Calculate partition functions U(T)
3. Build Boltzmann plot (ln(Iλ/gA) vs E_k)
4. Fit slope → temperature
5. Extract intercepts → concentrations
6. Apply closure (ΣC = 1)
7. Update n_e via Saha equation
8. Iterate until convergence

See the literature (Ciucci et al., Tognoni et al.) for detailed methodology.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
- Voigt/Lorentzian line profiles
- Stark broadening implementation
- Classic CF-LIBS solver
- Self-absorption correction

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- NIST Atomic Spectra Database for atomic data
- CF-LIBS methodology from the LIBS research community
