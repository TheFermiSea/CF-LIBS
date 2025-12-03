Replaces the current README. It acts as the executive summary and entry point.

Markdown
# CF‑LIBS: Industrial Ultrafast Spectroscopy Engine

**Real-time, Physics-Based Compositional Analysis for Additive Manufacturing (LPBF/DED)**

CF‑LIBS is a production-grade framework for **Ultrafast Laser-Induced Breakdown Spectroscopy**. Unlike traditional approaches that rely on slow, iterative inverse solvers, this project uses a **Spectral Search Engine** paradigm: it pre-computes a massive "Digital Twin" (Manifold) of physical reality using HPC resources, allowing the runtime sensor to perform inference via high-speed vector search (<1 ms latency).

---

## 🏗️ System Architecture

The workflow is decoupled into two asynchronous phases to maximize hardware utilization:

| Phase | Role | Hardware | Tech Stack |
| :--- | :--- | :--- | :--- |
| **1. Simulator** (Offline) | Generates a 1TB+ Manifold of synthetic spectra from first principles (Saha-Boltzmann). | **HPC Cluster** (Tesla V100s) | **Python + JAX** (XLA) |
| **2. Sensor** (Online) | Ingests live data, cleans signals, and identifies composition by searching the Manifold. | **Edge Node** (Xeon Gold) | **Rust** + AVX-512 |

## 📂 Repository Structure


├── docs/                   # Detailed Technical Documentation
│   ├── PHYSICS.md          # Saha-Boltzmann, Time-Integration, & Opacity Models
│   ├── ARCHITECTURE.md     # JAX Simulator & Rust Inference Engine Design
│   └── WORKFLOW.md         # Setup, Data Generation, and Deployment Guide
├── src/                    # Rust Inference Engine (The "Online" Sensor)
│   ├── main.rs             # Actor-based Service Entry
│   ├── identifier.rs       # AirPLS & NNLS Algorithms
│   └── db.rs               # In-Memory Manifold Management
├── scripts/                # Python/JAX Physics Engine (The "Offline" Simulator)
│   ├── manifold_generator.py # HPC JAX Kernel for Manifold Generation
│   ├── datagen_v2.py       # NIST Database Scraper & Pruner
│   └── saha_eggert.py      # Core Physics Prototyping
└── libs_production.db      # SQLite Atomic Database (Generated artifact)
🚀 Key Features
Physics-Informed, Not "AI": Results are fully traceable to NIST atomic transition probabilities. No neural networks, no "black box" hallucination risk.

Ultrafast Physics: Models the unique cooling curve (T∝t 
−α
 ) of femtosecond/picosecond plasmas.

HPC Native: Uses JAX pmap to distribute physics calculations across multi-node GPU clusters.

Industrial Robustness: Implements AirPLS for melt-pool blackbody rejection and NNLS for sparse deconvolution of overlapping lines.

🔗 Documentation Links
Physics Model: How we solve the Saha-Boltzmann equations at scale.

System Architecture: Deep dive into JAX vectorization and Rust actor models.

Development Workflow: How to build the database and run the engine.

Copyright (c) 2025 TheFermiSea. MIT License.


---

### 2. `docs/PHYSICS.md` (The Science)
*Consolidates the physics sections from `DEVELOPMENT_GUIDE_V1` and the original `README`.*

# Physics Model: The "White Box" Manifold

This document details the physical principles governing the **Forward Modeling** engine. Instead of approximating plasma parameters via iterative fitting (Inverse Modeling), we generate a discrete **Manifold $\mathcal{M}$** representing the solution space of the Saha-Boltzmann equations for our specific optical geometry.

## 1. The Forward Model Equation

For a set of plasma parameters $\theta = \{T_{max}, n_{e,max}, \mathbf{C}_{species}\}$, the synthetic spectral radiance $I_{\text{synth}}(\lambda)$ is the time-integrated emission of the cooling plasma trail:

$$
I_{\text{synth}}(\lambda; \theta) = \int_{t_{delay}}^{t_{gate}} \sum_{s \in \text{Species}} \sum_{k \in \text{Lines}} \epsilon_{s,k}(\lambda, T_e(t), n_e(t)) \cdot \text{optics}(\lambda) \, dt
$$

### 1.1 Instantaneous Emissivity
The emissivity $\epsilon_{s,k}$ of a transition $k$ is calculated explicitly:

$$
\epsilon_{s,k} = \frac{h c}{4 \pi \lambda_k} A_{ki} n_s(T, n_e) \frac{g_k}{U_s(T_e)} \exp\left(-\frac{E_k}{k_B T_e}\right) \times \mathcal{V}(\lambda; \lambda_k, \sigma_{Dopp}, \gamma_{Stark})
$$

Where:
* $A_{ki}$: Einstein transition probability (from NIST).
* $n_s$: Species number density, solved via **Saha-Eggert Ionization Balance** to ensure charge neutrality ($n_e = \sum Z_i n_i$).
* $\mathcal{V}$: The **Voigt Profile**, convolving Doppler broadening (Gaussian, $\sim \sqrt{T}$) and Stark broadening (Lorentzian, $\sim n_e$).

## 2. Ultrafast Plasma Specifics

### 2.1 Time-Integration (The "Cooling Trail")
Unlike nanosecond LIBS, ultrafast plasmas have lifetimes $< 10 \mu s$ and expand rapidly. A standard CCD captures the **entire history** of the plasma.
We model the cooling trajectory as a power law:
$$
T_e(t) = T_{max} \left(1 + \frac{t}{t_0}\right)^{-\alpha}, \quad n_e(t) = n_{e,max} \left(1 + \frac{t}{t_0}\right)^{-\beta}
$$
The Manifold Generator integrates this trajectory to produce the spectral signature actually seen by the detector.

### 2.2 Stoichiometry
Ultrafast ablation (<10 ps) occurs faster than the electron-phonon coupling time, leading to **Coulomb Explosion**. This minimizes fractional vaporization, meaning the plasma stoichiometry closely matches the solid alloy ($\mathbf{C}_{plasma} \approx \mathbf{C}_{solid}$), simplifying the matrix correction factors required.

## 3. Opacity & Radiative Transfer
To account for self-absorption in the dense "hot track" plasma, we apply a radiative transfer correction to the source function $S_\lambda$:
$$
I(\lambda) = S_\lambda(\lambda) \left[ 1 - e^{-\tau(\lambda)} \right]
$$
Where optical depth $\tau(\lambda) \propto n_s \cdot f_{osc} \cdot \mathcal{V}(\lambda)$. This naturally saturates strong resonance lines (e.g., Al I 396nm) in the synthetic manifold, allowing the inference engine to match self-absorbed peaks accurately.
3. docs/ARCHITECTURE.md (The Engineering)

Consolidates HIGH_THROUGHPUT_FRAMEWORK.md and hardware implementation details.

Markdown
# System Architecture & HPC Implementation

The system is designed to shift computational load from **Run-Time** (Online) to **Compile-Time** (Offline).

## 1. Offline: The Manifold Generator (JAX)

**Role:** Pre-compute the "Platonic Ideal" of every possible spectrum.
**Hardware:** 3-Node Cluster (Tesla V100 GPUs).

### Why JAX?
We utilize **Google JAX** to achieve near-native CUDA performance with Python readability.
* **XLA Compilation:** Compiles physics functions into optimized fused kernels, eliminating Python overhead.
* **`vmap` (Vectorization):** Automatically transforms scalar physics logic ($1 \to 1$) into vector instructions ($10^8 \to 10^8$), handling grid-strided loops implicitly.
* **`pmap` (Parallelism):** Distributes the parameter grid across multiple GPUs (SPMD paradigm).

### Data Artifact
* **Input:** Parameter Grid ($T_e$: 0.5-2.0eV, $n_e$: $10^{16}$-$10^{19}$, Composition: 0-100%).
* **Output:** Hierarchical Data Format (HDF5) or Apache Parquet (~1 TB).

## 2. Online: The Inference Engine (Rust)

**Role:** High-speed signal processing and manifold search.
**Hardware:** Edge Node (Xeon Gold, >256 GB RAM).

### Microservice Design (Actor Model)
Implemented in **Rust** using the **Tokio** runtime to ensure zero-cost abstractions and memory safety.

1.  **Ingest Actor:** Listens on `ZeroMQ` (PUB/SUB) for spectra from the ICCD driver.
2.  **Compute Actor (CPU Bound):**
    * **AirPLS (Baseline):** Uses sparse Cholesky decomposition to remove the >2000K melt-pool blackbody background.
    * **NNLS (Deconvolution):** Solves $\min \| \mathbf{A}\mathbf{x} - \mathbf{b} \|^2$ s.t. $\mathbf{x} \ge 0$ to separate overlapping peaks.
3.  **Inference Actor:** Performs a k-NN or Dot-Product search against the in-memory Manifold to find the matching $(\theta)$.

### Hardware Optimizations
* **AVX-512:** The Rust linear algebra kernels (`nalgebra`) are compiled with SIMD optimizations for the Xeon Gold processors.
* **Core Pinning:** Worker threads are pinned to specific CPU cores to prevent L1/L2 cache thrashing during matrix operations.
4. docs/WORKFLOW.md (The User Guide)

Consolidates instructions from datagen.py, README.md, and deployment steps.

Markdown
# Development & Deployment Workflow

## 1. Environment Setup

**Prerequisites:**
* Python 3.10+ (JAX, Pandas, NumPy)
* Rust 1.70+ (Cargo)
* HDF5 tools
* Access to NIST ASD (internet connection for initial scrape)

## 2. Data Pipeline (Offline Phase)

### Step A: Build Atomic Database
Run the scraper to build the SQLite database. This pulls line data and filters out high-energy states (>12 eV) invisible to ultrafast LIBS.
```bash
python scripts/datagen_v2.py
# Output: libs_production.db
Step B: Generate Spectral Manifold

Run the JAX simulator on the GPU cluster. Ensure libs_production.db is present.

Bash
# Slurm Example
sbatch --gres=gpu:v100:1 python scripts/manifold_generator.py
# Output: spectral_manifold.h5
3. Sensor Deployment (Online Phase)
Step A: Configure the Rust Engine

Edit src/main.rs to point to your Manifold file and ZeroMQ ports.

Step B: Compile & Run

Build in release mode to enable AVX-512 optimizations.

Bash
cargo build --release
./target/release/cf-libs-rust
4. Operational Logic
Start: Engine loads libs_production.db and spectral_manifold.h5 into RAM (approx. boot time: 30s).

Loop:

Receive Spectrum (ZMQ).

Apply AirPLS (remove thermal background).

Search Manifold for nearest neighbor.

Output Composition Vector via TCP.
```
