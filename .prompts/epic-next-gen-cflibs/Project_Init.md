gpd_version: 1.0.0
epic_id: next-gen-cflibs-analyzer
domain: computational-plasma-physics
primary_goal: Design, optimize, and verify a hybrid PINN/Bayesian CF-LIBS inversion algorithm.

Project: Next-Generation SOTA CF-LIBS Analyzer

1. Executive Summary

Traditional Saha-Boltzmann iterative methods suffer from convergence instabilities and slow partition function evaluations when scaling to high-throughput Planetary Data System (PDS) datasets.

This epic orchestrates the creation of a next-generation Calibration-Free LIBS (CF-LIBS) analyzer that integrates:

Manifold-Accelerated Line Identification: Rust-based high-dimensional vector indexing.

Physics-Informed Neural Networks (PINNs): Deep learning layers constrained by Local Thermodynamic Equilibrium (LTE) equations.

Distributed Bayesian MCMC: Robust uncertainty quantification for plasma state parameters (T, Ne).

2. GPD Subagent Delegation

The work is strictly partitioned into 5 phases. Subagents must sequentially execute their phase files and deposit signed contracts upon completion.

Phase 1: Theory & Formalism (PHASE_1_THEORY.md) -> gpd-literature-reviewer, gpd-project-researcher

Phase 2: Architecture & Contracts (PHASE_2_ARCHITECTURE.md) -> gpd-planner, gpd-plan-checker

Phase 3: Execution & FFI (PHASE_3_EXECUTION.md) -> gpd-executor, gpd-debugger

Phase 4: Verification Parity (PHASE_4_VERIFICATION.md) -> gpd-verifier, gpd-experiment-designer

Phase 5: Synthesis & Publication (PHASE_5_SYNTHESIS.md) -> gpd-research-synthesizer, gpd-paper-writer

3. Strict Codebase Constraints

Performance: All O(N^2) combinatorial matchings must reside in cflibs-core/src/comb_matching.rs.

GPU Compute: MCMC and PINN training must utilize cflibs/core/jax_runtime.py and cflibs/hpc/gpu_config.py.

Validation: The algorithm must pass scripts/validate_nist_parity.py without regressions compared to the legacy iterative solver.
