Phase 1: Theoretical Setup & Literature Review

Assigned Agents: @gpd-literature-reviewer, @gpd-project-researcher

Objectives

Establish the rigorous mathematical foundation for integrating Physics-Informed Neural Networks (PINNs) with Bayesian inversion for CF-LIBS.

Input Context

Read and synthesize the following existing artifacts:

docs/literature/matrix_effect_correction_methods.md

docs/literature/high_performance_libs_algorithms.md

cflibs/plasma/saha_boltzmann.py (Current implementation limitations)

Tasks

Define the PINN Loss Function:

Formulate the data loss term ($\mathcal{L}_{data}$) matching synthetic/empirical spectra.

Formulate the physics loss term ($\mathcal{L}_{physics}$) strictly enforcing the Saha ionization equations and Boltzmann distribution over excited states.

Account for optical thinness approximations and self-absorption penalties (cflibs/inversion/self_absorption.py).

Define the Bayesian Priors:

Establish mathematically sound priors for electron density ($N_e$) and plasma temperature ($T$).

Define Dirichlet priors for elemental mass fractions ($C_i$) (reference tests/test_closure_dirichlet.py).

Deliverable

Generate docs/design/pinn_bayesian_formalism.md. This document must contain the complete LaTeX mathematical derivations and the theoretical proofs required by the gpd-plan-checker.
