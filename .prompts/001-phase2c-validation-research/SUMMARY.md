# Phase 2c Validation Research Summary

**3-tier testing strategy: unit tests per module → integration with mock data → round-trip validation with synthetic spectra**

## Version
v1

## Key Findings
- QualityAssessor has documented thresholds (R²>0.95 excellent, closure<0.01 excellent) enabling deterministic unit tests
- SelfAbsorptionCorrector uses invertible formula f(τ)=(1-exp(-τ))/τ - can test by applying absorption then verifying correction recovers original
- LineSelector scoring is multiplicative (SNR × 1/uncertainty × isolation) with rejection reasons tracked - easily testable with controlled inputs
- Existing test_boltzmann.py:12-52 has create_synthetic_lines() pattern to extend for Phase 2c fixtures

## Decisions Needed
- Whether to use real AtomicDatabase or mocks for integration tests
- Coverage target threshold (recommend 80%+)

## Blockers
None

## Next Step
Create 002-phase2c-validation-plan.md with detailed test specifications per module

---
*Confidence: High*
*Full output: phase2c-validation-research.md*
