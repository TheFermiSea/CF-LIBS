# Phase 2c Validation Testing Research

<session_initialization>
Before beginning research, verify today's date:
!`date +%Y-%m-%d`

Use this date when searching for "current" or "latest" information.
</session_initialization>

<research_objective>
Research testing and validation approaches for CF-LIBS Phase 2c inversion components.

Purpose: Inform test implementation strategy for quality metrics, self-absorption correction, and line selection modules
Scope: Testing patterns, validation benchmarks, synthetic spectra generation, round-trip verification
Output: `.prompts/001-phase2c-validation-research/phase2c-validation-research.md`
</research_objective>

<context>
## Project Context

CF-LIBS (Calibration-Free Laser-Induced Breakdown Spectroscopy) library with completed Phase 2a+2b:
- Forward model: plasma → atomic_db → emissivity → profiles → instrument → spectrum
- Inversion: Boltzmann plots, closure equation, iterative solver

## Phase 2c Modules to Validate

@cflibs/inversion/quality.py - Quality metrics (R², Saha-Boltzmann consistency, χ²)
@cflibs/inversion/self_absorption.py - Optical depth estimation, recursive correction
@cflibs/inversion/line_selection.py - Quality scoring, resonance line identification

## Existing Test Patterns

@tests/ - Existing test structure
@pytest.ini - Test configuration with markers (slow, requires_db, requires_jax, unit, integration)
</context>

<research_scope>
<include>
1. **Codebase Analysis**
   - Current test patterns in cflibs/tests/
   - How forward model is tested
   - Existing fixtures and test utilities
   - Integration test patterns

2. **Phase 2c Module Internals**
   - QualityAssessor: What metrics does it compute? What are valid thresholds?
   - SelfAbsorptionCorrector: How does optical depth estimation work? What inputs does it need?
   - LineSelector: How is quality scoring calculated? What makes a "good" line?

3. **Validation Strategies**
   - Round-trip testing: forward model → add noise → inversion → compare to ground truth
   - Unit testing individual components (fitting, correction, scoring)
   - Integration testing full inversion pipeline
   - Edge cases and failure modes

4. **Synthetic Test Data Generation**
   - How to create spectra with known ground truth (T, n_e, concentrations)
   - Realistic noise models (Poisson, Gaussian, baseline)
   - Self-absorbed line generation for testing correction
   - Multi-element synthetic plasmas

5. **Scientific Validation**
   - CF-LIBS literature benchmarks (if any standard test cases exist)
   - Expected accuracy bounds (T±5%, n_e±20% typical)
   - What constitutes validation success?
</include>

<exclude>
- Implementation details (for planning phase)
- Actual test code writing (for implementation phase)
- Performance optimization
- JAX/GPU-specific testing
</exclude>
</research_scope>

<verification_checklist>
□ Read all Phase 2c module source files (quality.py, self_absorption.py, line_selection.py)
□ Enumerate all public classes and functions in each module
□ Identify all existing test files and their patterns
□ Document test fixtures and utilities available
□ Check for any existing validation in ROADMAP.md or beads issues
□ Verify understanding of each quality metric formula
□ Document self-absorption correction algorithm steps
□ Understand line scoring formula and its components
</verification_checklist>

<research_quality_assurance>
Before completing research, perform these checks:

<completeness_check>
- [ ] All three Phase 2c modules analyzed in detail
- [ ] Each public API documented with inputs/outputs
- [ ] Existing test patterns catalogued
- [ ] At least 2 validation strategies identified per module
</completeness_check>

<source_verification>
- [ ] Module behavior confirmed by reading source code
- [ ] Any physics claims verified against CLAUDE.md or docstrings
- [ ] Test patterns verified by reading actual test files
</source_verification>

<blind_spots_review>
Ask yourself: "What might I have missed?"
- [ ] Are there edge cases not covered by obvious tests?
- [ ] What happens with pathological inputs (empty arrays, NaN, single lines)?
- [ ] Are there interactions between modules that need testing?
- [ ] What about performance/convergence testing?
</blind_spots_review>
</research_quality_assurance>

<output_structure>
Save to: `.prompts/001-phase2c-validation-research/phase2c-validation-research.md`

Write findings incrementally as you discover them:

1. Create file with initial XML skeleton
2. Append each finding as discovered
3. Add code examples as found
4. Finalize metadata and summary at end

Structure:
```xml
<research>
  <summary>
    {2-3 paragraph executive summary}
  </summary>

  <findings>
    <finding category="quality_metrics">
      <title>{Finding}</title>
      <detail>{Details}</detail>
      <source>{File:line or documentation}</source>
      <relevance>{Why this matters for testing}</relevance>
    </finding>
    <!-- More findings per category -->
  </findings>

  <recommendations>
    <recommendation priority="high|medium|low">
      <action>{What to do}</action>
      <rationale>{Why}</rationale>
    </recommendation>
  </recommendations>

  <code_examples>
    {Test patterns, fixture examples, validation code}
  </code_examples>

  <metadata>
    <confidence level="high|medium|low">{Why}</confidence>
    <dependencies>{What's needed}</dependencies>
    <open_questions>{Unknowns}</open_questions>
    <assumptions>{What was assumed}</assumptions>
  </metadata>
</research>
```
</output_structure>

<summary_requirements>
Create `.prompts/001-phase2c-validation-research/SUMMARY.md`

Template:
```markdown
# Phase 2c Validation Research Summary

**{Substantive one-liner: e.g., "3-tier testing strategy: unit → integration → round-trip with synthetic spectra"}**

## Version
v1

## Key Findings
- {Most important finding}
- {Second key finding}
- {Third key finding}

## Decisions Needed
{Specific decisions or "None"}

## Blockers
{External impediments or "None"}

## Next Step
Create 002-phase2c-validation-plan.md

---
*Confidence: {High|Medium|Low}*
*Full output: phase2c-validation-research.md*
```
</summary_requirements>

<success_criteria>
- All three Phase 2c modules analyzed with public API documented
- At least 3 validation strategies identified with pros/cons
- Synthetic data generation approach outlined
- Test file locations and patterns documented
- Clear recommendations for test implementation prioritization
- SUMMARY.md created with substantive one-liner
</success_criteria>

<tool_hints>
- Use Read tool for source files (quality.py, self_absorption.py, line_selection.py)
- Use Grep to find existing test patterns
- Use Glob to discover test file locations
- Check beads issues: `bd list` for any related validation tasks
</tool_hints>
