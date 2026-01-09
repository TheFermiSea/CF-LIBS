# Phase 2c Validation Test Implementation Plan

<objective>
Create a detailed implementation plan for Phase 2c validation tests.

Purpose: Guide test implementation for quality metrics, self-absorption, and line selection
Input: Research findings from @.prompts/001-phase2c-validation-research/
Output: `.prompts/002-phase2c-validation-plan/phase2c-validation-plan.md`
</objective>

<context>
## Research Reference
@.prompts/001-phase2c-validation-research/phase2c-validation-research.md
@.prompts/001-phase2c-validation-research/SUMMARY.md

## Project Files
@cflibs/inversion/quality.py
@cflibs/inversion/self_absorption.py
@cflibs/inversion/line_selection.py
@tests/
@pytest.ini
</context>

<planning_requirements>
Based on research findings, create a plan that addresses:

1. **Test Structure**
   - File organization (where tests live, naming conventions)
   - Fixture design (shared test data, synthetic spectra generators)
   - Marker usage (unit, integration, slow, requires_db)

2. **Phase Breakdown**
   - Phase 1: Unit tests for individual functions
   - Phase 2: Integration tests for module interactions
   - Phase 3: Round-trip validation tests
   - Phase 4: Edge case and failure mode tests

3. **Per-Module Test Plans**
   - quality.py: Which metrics, what thresholds, how to verify
   - self_absorption.py: How to generate optically thick test cases
   - line_selection.py: How to verify scoring and selection logic

4. **Synthetic Data Strategy**
   - Parameters for test plasmas (T, n_e, composition)
   - Noise models and levels
   - Ground truth storage format

5. **Success Criteria**
   - Coverage targets
   - Accuracy bounds for round-trip tests
   - Regression prevention strategy
</planning_requirements>

<output_structure>
Save to: `.prompts/002-phase2c-validation-plan/phase2c-validation-plan.md`

```xml
<plan>
  <summary>{1-2 paragraph overview}</summary>

  <phases>
    <phase number="1" name="{name}">
      <objective>{What this phase achieves}</objective>
      <tasks>
        <task>{Specific task}</task>
      </tasks>
      <deliverables>
        <deliverable>{Output file or artifact}</deliverable>
      </deliverables>
      <dependencies>{What must exist first}</dependencies>
    </phase>
    <!-- More phases -->
  </phases>

  <test_specifications>
    <module name="quality">
      <tests>
        <test name="{test function name}">
          <purpose>{What it verifies}</purpose>
          <approach>{How it works}</approach>
        </test>
      </tests>
    </module>
    <!-- More modules -->
  </test_specifications>

  <fixtures>
    <fixture name="{name}">
      <purpose>{What it provides}</purpose>
      <implementation_notes>{Key details}</implementation_notes>
    </fixture>
  </fixtures>

  <metadata>
    <confidence level="high|medium|low">{Why}</confidence>
    <assumptions>{What was assumed}</assumptions>
    <risks>{Potential issues}</risks>
  </metadata>
</plan>
```
</output_structure>

<summary_requirements>
Create `.prompts/002-phase2c-validation-plan/SUMMARY.md`

Template:
```markdown
# Phase 2c Validation Plan Summary

**{Substantive one-liner: e.g., "4-phase plan: fixtures → unit tests → integration → round-trip validation"}**

## Version
v1

## Key Findings
- {Phase 1 focus}
- {Phase 2 focus}
- {Key architectural decision}

## Decisions Needed
{Specific decisions or "None"}

## Blockers
{External impediments or "None"}

## Next Step
Execute 003-phase2c-validation-implement.md (Phase 1)

---
*Confidence: {High|Medium|Low}*
*Full output: phase2c-validation-plan.md*
```
</summary_requirements>

<success_criteria>
- Clear phase breakdown with dependencies
- Specific test specifications for each Phase 2c module
- Fixture design documented
- Success criteria defined with measurable targets
- Ready for implementation prompt execution
</success_criteria>
