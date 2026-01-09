# Phase 2c Validation Test Implementation

<objective>
Implement validation tests for CF-LIBS Phase 2c inversion components.

Purpose: Create comprehensive test coverage for quality metrics, self-absorption, and line selection
Input: Implementation plan from @.prompts/002-phase2c-validation-plan/
Output: Test files in tests/ directory
</objective>

<context>
## Plan Reference
@.prompts/002-phase2c-validation-plan/phase2c-validation-plan.md
@.prompts/002-phase2c-validation-plan/SUMMARY.md

## Research Reference
@.prompts/001-phase2c-validation-research/phase2c-validation-research.md

## Implementation Targets
@cflibs/inversion/quality.py
@cflibs/inversion/self_absorption.py
@cflibs/inversion/line_selection.py

## Test Infrastructure
@tests/
@tests/conftest.py (if exists)
@pytest.ini
</context>

<requirements>
## Functional Requirements
- Implement all tests specified in the plan
- Follow existing test patterns in the codebase
- Use appropriate pytest markers (unit, integration, slow, requires_db)
- Create reusable fixtures for synthetic spectra generation

## Quality Requirements
- All tests must pass
- Tests must be deterministic (seeded random where applicable)
- Clear test names that describe what is being verified
- Docstrings for complex test logic

## Constraints
- Follow cflibs code conventions (type hints, NumPy docstrings)
- Keep test files organized by module
- Avoid code duplication through fixtures
</requirements>

<implementation>
Follow the phases defined in the plan:

1. **Phase 1**: Fixtures and test utilities
2. **Phase 2**: Unit tests
3. **Phase 3**: Integration tests
4. **Phase 4**: Round-trip validation

For each phase:
- Read the plan specifications
- Implement the tests
- Run tests to verify they pass
- Move to next phase
</implementation>

<output>
Create test files as specified in the plan. Expected structure:
```
tests/
├── conftest.py          # Shared fixtures (add to existing or create)
├── test_quality.py      # Quality metrics tests
├── test_self_absorption.py  # Self-absorption tests
└── test_line_selection.py   # Line selection tests
```
</output>

<verification>
After implementation:
```bash
# Run all new tests
pytest tests/test_quality.py tests/test_self_absorption.py tests/test_line_selection.py -v

# Run with coverage
pytest tests/test_quality.py tests/test_self_absorption.py tests/test_line_selection.py --cov=cflibs/inversion --cov-report=term-missing

# Run only unit tests (fast feedback)
pytest -m unit tests/test_quality.py tests/test_self_absorption.py tests/test_line_selection.py -v
```
</verification>

<summary_requirements>
Create `.prompts/003-phase2c-validation-implement/SUMMARY.md`

Template:
```markdown
# Phase 2c Validation Implementation Summary

**{Substantive one-liner: e.g., "27 tests across 3 files with 94% coverage of Phase 2c modules"}**

## Version
v1

## Key Findings
- {Test count and pass status}
- {Coverage achieved}
- {Any issues discovered}

## Files Created
- `tests/test_quality.py` - Quality metrics tests
- `tests/test_self_absorption.py` - Self-absorption tests
- `tests/test_line_selection.py` - Line selection tests
- `tests/conftest.py` - Shared fixtures (modified)

## Decisions Needed
{Any follow-up decisions or "None"}

## Blockers
{Any issues encountered or "None"}

## Next Step
{Next action: e.g., "Run full test suite", "Update ROADMAP.md", "Close beads issues"}

---
*Confidence: {High|Medium|Low}*
```
</summary_requirements>

<success_criteria>
- All specified tests implemented and passing
- Coverage target met (per plan specifications)
- Tests follow codebase conventions
- Fixtures are reusable and documented
- SUMMARY.md reflects actual implementation results
</success_criteria>
