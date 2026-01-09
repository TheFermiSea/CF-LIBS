# Phase 2c Validation Test Implementation Plan

<plan>
  <summary>
    This plan defines a 4-phase implementation strategy for Phase 2c validation testing, covering quality metrics, self-absorption correction, and line selection modules. The approach prioritizes reusable fixtures, deterministic unit tests, and round-trip validation using synthetic spectra with known ground truth.

    Total scope: ~102 tests across 3 test files, with 7 shared fixture factories in conftest.py. Target coverage: 80%+ for all Phase 2c modules.
  </summary>

  <phases>
    <phase number="1" name="Fixtures and Test Utilities">
      <objective>Create reusable test fixtures and synthetic data generators</objective>
      <tasks>
        <task>Add synthetic_observations fixture factory (T, n_e, wavelength, intensity arrays)</task>
        <task>Add quality_input_set fixture (pre-configured QualityMetrics inputs)</task>
        <task>Add self_absorption_test_line fixture (optical depth, original/absorbed intensities)</task>
        <task>Add line_selector_test_data fixture (scored lines with known isolation)</task>
        <task>Add boltzmann_plot_result fixture (mock BoltzmannPlotResult objects)</task>
        <task>Add closure_result fixture (mock ClosureResult objects)</task>
        <task>Add sample_atomic_transitions fixture (realistic transition data)</task>
      </tasks>
      <deliverables>
        <deliverable>tests/conftest.py (extended with Phase 2c fixtures)</deliverable>
      </deliverables>
      <dependencies>Existing test infrastructure, NumPy</dependencies>
    </phase>

    <phase number="2" name="Unit Tests">
      <objective>Test individual functions and methods in isolation</objective>
      <tasks>
        <task>Implement test_quality.py unit tests (38 tests)</task>
        <task>Implement test_self_absorption.py unit tests (30 tests)</task>
        <task>Implement test_line_selection.py unit tests (34 tests)</task>
      </tasks>
      <deliverables>
        <deliverable>tests/test_quality.py</deliverable>
        <deliverable>tests/test_self_absorption.py</deliverable>
        <deliverable>tests/test_line_selection.py</deliverable>
      </deliverables>
      <dependencies>Phase 1 fixtures</dependencies>
    </phase>

    <phase number="3" name="Integration Tests">
      <objective>Test module interactions and full pipelines</objective>
      <tasks>
        <task>Add integration tests for QualityAssessor with real BoltzmannPlotResult</task>
        <task>Add integration tests for SelfAbsorptionCorrector with mock AtomicDatabase</task>
        <task>Add integration tests for LineSelector with AtomicDatabase queries</task>
      </tasks>
      <deliverables>
        <deliverable>Integration test sections in each test file</deliverable>
      </deliverables>
      <dependencies>Phase 2 unit tests, mock atomic database</dependencies>
    </phase>

    <phase number="4" name="Round-Trip Validation">
      <objective>Verify correction algorithms can recover known ground truth</objective>
      <tasks>
        <task>Self-absorption round-trip: apply absorption → correct → verify recovery</task>
        <task>Quality metrics validation: known-good vs known-bad spectra</task>
        <task>Line selection validation: verify scoring ranks lines correctly</task>
      </tasks>
      <deliverables>
        <deliverable>Round-trip test functions in each test file</deliverable>
      </deliverables>
      <dependencies>Phase 2-3 tests passing</dependencies>
    </phase>
  </phases>

  <test_specifications>
    <module name="quality">
      <tests>
        <test name="test_quality_metrics_dataclass_creation">
          <purpose>Verify QualityMetrics can be instantiated with all fields</purpose>
          <approach>Create instance, verify all attributes accessible</approach>
        </test>
        <test name="test_quality_metrics_to_dict">
          <purpose>Verify serialization to dictionary</purpose>
          <approach>Create instance, call to_dict(), verify keys and values</approach>
        </test>
        <test name="test_quality_assessor_init_default_thresholds">
          <purpose>Verify default thresholds are set correctly</purpose>
          <approach>Instantiate QualityAssessor, check THRESHOLDS dict</approach>
        </test>
        <test name="test_assess_boltzmann_fit_excellent">
          <purpose>Verify R2 > 0.95 classified as excellent</purpose>
          <approach>Mock BoltzmannPlotResult with R2=0.98, verify grade</approach>
        </test>
        <test name="test_assess_boltzmann_fit_good">
          <purpose>Verify 0.90 < R2 < 0.95 classified as good</purpose>
          <approach>Mock BoltzmannPlotResult with R2=0.92, verify grade</approach>
        </test>
        <test name="test_assess_boltzmann_fit_acceptable">
          <purpose>Verify 0.80 < R2 < 0.90 classified as acceptable</purpose>
          <approach>Mock BoltzmannPlotResult with R2=0.85, verify grade</approach>
        </test>
        <test name="test_assess_boltzmann_fit_poor">
          <purpose>Verify R2 < 0.80 classified as poor</purpose>
          <approach>Mock BoltzmannPlotResult with R2=0.65, verify grade</approach>
        </test>
        <test name="test_assess_closure_excellent">
          <purpose>Verify closure < 0.01 classified as excellent</purpose>
          <approach>Mock ClosureResult with closure=0.005, verify grade</approach>
        </test>
        <test name="test_assess_closure_good">
          <purpose>Verify 0.01 < closure < 0.05 classified as good</purpose>
          <approach>Mock ClosureResult with closure=0.03, verify grade</approach>
        </test>
        <test name="test_assess_closure_acceptable">
          <purpose>Verify 0.05 < closure < 0.10 classified as acceptable</purpose>
          <approach>Mock ClosureResult with closure=0.07, verify grade</approach>
        </test>
        <test name="test_assess_closure_poor">
          <purpose>Verify closure > 0.10 classified as poor</purpose>
          <approach>Mock ClosureResult with closure=0.15, verify grade</approach>
        </test>
        <test name="test_compute_reconstruction_chi_squared_perfect">
          <purpose>Verify chi2=0 for identical observed/reconstructed</purpose>
          <approach>Pass identical arrays, verify chi2 ≈ 0</approach>
        </test>
        <test name="test_compute_reconstruction_chi_squared_with_noise">
          <purpose>Verify chi2 scales with residual magnitude</purpose>
          <approach>Add known noise level, verify chi2 proportional</approach>
        </test>
        <test name="test_assess_full_metrics">
          <purpose>Verify full assessment returns all metrics</purpose>
          <approach>Provide complete inputs, verify QualityMetrics populated</approach>
        </test>
        <test name="test_quality_assessor_with_none_inputs">
          <purpose>Verify graceful handling of None optional inputs</purpose>
          <approach>Pass None for optional parameters, verify no crash</approach>
        </test>
        <test name="test_quality_metrics_immutability">
          <purpose>Verify dataclass is frozen/immutable</purpose>
          <approach>Attempt to modify field, expect AttributeError</approach>
        </test>
      </tests>
    </module>

    <module name="self_absorption">
      <tests>
        <test name="test_self_absorption_corrector_init">
          <purpose>Verify corrector initializes with default parameters</purpose>
          <approach>Instantiate, verify optical_depth_threshold=0.1, max_iterations=10</approach>
        </test>
        <test name="test_correction_factor_optically_thin">
          <purpose>Verify f(τ) ≈ 1 for τ << 1</purpose>
          <approach>Calculate f(0.01), verify ≈ 1.0 within tolerance</approach>
        </test>
        <test name="test_correction_factor_moderate_depth">
          <purpose>Verify f(τ) formula for τ = 1</purpose>
          <approach>Calculate f(1), verify = (1-exp(-1))/1 ≈ 0.632</approach>
        </test>
        <test name="test_correction_factor_optically_thick">
          <purpose>Verify f(τ) → 0 as τ → ∞</purpose>
          <approach>Calculate f(10), verify < 0.1</approach>
        </test>
        <test name="test_round_trip_correction">
          <purpose>Verify absorption then correction recovers original</purpose>
          <approach>Apply f(τ) to intensity, then inverse, compare to original</approach>
        </test>
        <test name="test_estimate_optical_depth_from_doublet_ratio">
          <purpose>Verify optical depth estimation from intensity ratio</purpose>
          <approach>Create doublet with known τ, verify estimated τ matches</approach>
        </test>
        <test name="test_recursive_correction_convergence">
          <purpose>Verify recursive correction converges within max_iterations</purpose>
          <approach>Apply correction, verify convergence flag True</approach>
        </test>
        <test name="test_recursive_correction_max_iterations">
          <purpose>Verify iteration limit is respected</purpose>
          <approach>Set max_iterations=2 for high τ, verify stops at 2</approach>
        </test>
        <test name="test_correction_preserves_relative_intensities">
          <purpose>Verify correction doesn't distort relative line intensities</purpose>
          <approach>Correct multiple lines, verify ratios maintained for thin lines</approach>
        </test>
        <test name="test_mask_threshold_skips_weak_lines">
          <purpose>Verify weak lines below mask threshold are skipped</purpose>
          <approach>Set mask_threshold=3.0, pass weak line, verify uncorrected</approach>
        </test>
        <test name="test_negative_optical_depth_clamped">
          <purpose>Verify negative τ estimates are clamped to 0</purpose>
          <approach>Pass ratio > theoretical, verify τ = 0</approach>
        </test>
        <test name="test_correction_with_zero_intensity">
          <purpose>Verify zero intensity handled gracefully</purpose>
          <approach>Pass zero intensity, verify no division by zero</approach>
        </test>
      </tests>
    </module>

    <module name="line_selection">
      <tests>
        <test name="test_line_selector_init">
          <purpose>Verify selector initializes with default parameters</purpose>
          <approach>Instantiate, verify min_snr=3.0, isolation_wavelength_nm=0.1</approach>
        </test>
        <test name="test_compute_score_formula">
          <purpose>Verify score = SNR * (1/uncertainty) * isolation</purpose>
          <approach>Create line with known values, verify computed score</approach>
        </test>
        <test name="test_compute_isolation_zero_separation">
          <purpose>Verify isolation = 0 when lines overlap</purpose>
          <approach>Pass separation=0, verify isolation=0</approach>
        </test>
        <test name="test_compute_isolation_large_separation">
          <purpose>Verify isolation ≈ 1 for well-separated lines</purpose>
          <approach>Pass separation >> isolation_wavelength_nm, verify ≈ 1</approach>
        </test>
        <test name="test_compute_isolation_formula">
          <purpose>Verify 1 - exp(-separation/isolation_wavelength_nm)</purpose>
          <approach>Pass known separation, verify exact formula result</approach>
        </test>
        <test name="test_select_lines_respects_min_snr">
          <purpose>Verify lines below min_snr are rejected</purpose>
          <approach>Pass lines with SNR < 3.0, verify rejected</approach>
        </test>
        <test name="test_select_lines_respects_max_atomic_uncertainty">
          <purpose>Verify high uncertainty lines rejected</purpose>
          <approach>Pass lines with uncertainty > threshold, verify rejected</approach>
        </test>
        <test name="test_rejection_reasons_tracked">
          <purpose>Verify rejection reasons are recorded</purpose>
          <approach>Pass various bad lines, verify each has rejection_reason</approach>
        </test>
        <test name="test_select_lines_returns_sorted_by_score">
          <purpose>Verify selected lines sorted by score descending</purpose>
          <approach>Pass unsorted lines, verify output sorted</approach>
        </test>
        <test name="test_min_lines_per_element">
          <purpose>Verify minimum lines per element constraint</purpose>
          <approach>Set min_lines_per_element=3, verify at least 3 returned per element</approach>
        </test>
        <test name="test_max_lines_per_element">
          <purpose>Verify maximum lines per element constraint</purpose>
          <approach>Set max_lines_per_element=5, verify at most 5 returned</approach>
        </test>
        <test name="test_identify_resonance_lines_stub">
          <purpose>Verify resonance identification returns empty set (stub)</purpose>
          <approach>Call identify_resonance_lines, verify returns set()</approach>
        </test>
        <test name="test_select_lines_empty_input">
          <purpose>Verify empty input returns empty output</purpose>
          <approach>Pass empty list, verify empty result</approach>
        </test>
        <test name="test_select_lines_all_rejected">
          <purpose>Verify graceful handling when all lines rejected</purpose>
          <approach>Pass all low-SNR lines, verify empty result with reasons</approach>
        </test>
      </tests>
    </module>
  </test_specifications>

  <fixtures>
    <fixture name="synthetic_observations">
      <purpose>Generate synthetic spectral observations with known parameters</purpose>
      <implementation_notes>
        Factory function taking T, n_e, wavelength range. Uses Boltzmann distribution
        to generate line intensities. Returns dict with wavelength, intensity arrays
        and ground_truth dict containing T, n_e, concentrations.
      </implementation_notes>
    </fixture>

    <fixture name="quality_input_set">
      <purpose>Pre-configured inputs for QualityAssessor testing</purpose>
      <implementation_notes>
        Returns tuple of (BoltzmannPlotResult, ClosureResult, observed, reconstructed)
        with parametrized quality levels (excellent, good, acceptable, poor).
      </implementation_notes>
    </fixture>

    <fixture name="self_absorption_test_line">
      <purpose>Generate line with known optical depth for testing correction</purpose>
      <implementation_notes>
        Factory taking optical_depth parameter. Returns dict with original_intensity,
        absorbed_intensity (via f(τ)), wavelength, and expected corrected intensity.
        Uses formula: absorbed = original * (1 - exp(-τ)) / τ
      </implementation_notes>
    </fixture>

    <fixture name="line_selector_test_data">
      <purpose>Generate lines with known scores for testing selection</purpose>
      <implementation_notes>
        Factory returning list of AtomicTransition-like objects with configurable
        SNR, uncertainty, and isolation. Includes pre-computed expected scores.
      </implementation_notes>
    </fixture>

    <fixture name="mock_boltzmann_plot_result">
      <purpose>Create mock BoltzmannPlotResult with configurable R2</purpose>
      <implementation_notes>
        Uses @dataclass or SimpleNamespace. Accepts r_squared, temperature,
        intercept, n_points parameters. Used for QualityAssessor testing.
      </implementation_notes>
    </fixture>

    <fixture name="mock_closure_result">
      <purpose>Create mock ClosureResult with configurable closure error</purpose>
      <implementation_notes>
        Accepts closure_error, concentrations, electron_density parameters.
        Used for QualityAssessor closure grading tests.
      </implementation_notes>
    </fixture>

    <fixture name="sample_atomic_transitions">
      <purpose>Realistic transition data for integration tests</purpose>
      <implementation_notes>
        Returns list of transitions for common elements (Fe, Cu, Al).
        Includes wavelength, gA, E_upper, element, ion_stage.
        Can be filtered by element or ion stage.
      </implementation_notes>
    </fixture>
  </fixtures>

  <metadata>
    <confidence level="high">
      All Phase 2c modules have been analyzed. Thresholds and formulas are documented
      in source code docstrings. Test patterns follow existing codebase conventions.
    </confidence>
    <assumptions>
      - AtomicDatabase can be mocked for unit tests
      - Existing test_boltzmann.py patterns are acceptable
      - 80%+ coverage is achievable target
    </assumptions>
    <risks>
      - SelfAbsorptionCorrector.identify_resonance_lines() is a stub (returns empty)
      - Some edge cases may require real database for integration tests
    </risks>
  </metadata>
</plan>
