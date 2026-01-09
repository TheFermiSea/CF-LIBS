<research>
  <summary>
    Phase 2c validation requires comprehensive testing of three inversion modules: QualityAssessor (quality metrics), SelfAbsorptionCorrector, and LineSelector. The existing test suite provides excellent patterns including synthetic line generation (test_boltzmann.py:12-52), mock database fixtures (conftest.py, test_solver.py), and integration workflows (test_integration.py).

    The key validation strategy is round-trip testing: generate synthetic spectra with known T, n_e, and concentrations using the forward model, add controlled noise, run inversion, and verify parameter recovery within tolerance bounds (T±5%, n_e±20%, C±10% per CF-LIBS literature). Each Phase 2c module also requires unit tests for edge cases: QualityAssessor thresholds, SelfAbsorptionCorrector optical depth estimation, and LineSelector scoring algorithm.

    Critical finding: The existing `create_synthetic_lines()` helper in test_boltzmann.py provides the foundation for test fixtures. Self-absorption testing requires generating optically thick lines with known τ, which can be achieved by manipulating intensities according to the correction formula f(τ) = (1 - exp(-τ))/τ.
  </summary>

  <findings>
    <finding category="module_analysis">
      <title>QualityAssessor has well-defined thresholds</title>
      <detail>
        The QualityMetrics dataclass (quality.py:19-65) documents explicit thresholds:
        - r_squared_boltzmann: >0.95 excellent, >0.90 good, >0.80 acceptable
        - saha_boltzmann_consistency: <0.10 excellent, <0.20 good, <0.30 acceptable
        - inter_element_t_std_frac: <0.05 excellent, <0.10 good, <0.15 acceptable
        - closure_residual: <0.01 excellent, <0.05 good, <0.10 acceptable

        The THRESHOLDS dict (quality.py:74-80) also includes reduced_chi2 thresholds.
        Quality flags: "excellent", "good", "acceptable", "poor", "reject"
      </detail>
      <source>cflibs/inversion/quality.py:19-80</source>
      <relevance>Tests can verify threshold behavior by crafting inputs that produce known metric values</relevance>
    </finding>

    <finding category="module_analysis">
      <title>QualityAssessor.assess() requires complex inputs</title>
      <detail>
        The assess() method (quality.py:105-198) requires:
        - observations: List[LineObservation]
        - temperature_K, electron_density_cm3: floats
        - concentrations: Dict[str, float]
        - ionization_potentials: Dict[str, float]
        - partition_funcs_I, partition_funcs_II: Dict[str, float]

        This requires fixtures that provide all these parameters consistently.
        Key internal methods to test:
        - _compute_pooled_r_squared() (quality.py:200-263)
        - _compute_per_element_fits() (quality.py:265-291)
        - _compute_saha_consistency() (quality.py:293-349)
        - _determine_quality_flag() (quality.py:351-371)
      </detail>
      <source>cflibs/inversion/quality.py:105-198</source>
      <relevance>Need to build comprehensive fixture that provides all required inputs</relevance>
    </finding>

    <finding category="module_analysis">
      <title>SelfAbsorptionCorrector uses scaling formula</title>
      <detail>
        Core correction: I_true = I_measured / f(τ) where f(τ) = (1 - exp(-τ))/τ

        Optical depth estimation (self_absorption.py:188-246):
        τ ≈ SCALE_FACTOR × A_ki × λ³ × n_i × L
        Where SCALE_FACTOR = 1e-25 (empirical)

        Thresholds (self_absorption.py:61-68):
        - optical_depth_threshold: 0.1 (below = optically thin, no correction)
        - mask_threshold: 3.0 (above = mask line instead of correct)

        Recursive correction iterates up to max_iterations=5 with convergence_tolerance=0.01
      </detail>
      <source>cflibs/inversion/self_absorption.py:49-186</source>
      <relevance>Can test by creating lines with known τ and verifying correction factor</relevance>
    </finding>

    <finding category="module_analysis">
      <title>estimate_optical_depth_from_intensity_ratio() is independently testable</title>
      <detail>
        Standalone function (self_absorption.py:323-385) that estimates τ from doublet line ratios.
        Uses bisection search to solve f(τ) = ratio_reduction.

        Test approach: Create doublet with known theoretical_ratio, apply self-absorption
        to stronger line, verify estimated τ matches applied τ.
      </detail>
      <source>cflibs/inversion/self_absorption.py:323-385</source>
      <relevance>Unit test candidate - no complex dependencies</relevance>
    </finding>

    <finding category="module_analysis">
      <title>LineSelector scoring formula is multiplicative</title>
      <detail>
        Score = SNR × (1/atomic_uncertainty) × isolation_factor (line_selection.py:244-248)

        Key parameters:
        - min_snr: 10.0 (default)
        - min_energy_spread_ev: 2.0
        - min_lines_per_element: 3
        - isolation_wavelength_nm: 0.1
        - max_lines_per_element: 20

        Isolation computed via exponential decay (line_selection.py:259-286):
        isolation = 1.0 - exp(-min_separation / isolation_wavelength_nm)

        Rejection reasons tracked in LineScore.rejection_reason
      </detail>
      <source>cflibs/inversion/line_selection.py:43-257</source>
      <relevance>Can test scoring with controlled SNR, wavelength spacing, and uncertainty values</relevance>
    </finding>

    <finding category="module_analysis">
      <title>identify_resonance_lines() is a stub</title>
      <detail>
        Function at line_selection.py:325-350 currently returns empty set.
        Comment indicates it requires lower level energy from database.

        This is a known limitation - tests should document this behavior.
      </detail>
      <source>cflibs/inversion/line_selection.py:325-350</source>
      <relevance>Test should verify stub behavior; future work to implement properly</relevance>
    </finding>

    <finding category="test_patterns">
      <title>Synthetic line generation pattern established</title>
      <detail>
        test_boltzmann.py:12-52 has create_synthetic_lines() that:
        1. Takes T_K, n_points, noise_level
        2. Generates energies in 2-6 eV range
        3. Computes expected y = ln(const) - E/T_eV
        4. Adds Gaussian noise
        5. Back-calculates intensity
        6. Returns List[LineObservation]

        This can be extended for Phase 2c testing.
      </detail>
      <source>tests/test_boltzmann.py:12-52</source>
      <relevance>Foundation for Phase 2c test fixtures</relevance>
    </finding>

    <finding category="test_patterns">
      <title>Mock database pattern for solver tests</title>
      <detail>
        test_solver.py:13-29 creates mock_db fixture with:
        - MagicMock(spec=AtomicDatabase)
        - Mocked get_ionization_potential()
        - Mocked get_partition_coefficients() returning PartitionFunction

        This avoids need for real database in unit tests.
      </detail>
      <source>tests/test_solver.py:13-29</source>
      <relevance>Can reuse pattern for Phase 2c tests that need atomic data</relevance>
    </finding>

    <finding category="test_patterns">
      <title>conftest.py provides temp_db and atomic_db fixtures</title>
      <detail>
        conftest.py:17-128 creates temporary SQLite database with:
        - Fe I lines (3 lines at 371-375 nm)
        - H I lines (2 lines - H-alpha, H-beta)
        - Energy levels and ionization potentials

        Used by integration tests requiring real database queries.
      </detail>
      <source>tests/conftest.py:17-128</source>
      <relevance>Can extend for Phase 2c if real database needed</relevance>
    </finding>

    <finding category="test_patterns">
      <title>Integration test shows forward model workflow</title>
      <detail>
        test_integration.py:17-42 demonstrates:
        1. Create plasma (SingleZoneLTEPlasma)
        2. Create instrument (InstrumentModel)
        3. Create SpectrumModel
        4. compute_spectrum() -> wavelength, intensity

        Temperature/density scans at test_integration.py:126-162 show
        parametric testing pattern.
      </detail>
      <source>tests/test_integration.py:17-162</source>
      <relevance>Pattern for round-trip tests: forward model → inversion → compare</relevance>
    </finding>

    <finding category="validation_strategy">
      <title>Round-trip testing is the gold standard</title>
      <detail>
        Literature (Ciucci, Tognoni) and ROADMAP.md (lines 215-220) specify:
        - Generate spectra with known T, n_e, concentrations
        - Add realistic noise (Poisson + Gaussian)
        - Run inversion
        - Verify recovery: T±5%, n_e±20%

        Implementation approach:
        1. Use forward model to generate "golden spectra"
        2. Extract LineObservations from known lines
        3. Run through inversion pipeline
        4. Compare recovered parameters to ground truth
      </detail>
      <source>ROADMAP.md:215-220</source>
      <relevance>Primary validation approach for Phase 2c integration tests</relevance>
    </finding>

    <finding category="validation_strategy">
      <title>Unit test per metric for QualityAssessor</title>
      <detail>
        Test each quality metric independently:
        1. R² test: Create perfect fit → expect R²≈1.0, create scattered → expect low R²
        2. Closure test: Sum concentrations to various values → verify closure_residual
        3. T consistency: Multi-element with same T → low std, different T → high std
        4. Flag determination: Craft inputs to hit each threshold boundary
      </detail>
      <source>Analysis of quality.py</source>
      <relevance>Systematic unit test coverage</relevance>
    </finding>

    <finding category="validation_strategy">
      <title>Optical depth inversion test for self-absorption</title>
      <detail>
        Test strategy:
        1. Create line with known intensity I_true
        2. Apply self-absorption: I_meas = I_true × f(τ)
        3. Run correction
        4. Verify |I_corrected - I_true| / I_true < tolerance

        Test τ values: 0.05 (thin), 0.5 (moderate), 2.0 (thick), 5.0 (masked)
      </detail>
      <source>Analysis of self_absorption.py</source>
      <relevance>Direct verification of correction algorithm</relevance>
    </finding>

    <finding category="validation_strategy">
      <title>Scoring verification for LineSelector</title>
      <detail>
        Test approach:
        1. Create lines with known SNR, isolation, uncertainty
        2. Verify score = SNR × (1/unc) × isolation
        3. Test rejection criteria:
           - Low SNR → rejected
           - Blended (isolation < 0.5) → rejected
           - Resonance (if provided) → rejected
        4. Test energy spread warning
        5. Test max_lines_per_element limit
      </detail>
      <source>Analysis of line_selection.py</source>
      <relevance>Comprehensive unit test for selection logic</relevance>
    </finding>

    <finding category="edge_cases">
      <title>Edge cases to test</title>
      <detail>
        QualityAssessor:
        - Empty observations list
        - Single element (no inter-element T comparison)
        - All concentrations zero
        - Infinite or NaN inputs

        SelfAbsorptionCorrector:
        - τ = 0 (optically thin, no change)
        - τ > mask_threshold (should mask, not correct)
        - Negative intensity (invalid input)
        - Zero concentration (τ should be 0)

        LineSelector:
        - Empty observations
        - All lines rejected (low SNR)
        - Single line per element (below min_lines_per_element)
        - Identical wavelengths (isolation = 0)
        - All energies same (energy_spread = 0)
      </detail>
      <source>Analysis of all Phase 2c modules</source>
      <relevance>Robustness testing</relevance>
    </finding>
  </findings>

  <recommendations>
    <recommendation priority="high">
      <action>Create test_quality.py with unit tests for QualityAssessor</action>
      <rationale>QualityAssessor has clear thresholds and deterministic behavior. Unit tests can verify each metric calculation and flag determination independently.</rationale>
    </recommendation>

    <recommendation priority="high">
      <action>Create test_self_absorption.py with correction inversion tests</action>
      <rationale>Self-absorption correction has a mathematical inverse - apply absorption then correct should recover original intensity. Also test estimate_optical_depth_from_intensity_ratio() standalone.</rationale>
    </recommendation>

    <recommendation priority="high">
      <action>Create test_line_selection.py with scoring and filtering tests</action>
      <rationale>LineSelector scoring is multiplicative and deterministic. Easy to verify with controlled inputs.</rationale>
    </recommendation>

    <recommendation priority="medium">
      <action>Extend conftest.py with Phase 2c fixtures</action>
      <rationale>Create reusable fixtures: synthetic_observations(), quality_input_set(), self_absorption_test_lines(). Avoid duplicating setup code across test files.</rationale>
    </recommendation>

    <recommendation priority="medium">
      <action>Add round-trip integration test</action>
      <rationale>Full pipeline validation: forward model → extract lines → inversion → compare. Use existing SpectrumModel + IterativeCFLIBSSolver.</rationale>
    </recommendation>

    <recommendation priority="low">
      <action>Document identify_resonance_lines() stub behavior</action>
      <rationale>Current implementation returns empty set. Test should verify this and mark as known limitation.</rationale>
    </recommendation>
  </recommendations>

  <code_examples>
    <example name="synthetic_observations_fixture">
```python
def create_synthetic_observations(
    T_K: float = 10000.0,
    elements: dict[str, float] = None,
    n_lines_per_element: int = 5,
    noise_level: float = 0.05,
    seed: int = 42,
) -> list[LineObservation]:
    """Generate synthetic line observations for testing.

    Parameters
    ----------
    T_K : float
        Plasma temperature
    elements : dict[str, float]
        Element -> relative concentration (will be normalized)
    n_lines_per_element : int
        Number of lines per element
    noise_level : float
        Relative noise (sigma/mean)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    List[LineObservation]
    """
    if elements is None:
        elements = {"Fe": 0.5, "Ti": 0.5}

    np.random.seed(seed)
    T_eV = T_K * KB_EV

    obs = []
    for element, rel_conc in elements.items():
        energies = np.linspace(2.0, 6.0, n_lines_per_element)
        intercept = np.log(rel_conc * 1e10)  # Arbitrary scale

        for i, E_k in enumerate(energies):
            y = intercept - E_k / T_eV
            y_noisy = y + np.random.normal(0, noise_level)
            intensity = np.exp(y_noisy)

            obs.append(LineObservation(
                wavelength_nm=400.0 + i * 10 + hash(element) % 100,
                intensity=intensity,
                intensity_uncertainty=intensity * 0.05,
                element=element,
                ionization_stage=1,
                E_k_ev=E_k,
                g_k=2 * (i + 1),
                A_ki=1e7,
            ))

    return obs
```
    </example>

    <example name="quality_assessor_test">
```python
def test_quality_flag_excellent():
    """Test that excellent data gets excellent flag."""
    # Create perfect Boltzmann data (very low noise)
    obs = create_synthetic_observations(T_K=10000, noise_level=0.001)

    assessor = QualityAssessor()
    metrics = assessor.assess(
        observations=obs,
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Fe": 0.5, "Ti": 0.5},
        ionization_potentials={"Fe": 7.87, "Ti": 6.82},
        partition_funcs_I={"Fe": 25.0, "Ti": 30.0},
        partition_funcs_II={"Fe": 15.0, "Ti": 20.0},
    )

    assert metrics.r_squared_boltzmann > 0.95
    assert metrics.closure_residual < 0.01
    assert metrics.quality_flag in ("excellent", "good")
```
    </example>

    <example name="self_absorption_round_trip">
```python
def test_self_absorption_correction_round_trip():
    """Test that correction recovers original intensity."""
    # Create observation with known intensity
    I_true = 1000.0
    tau = 1.0  # Moderate optical depth

    # Apply self-absorption
    f_tau = (1.0 - np.exp(-tau)) / tau
    I_measured = I_true * f_tau

    obs = LineObservation(
        wavelength_nm=500.0,
        intensity=I_measured,
        intensity_uncertainty=I_measured * 0.05,
        element="Fe",
        ionization_stage=1,
        E_k_ev=3.0,
        g_k=10,
        A_ki=1e8,
    )

    corrector = SelfAbsorptionCorrector(
        optical_depth_threshold=0.1,
        mask_threshold=5.0,
    )

    # Note: Correction requires concentration/density info
    # For unit test, mock the optical depth calculation
    result = corrector._apply_recursive_correction(obs, tau)

    # Should recover original intensity within tolerance
    assert abs(result.corrected_intensity - I_true) / I_true < 0.05
```
    </example>

    <example name="line_selection_test">
```python
def test_line_selection_scoring():
    """Test that line scores are calculated correctly."""
    obs = LineObservation(
        wavelength_nm=500.0,
        intensity=1000.0,
        intensity_uncertainty=10.0,  # SNR = 100
        element="Fe",
        ionization_stage=1,
        E_k_ev=3.0,
        g_k=10,
        A_ki=1e8,
    )

    selector = LineSelector(min_snr=10.0)

    # With single line, isolation = 1.0
    # Score = SNR * (1/unc) * isolation = 100 * (1/0.10) * 1.0 = 1000
    result = selector.select([obs])

    assert len(result.scores) == 1
    assert result.scores[0].snr == 100.0
    assert result.scores[0].isolation_factor == 1.0
    # Default atomic_uncertainty is 0.10
    expected_score = 100.0 * (1.0 / 0.10) * 1.0
    assert abs(result.scores[0].score - expected_score) < 0.01
```
    </example>
  </code_examples>

  <metadata>
    <confidence level="high">
      Module analysis based on direct source code reading. Test patterns verified from existing test files. Validation strategies derived from CF-LIBS literature and ROADMAP.md.
    </confidence>
    <dependencies>
      - pytest (already installed)
      - numpy (already installed)
      - No additional dependencies needed
    </dependencies>
    <open_questions>
      - Should round-trip tests use real AtomicDatabase or mocks?
      - What noise levels are realistic for actual LIBS data?
      - Are there standard CF-LIBS benchmark datasets to validate against?
    </open_questions>
    <assumptions>
      - Tests will run on CPU (JAX_PLATFORMS=cpu via pytest.ini)
      - Synthetic data is sufficient for Phase 2c validation (no real spectra needed)
      - Module implementations are correct; tests verify behavior, not physics
    </assumptions>
    <quality_report>
      <sources_consulted>
        - cflibs/inversion/quality.py (read in full)
        - cflibs/inversion/self_absorption.py (read in full)
        - cflibs/inversion/line_selection.py (read in full)
        - cflibs/inversion/boltzmann.py (read in full)
        - tests/conftest.py (read in full)
        - tests/test_boltzmann.py (read in full)
        - tests/test_closure.py (read in full)
        - tests/test_solver.py (read in full)
        - tests/test_integration.py (read in full)
        - ROADMAP.md (read in full)
        - CLAUDE.md (read in full)
      </sources_consulted>
      <claims_verified>
        - QualityMetrics thresholds documented in source (quality.py:23-28, 74-80)
        - Self-absorption formula f(τ) = (1-exp(-τ))/τ in source (self_absorption.py:56-58)
        - LineSelector scoring formula in source (line_selection.py:244-248)
        - Existing create_synthetic_lines() helper in test_boltzmann.py:12-52
      </claims_verified>
      <claims_assumed>
        - Noise levels typical for LIBS (assumed 5% based on test patterns)
        - Round-trip accuracy bounds (T±5%, n_e±20%) from ROADMAP.md
      </claims_assumed>
      <contradictions_encountered>
        None - all sources consistent
      </contradictions_encountered>
      <confidence_by_finding>
        - Module API documentation: High (direct source reading)
        - Test pattern recommendations: High (based on existing tests)
        - Round-trip strategy: Medium (based on ROADMAP, not implemented yet)
        - Edge case coverage: Medium (inferred from code, not validated)
      </confidence_by_finding>
    </quality_report>
  </metadata>
</research>
