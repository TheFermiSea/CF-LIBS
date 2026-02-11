"""
Tests for ALIAS element identification algorithm.
"""

import pytest
import numpy as np
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.element_id import ElementIdentificationResult


def test_detect_peaks(atomic_db, synthetic_libs_spectrum):
    """Test peak detection with 2nd derivative enhancement."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (400.0, 500.0), (450.0, 200.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db)
    peaks = identifier._detect_peaks(spectrum["wavelength"], spectrum["intensity"])

    # Should detect 3 peaks
    assert len(peaks) > 0
    assert isinstance(peaks, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in peaks)

    # Peaks should be (index, wavelength) tuples
    peak_wavelengths = [p[1] for p in peaks]

    # Should find peaks near expected positions (within 1 nm)
    expected_wls = [371.99, 400.0, 450.0]
    for expected_wl in expected_wls:
        closest = min(peak_wavelengths, key=lambda x: abs(x - expected_wl))
        assert (
            abs(closest - expected_wl) < 1.0
        ), f"Expected peak at {expected_wl}, closest found at {closest}"


def test_compute_element_emissivities(atomic_db):
    """Test emissivity calculation for Fe I lines."""
    identifier = ALIASIdentifier(atomic_db)

    # Compute emissivities for Fe in 370-376 nm range (covers test lines)
    line_data = identifier._compute_element_emissivities("Fe", 370.0, 376.0)

    assert len(line_data) > 0
    for line in line_data:
        assert "transition" in line
        assert "avg_emissivity" in line
        assert "wavelength_nm" in line
        assert line["avg_emissivity"] > 0
        assert 370.0 <= line["wavelength_nm"] <= 376.0


def test_fuse_lines(atomic_db):
    """Test line fusion within resolution element."""
    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Get Fe lines
    line_data = identifier._compute_element_emissivities("Fe", 370.0, 376.0)
    wavelength = np.linspace(370.0, 376.0, 1000)

    # Fuse lines
    fused = identifier._fuse_lines(line_data, wavelength)

    assert len(fused) > 0
    for line in fused:
        assert "transition" in line
        assert "avg_emissivity" in line
        assert "wavelength_nm" in line
        assert "n_fused" in line
        assert line["n_fused"] >= 1


def test_match_lines(atomic_db):
    """Test matching theoretical lines to experimental peaks."""
    identifier = ALIASIdentifier(atomic_db)

    # Create fused lines at specific wavelengths
    from cflibs.atomic.structures import Transition

    trans1 = Transition("Fe", 1, 372.0, 1e7, 3.33, 0.0, 11, 9)
    trans2 = Transition("Fe", 1, 373.5, 5e6, 3.32, 0.0, 9, 9)
    fused_lines = [
        {"transition": trans1, "avg_emissivity": 1000.0, "wavelength_nm": 372.0},
        {"transition": trans2, "avg_emissivity": 500.0, "wavelength_nm": 373.5},
    ]

    # Create peaks near theoretical wavelengths
    peaks = [(100, 372.01), (200, 373.49)]  # (index, wavelength)

    matched_mask, wavelength_shifts, matched_peak_idx = identifier._match_lines(
        fused_lines, peaks
    )

    # Both lines should match
    assert matched_mask[0] == True
    assert matched_mask[1] == True

    # Shifts should be small
    assert abs(wavelength_shifts[0]) < 0.1
    assert abs(wavelength_shifts[1]) < 0.1

    # Peak indices should be valid
    assert matched_peak_idx[0] >= 0
    assert matched_peak_idx[1] >= 0


def test_identify_basic(atomic_db, synthetic_libs_spectrum):
    """Test full identify() with synthetic spectrum containing Fe lines."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    assert result.algorithm == "alias"
    assert result.n_peaks > 0

    # Fe should be in the results (detected or rejected)
    fe_elements = [e for e in result.all_elements if e.element == "Fe"]
    assert len(fe_elements) == 1

    fe_result = fe_elements[0]
    assert fe_result.n_matched_lines > 0


def test_identify_returns_result_type(atomic_db, synthetic_libs_spectrum):
    """Test that identify() returns ElementIdentificationResult."""
    spectrum = synthetic_libs_spectrum()

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    assert hasattr(result, "detected_elements")
    assert hasattr(result, "rejected_elements")
    assert hasattr(result, "all_elements")
    assert hasattr(result, "experimental_peaks")
    assert hasattr(result, "algorithm")
    assert result.algorithm == "alias"


def test_identify_no_elements(atomic_db, synthetic_libs_spectrum):
    """Test identify with no matching elements (edge case)."""
    # Create spectrum with only H line, but search for Cu
    spectrum = synthetic_libs_spectrum(
        elements={"H": [(656.28, 5000.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Cu"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    assert isinstance(result, ElementIdentificationResult)
    # Cu should not be detected (no Cu lines in wavelength range of test DB)
    cu_elements = [e for e in result.detected_elements if e.element == "Cu"]
    assert len(cu_elements) == 0


def test_detect_peaks_low_snr(atomic_db):
    """Peaks at SNR=5-15 should be detected with new threshold."""
    rng = np.random.default_rng(42)
    wavelength = np.linspace(200, 400, 2000)
    noise_level = 10.0
    baseline = 100 + 0.3 * wavelength  # sloped continuum
    noise = rng.normal(0, noise_level, 2000)

    # Add peaks at SNR 5, 8, 12, 15
    peaks_data = [(400, 50), (800, 80), (1200, 120), (1600, 150)]
    signal = np.zeros(2000)
    for loc, height in peaks_data:
        signal[loc - 2 : loc + 3] = height

    intensity = baseline + signal + noise
    identifier = ALIASIdentifier(atomic_db)
    detected = identifier._detect_peaks(wavelength, intensity)

    # Should detect at least the SNR=12 and SNR=15 peaks
    assert len(detected) >= 2, f"Only detected {len(detected)} peaks at SNR 5-15"


def test_noise_only_no_detection(atomic_db):
    """Pure noise should not detect any elements."""
    rng = np.random.default_rng(42)
    wavelength = np.linspace(200, 400, 2000)
    intensity = 100 + rng.normal(0, 10, 2000)

    identifier = ALIASIdentifier(atomic_db, elements=["Fe"])
    result = identifier.identify(wavelength, intensity)
    assert len(result.detected_elements) == 0, (
        f"False detections in noise: {[e.element for e in result.detected_elements]}"
    )


def test_scores_between_zero_and_one(atomic_db, synthetic_libs_spectrum):
    """Test that all scores are in [0, 1] range."""
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0)]},
        noise_level=0.01,
    )

    identifier = ALIASIdentifier(atomic_db, elements=["Fe", "H"])
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])

    for element_id in result.all_elements:
        # Check main scores
        assert 0.0 <= element_id.score <= 1.0
        assert 0.0 <= element_id.confidence <= 1.0

        # Check metadata scores
        metadata = element_id.metadata
        if "k_sim" in metadata:
            assert 0.0 <= metadata["k_sim"] <= 1.0
        if "k_rate" in metadata:
            assert 0.0 <= metadata["k_rate"] <= 1.0
        if "k_shift" in metadata:
            assert 0.0 <= metadata["k_shift"] <= 1.0
        if "k_det" in metadata:
            assert 0.0 <= metadata["k_det"] <= 1.0


def test_max_lines_per_element_parameter(atomic_db):
    """Test that max_lines_per_element caps transition count."""
    identifier = ALIASIdentifier(atomic_db, max_lines_per_element=5)
    assert identifier.max_lines_per_element == 5

    # Default should be 50
    identifier_default = ALIASIdentifier(atomic_db)
    assert identifier_default.max_lines_per_element == 50


def test_default_detection_threshold_lowered(atomic_db):
    """Test that default detection_threshold is 0.02."""
    identifier = ALIASIdentifier(atomic_db)
    assert identifier.detection_threshold == 0.02


# ---------------------------------------------------------------------------
# Tests for the 4 bug-fixes (survivorship bias, uniqueness, P_maj, P_ab)
# ---------------------------------------------------------------------------


def test_k_sim_penalizes_unmatched_lines(atomic_db):
    """Element with many unmatched theoretical lines should get lower k_sim."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # Scenario A: 2 lines, both matched → high k_sim
    fused_a = [
        {"transition": Transition("Fe", 1, 372.0, 1e7, 3.3, 0.0, 11, 9),
         "avg_emissivity": 1000.0, "wavelength_nm": 372.0},
        {"transition": Transition("Fe", 1, 374.0, 5e6, 3.3, 0.0, 9, 9),
         "avg_emissivity": 800.0, "wavelength_nm": 374.0},
    ]
    peaks = [(100, 372.01), (200, 374.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 500.0
    intensity[200] = 400.0
    matched_a = np.array([True, True])
    peak_idx_a = np.array([0, 1])
    shifts_a = np.array([0.01, 0.01])

    k_sim_a, _, _, _, _ = identifier._compute_scores(
        fused_a, matched_a, peak_idx_a, shifts_a, intensity, peaks,
        emissivity_threshold=-np.inf,
    )

    # Scenario B: 10 lines but only 2 matched — identical matched intensities
    fused_b = list(fused_a)  # first two are the same
    for i in range(8):
        wl = 375.0 + i * 0.5
        fused_b.append({
            "transition": Transition("Fe", 1, wl, 1e6, 3.3, 0.0, 7, 7),
            "avg_emissivity": 600.0, "wavelength_nm": wl,
        })
    matched_b = np.array([True, True] + [False] * 8)
    peak_idx_b = np.array([0, 1] + [-1] * 8)
    shifts_b = np.array([0.01, 0.01] + [0.0] * 8)

    k_sim_b, _, _, _, _ = identifier._compute_scores(
        fused_b, matched_b, peak_idx_b, shifts_b, intensity, peaks,
        emissivity_threshold=-np.inf,
    )

    # k_sim_b should be strictly lower because the 8 unmatched lines
    # contribute zeros to the experimental vector
    assert k_sim_b < k_sim_a, (
        f"k_sim should be lower when many lines unmatched: {k_sim_b} vs {k_sim_a}"
    )


def test_uniqueness_penalty(atomic_db):
    """Many theoretical lines mapping to one peak should be penalised."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # 4 theoretical lines all within ~0.1 nm → all map to a single broad peak
    fused = []
    for i in range(4):
        wl = 400.0 + i * 0.02
        fused.append({
            "transition": Transition("Co", 1, wl, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 1000.0, "wavelength_nm": wl,
        })
    # Single experimental peak near 400.0
    peaks = [(250, 400.01)]
    intensity = np.full(500, 10.0)
    intensity[250] = 800.0

    matched = np.array([True, True, True, True])
    peak_idx = np.array([0, 0, 0, 0])  # all map to same peak
    shifts = np.array([0.01, -0.01, 0.01, -0.01])

    k_sim, _, _, _, _ = identifier._compute_scores(
        fused, matched, peak_idx, shifts, intensity, peaks,
        emissivity_threshold=-np.inf,
    )

    # uniqueness_factor = 1 unique peak / 4 matches = 0.25
    # So k_sim should be at most 0.25 (cosine sim capped then scaled)
    assert k_sim <= 0.30, (
        f"Uniqueness penalty should reduce k_sim when many-to-one: got {k_sim}"
    )


def test_P_maj_soft_coverage(atomic_db):
    """P_maj should be high when strongest line matched, lower when not."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db)

    # Scenario: strongest line IS matched → P_maj close to 1.0
    fused = [
        {"transition": Transition("Fe", 1, 372.0, 1e8, 3.3, 0.0, 11, 9),
         "avg_emissivity": 5000.0, "wavelength_nm": 372.0},  # strongest
        {"transition": Transition("Fe", 1, 374.0, 1e6, 3.3, 0.0, 9, 9),
         "avg_emissivity": 100.0, "wavelength_nm": 374.0},
    ]
    peaks = [(100, 372.01), (200, 374.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0
    intensity[200] = 50.0

    matched = np.array([True, True])
    peak_idx = np.array([0, 1])
    shifts = np.array([0.01, 0.01])

    _, _, _, P_maj_both, _ = identifier._compute_scores(
        fused, matched, peak_idx, shifts, intensity, peaks,
        emissivity_threshold=-np.inf,
    )
    # Both matched including strongest → P_maj should be 1.0
    assert P_maj_both > 0.9, f"P_maj should be ~1.0 when all matched: {P_maj_both}"

    # Scenario: strongest line NOT matched → P_maj should be lower
    matched_miss = np.array([False, True])
    peak_idx_miss = np.array([-1, 1])
    _, _, _, P_maj_miss, _ = identifier._compute_scores(
        fused, matched_miss, peak_idx_miss, shifts, intensity, peaks,
        emissivity_threshold=-np.inf,
    )
    assert P_maj_miss < P_maj_both, (
        f"P_maj should decrease when strongest line missed: {P_maj_miss} vs {P_maj_both}"
    )
    assert P_maj_miss >= 0.5, f"P_maj should be at least 0.5: {P_maj_miss}"


def test_N_expected_in_k_det_blend(atomic_db):
    """N_expected (not N_matched) should be used in k_det blend formula."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # 10 theoretical lines above threshold, only 1 matched
    fused = []
    for i in range(10):
        wl = 300.0 + i * 1.0
        fused.append({
            "transition": Transition("Co", 1, wl, 1e7, 3.0, 0.0, 9, 7),
            "avg_emissivity": 1000.0, "wavelength_nm": wl,
        })
    peaks = [(50, 300.01)]
    intensity = np.full(500, 10.0)
    intensity[50] = 800.0

    matched = np.array([True] + [False] * 9)
    peak_idx = np.array([0] + [-1] * 9)
    shifts = np.array([0.01] + [0.0] * 9)

    _, _, _, _, N_expected = identifier._compute_scores(
        fused, matched, peak_idx, shifts, intensity, peaks,
        emissivity_threshold=-np.inf,
    )

    # N_expected should be 10 (all above threshold), not 1 (matched)
    assert N_expected == 10, f"N_expected should be 10 but got {N_expected}"


def test_P_ab_tiers(atomic_db):
    """P_ab should be 1.0 for common, 0.75 for intermediate, 0.5 for rare."""
    identifier = ALIASIdentifier(atomic_db)

    # Common elements (>100 ppm)
    assert identifier._compute_P_ab("Fe") == 1.0
    assert identifier._compute_P_ab("Al") == 1.0
    assert identifier._compute_P_ab("Si") == 1.0
    assert identifier._compute_P_ab("Ca") == 1.0

    # Intermediate (0.001 - 100 ppm)
    assert identifier._compute_P_ab("Co") == 0.75  # 10^1.40 ≈ 25 ppm
    assert identifier._compute_P_ab("Cu") == 0.75  # 10^1.78 ≈ 60 ppm
    assert identifier._compute_P_ab("Sn") == 0.75  # 10^0.35 ≈ 2.2 ppm

    # Ag: 10^-0.62 ≈ 0.24 ppm → still intermediate (>= 0.001)
    assert identifier._compute_P_ab("Ag") == 0.75
    # Au: 10^-2.40 ≈ 0.004 ppm → still intermediate (>= 0.001)
    assert identifier._compute_P_ab("Au") == 0.75

    # Unknown element defaults to log_ppm=0.0 → 1 ppm → intermediate
    assert identifier._compute_P_ab("Xx") == 0.75


def test_N_penalty_sparse_evidence(atomic_db):
    """N_penalty should penalize elements with few above-threshold lines."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    def _make_scenario(n_lines, n_matched):
        """Build fused lines, peaks, and masks for a given line count."""
        fused = []
        for i in range(n_lines):
            wl = 300.0 + i * 2.0  # well-separated
            fused.append({
                "transition": Transition("Fe", 1, wl, 1e7, 3.0, 0.0, 9, 7),
                "avg_emissivity": 1000.0,
                "wavelength_nm": wl,
            })
        peaks_list = [
            (50 + i * 10, 300.0 + i * 2.0 + 0.01) for i in range(n_matched)
        ]
        intensity = np.full(500, 10.0)
        for idx, _ in peaks_list:
            intensity[idx] = 800.0
        matched = np.array(
            [True] * n_matched + [False] * (n_lines - n_matched)
        )
        pidx = np.array(
            list(range(n_matched)) + [-1] * (n_lines - n_matched)
        )
        shifts = np.array(
            [0.01] * n_matched + [0.0] * (n_lines - n_matched)
        )
        return fused, peaks_list, intensity, matched, pidx, shifts

    # N_expected=1, 1 matched → N_penalty=0.2
    f1, p1, i1, m1, pi1, s1 = _make_scenario(1, 1)
    k_sim1, k_rate1, k_shift1, P_maj1, N1 = identifier._compute_scores(
        f1, m1, pi1, s1, i1, p1, emissivity_threshold=-np.inf,
    )
    _, CL1 = identifier._decide(
        k_sim1, k_rate1, k_shift1, N1, i1, p1,
        element="Co", P_maj=P_maj1,
    )

    # N_expected=5, 5 matched → N_penalty=1.0
    f5, p5, i5, m5, pi5, s5 = _make_scenario(5, 5)
    k_sim5, k_rate5, k_shift5, P_maj5, N5 = identifier._compute_scores(
        f5, m5, pi5, s5, i5, p5, emissivity_threshold=-np.inf,
    )
    _, CL5 = identifier._decide(
        k_sim5, k_rate5, k_shift5, N5, i5, p5,
        element="Co", P_maj=P_maj5,
    )

    assert N1 == 1
    assert N5 == 5
    # CL for N=1 should be substantially lower than N=5 due to N_penalty
    assert CL1 < CL5, (
        f"N_penalty should reduce CL for N=1: CL1={CL1}, CL5={CL5}"
    )


def test_combined_k_rate(atomic_db):
    """k_rate should use geometric mean of emissivity-weighted and count-based."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=5000.0)

    # 5 lines, only 1 matched — but it has highest emissivity
    fused = [
        {"transition": Transition("Ni", 1, 350.0, 1e8, 3.0, 0.0, 11, 9),
         "avg_emissivity": 5000.0, "wavelength_nm": 350.0},  # matched
    ]
    for i in range(4):
        wl = 352.0 + i * 2.0
        fused.append({
            "transition": Transition("Ni", 1, wl, 1e6, 3.0, 0.0, 7, 7),
            "avg_emissivity": 100.0, "wavelength_nm": wl,
        })
    peaks = [(100, 350.01)]
    intensity = np.full(500, 10.0)
    intensity[100] = 1000.0

    matched = np.array([True] + [False] * 4)
    pidx = np.array([0] + [-1] * 4)
    shifts = np.array([0.01] + [0.0] * 4)

    _, k_rate, _, _, _ = identifier._compute_scores(
        fused, matched, pidx, shifts, intensity, peaks,
        emissivity_threshold=-np.inf,
    )

    # Pure emissivity-weighted would give ~5000/5400 ≈ 0.93
    # Count-based gives 1/5 = 0.2
    # Geometric mean ≈ sqrt(0.93 * 0.2) ≈ 0.43
    assert k_rate < 0.6, (
        f"Combined k_rate should be pulled down by count rate: got {k_rate}"
    )
    assert k_rate > 0.1, (
        f"Combined k_rate should still be positive: got {k_rate}"
    )


def test_one_to_one_peak_assignment(atomic_db):
    """Each peak should match at most one theoretical line (highest emissivity wins)."""
    from cflibs.atomic.structures import Transition

    identifier = ALIASIdentifier(atomic_db, resolving_power=500.0)

    # 3 lines near the same wavelength — all would match the same peak
    # without one-to-one enforcement
    fused = [
        {"transition": Transition("Co", 1, 400.0, 1e7, 3.0, 0.0, 9, 7),
         "avg_emissivity": 500.0, "wavelength_nm": 400.0},
        {"transition": Transition("Co", 1, 400.3, 1e7, 3.0, 0.0, 9, 7),
         "avg_emissivity": 1000.0, "wavelength_nm": 400.3},  # highest emissivity
        {"transition": Transition("Co", 1, 400.5, 1e7, 3.0, 0.0, 9, 7),
         "avg_emissivity": 200.0, "wavelength_nm": 400.5},
    ]
    # Single peak that's within delta_lambda of all three lines
    peaks = [(250, 400.2)]

    matched, _, peak_idx = identifier._match_lines(fused, peaks)

    # Only one line should be matched (highest emissivity = line at 400.3)
    assert int(np.sum(matched)) == 1, (
        f"One-to-one should allow only 1 match per peak, got {int(np.sum(matched))}"
    )
    # The matched line should be index 1 (highest emissivity)
    assert matched[1] is True or matched[1] == True, (
        "Highest emissivity line should win the peak assignment"
    )
