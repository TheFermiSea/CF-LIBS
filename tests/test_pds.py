"""
Tests for PDS (Planetary Data System) ingestion module.

Covers:
- Corpus definition (9t6.1): entries, compositions, metadata
- Cache layer (9t6.4): caching, paths, idempotency
- ChemCam parser (9t6.5): CSV parsing, spectrometer detection
- SuperCam parser (9t6.6): CSV parsing, sol extraction
- Validation schema (9t6.3): mapping, provenance, ground truth quality
- Regression (9t6.2): fixture-based parsing stability
"""

from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pds"
CHEMCAM_FIXTURE = FIXTURE_DIR / "chemcam" / "CL5_test_fixture.csv"
SUPERCAM_FIXTURE = FIXTURE_DIR / "supercam" / "SC3_test_fixture.csv"


# ============================================================================
# Corpus definition tests (CF-LIBS-9t6.1)
# ============================================================================


class TestPDSCorpus:
    """Tests for the curated PDS evaluation corpus."""

    def test_corpus_has_entries(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        assert len(corpus.entries) >= 7

    def test_corpus_has_chemcam_entries(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        cc = corpus.chemcam_entries()
        assert len(cc) >= 5, "Need at least 5 ChemCam calibration entries"

    def test_corpus_has_supercam_entries(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        sc = corpus.supercam_entries()
        assert len(sc) >= 2, "Need at least 2 SuperCam calibration entries"

    def test_all_entries_have_expected_elements(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        for entry in corpus.entries:
            assert entry.expected_elements, f"{entry.entry_id} has no expected elements"

    def test_calibration_targets_have_quantified_elements(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        for entry in corpus.calibration_entries():
            n_quantified = sum(1 for v in entry.expected_elements.values() if v is not None)
            assert (
                n_quantified >= 3
            ), f"{entry.entry_id} has only {n_quantified} quantified elements"

    def test_entry_compositions_are_physically_valid(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        for entry in corpus.entries:
            for el, frac in entry.expected_elements.items():
                if frac is not None:
                    assert (
                        0.0 <= frac <= 1.0
                    ), f"{entry.entry_id}: {el} fraction {frac} out of range"

    def test_entry_lookup_by_id(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        entry = corpus.get_entry("chemcam_ccct3_sol69")
        assert entry is not None
        assert entry.target_name == "CCCT3"

    def test_entry_lookup_missing_returns_none(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        assert corpus.get_entry("nonexistent") is None

    def test_corpus_summary(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        summary = corpus.summary()
        assert "ChemCam" in summary
        assert "SuperCam" in summary
        assert "Calibration" in summary

    def test_ccct_compositions_include_major_elements(self):
        """CCCT targets should include Si, Al, Fe, Ca, Mg (major rock-forming elements)."""
        from cflibs.pds.corpus import _CCCT_COMPOSITIONS

        for target, comp in _CCCT_COMPOSITIONS.items():
            if target == "CCCT9":
                # Ti alloy, not a rock
                assert "Ti" in comp
                continue
            # Rock targets should have Si
            assert "Si" in comp, f"{target} missing Si"

    def test_wavelength_ranges_are_valid(self):
        from cflibs.pds.corpus import PDSCorpus

        corpus = PDSCorpus()
        for entry in corpus.entries:
            assert len(entry.wavelength_ranges_nm) == 3
            for lo, hi in entry.wavelength_ranges_nm:
                assert lo < hi
                assert 200 <= lo <= 600
                assert 300 <= hi <= 1000


# ============================================================================
# Cache layer tests (CF-LIBS-9t6.4)
# ============================================================================


class TestPDSCache:
    """Tests for the PDS download and cache layer."""

    def test_cache_creation(self, tmp_path):
        from cflibs.pds.cache import PDSCache

        cache = PDSCache(cache_dir=tmp_path / "pds_cache")
        assert cache.cache_dir.exists()

    def test_cached_path_is_deterministic(self, tmp_path):
        from cflibs.pds.cache import PDSCache
        from cflibs.pds.corpus import PDSCorpus

        cache = PDSCache(cache_dir=tmp_path / "pds_cache")
        corpus = PDSCorpus()
        entry = corpus.entries[0]

        p1 = cache.cached_path(entry)
        p2 = cache.cached_path(entry)
        assert p1 == p2

    def test_is_cached_returns_false_for_missing(self, tmp_path):
        from cflibs.pds.cache import PDSCache
        from cflibs.pds.corpus import PDSCorpus

        cache = PDSCache(cache_dir=tmp_path / "pds_cache")
        corpus = PDSCorpus()
        assert not cache.is_cached(corpus.entries[0])

    def test_is_cached_returns_true_after_write(self, tmp_path):
        from cflibs.pds.cache import PDSCache
        from cflibs.pds.corpus import PDSCorpus

        cache = PDSCache(cache_dir=tmp_path / "pds_cache")
        entry = PDSCorpus().entries[0]

        # Simulate a cached file
        path = cache.cached_path(entry)
        path.write_text("test data")
        assert cache.is_cached(entry)

    def test_clear_specific_entry(self, tmp_path):
        from cflibs.pds.cache import PDSCache
        from cflibs.pds.corpus import PDSCorpus

        cache = PDSCache(cache_dir=tmp_path / "pds_cache")
        entry = PDSCorpus().entries[0]

        path = cache.cached_path(entry)
        path.write_text("test data")
        assert cache.is_cached(entry)

        cache.clear(entry)
        assert not cache.is_cached(entry)

    def test_cache_status(self, tmp_path):
        from cflibs.pds.cache import PDSCache

        cache = PDSCache(cache_dir=tmp_path / "pds_cache")
        status = cache.status()
        assert "n_files" in status
        assert "total_bytes" in status


# ============================================================================
# ChemCam parser tests (CF-LIBS-9t6.5)
# ============================================================================


class TestChemCamParser:
    """Tests for ChemCam CL5 PDS data parser."""

    @pytest.fixture
    def parser(self):
        from cflibs.pds.chemcam import ChemCamParser

        return ChemCamParser()

    def test_parse_fixture(self, parser):
        """Parser should load the test fixture without error."""
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        assert spectrum.wavelength is not None
        assert spectrum.intensity is not None
        assert len(spectrum.wavelength) == len(spectrum.intensity)

    def test_fixture_wavelength_range(self, parser):
        """Parsed spectrum should cover the expected ChemCam range."""
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        assert spectrum.wavelength[0] >= 230.0
        assert spectrum.wavelength[-1] <= 920.0

    def test_fixture_detects_three_spectrometers(self, parser):
        """Parser should detect the UV/VIO/VNIR gap boundaries."""
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        assert len(spectrum.spectrometer_ranges) == 3

    def test_fixture_extracts_metadata(self, parser):
        """Parser should extract sol and target from header."""
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        assert spectrum.sol == 69 or "sol" in spectrum.metadata

    def test_fixture_intensity_is_positive(self, parser):
        """Calibrated intensity should be mostly positive."""
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        frac_positive = np.mean(spectrum.intensity > 0)
        assert frac_positive > 0.95

    def test_parse_missing_file_raises(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.csv")

    def test_spectrometer_ranges_cover_all_points(self, parser):
        """Spectrometer ranges should account for all data points."""
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        total_points = sum(end - start for start, end in spectrum.spectrometer_ranges)
        assert total_points == len(spectrum.wavelength)


# ============================================================================
# SuperCam parser tests (CF-LIBS-9t6.6)
# ============================================================================


class TestSuperCamParser:
    """Tests for SuperCam PDS data parser."""

    @pytest.fixture
    def parser(self):
        from cflibs.pds.supercam import SuperCamParser

        return SuperCamParser()

    def test_parse_fixture(self, parser):
        spectrum = parser.parse(SUPERCAM_FIXTURE)
        assert spectrum.wavelength is not None
        assert len(spectrum.wavelength) > 0

    def test_fixture_wavelength_range(self, parser):
        spectrum = parser.parse(SUPERCAM_FIXTURE)
        assert spectrum.wavelength[0] >= 240.0
        assert spectrum.wavelength[-1] <= 860.0

    def test_fixture_detects_three_spectrometers(self, parser):
        spectrum = parser.parse(SUPERCAM_FIXTURE)
        assert len(spectrum.spectrometer_ranges) == 3

    def test_fixture_extracts_sol(self, parser):
        spectrum = parser.parse(SUPERCAM_FIXTURE)
        assert spectrum.sol == 82 or "sol" in spectrum.metadata

    def test_parse_missing_file_raises(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.csv")


# ============================================================================
# Validation schema tests (CF-LIBS-9t6.3)
# ============================================================================


class TestPDSValidation:
    """Tests for the PDS → validation schema mapping."""

    def test_chemcam_mapping(self):
        from cflibs.pds.chemcam import ChemCamParser
        from cflibs.pds.corpus import PDSCorpus
        from cflibs.pds.validation import map_chemcam_to_validation

        parser = ChemCamParser()
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        entry = PDSCorpus().get_entry("chemcam_ccct3_sol69")
        assert entry is not None

        dataset = map_chemcam_to_validation(spectrum, entry)

        assert dataset.instrument == "chemcam"
        assert dataset.target_name == "CCCT3"
        assert dataset.entry_id == "chemcam_ccct3_sol69"
        assert dataset.ground_truth_quality == "quantified"
        assert dataset.n_quantified_elements >= 3

    def test_supercam_mapping(self):
        from cflibs.pds.supercam import SuperCamParser
        from cflibs.pds.corpus import PDSCorpus
        from cflibs.pds.validation import map_supercam_to_validation

        parser = SuperCamParser()
        spectrum = parser.parse(SUPERCAM_FIXTURE)
        entry = PDSCorpus().get_entry("supercam_scct5_sol82")
        assert entry is not None

        dataset = map_supercam_to_validation(spectrum, entry)

        assert dataset.instrument == "supercam"
        assert dataset.target_name == "SCCT5"
        assert dataset.ground_truth_quality == "quantified"

    def test_provenance_tracked(self):
        from cflibs.pds.chemcam import ChemCamParser
        from cflibs.pds.corpus import PDSCorpus
        from cflibs.pds.validation import map_chemcam_to_validation

        parser = ChemCamParser()
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        entry = PDSCorpus().get_entry("chemcam_ccct3_sol69")

        dataset = map_chemcam_to_validation(spectrum, entry)

        assert "product_id" in dataset.provenance
        assert "instrument" in dataset.provenance

    def test_quantified_elements_property(self):
        from cflibs.pds.chemcam import ChemCamParser
        from cflibs.pds.corpus import PDSCorpus
        from cflibs.pds.validation import map_chemcam_to_validation

        parser = ChemCamParser()
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        entry = PDSCorpus().get_entry("chemcam_ccct3_sol69")

        dataset = map_chemcam_to_validation(spectrum, entry)

        # quantified_elements should exclude None-valued entries
        quant = dataset.quantified_elements
        assert all(v is not None for v in quant.values())
        assert all(0 <= v <= 1 for v in quant.values())

    def test_wavelength_range_property(self):
        from cflibs.pds.chemcam import ChemCamParser
        from cflibs.pds.corpus import PDSCorpus
        from cflibs.pds.validation import map_chemcam_to_validation

        parser = ChemCamParser()
        spectrum = parser.parse(CHEMCAM_FIXTURE)
        entry = PDSCorpus().get_entry("chemcam_ccct3_sol69")

        dataset = map_chemcam_to_validation(spectrum, entry)

        lo, hi = dataset.wavelength_range
        assert lo < hi
        assert lo >= 230
        assert hi <= 920
