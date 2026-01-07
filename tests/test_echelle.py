"""
Tests for echellogram extraction.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from cflibs.instrument.echelle import EchelleExtractor


def test_echelle_extractor_init():
    """Test EchelleExtractor initialization."""
    extractor = EchelleExtractor()
    assert extractor.orders == {}
    assert extractor.extraction_window == 5


def test_mock_calibration():
    """Test mock calibration generation."""
    extractor = EchelleExtractor()
    extractor.create_mock_calibration(width=2048, num_orders=3, wavelength_range=(300.0, 400.0))

    assert len(extractor.orders) == 3
    assert "50" in extractor.orders
    assert "49" in extractor.orders
    assert "48" in extractor.orders

    # Check structure
    for order_name, coeffs in extractor.orders.items():
        assert "y_coeffs" in coeffs
        assert "wl_coeffs" in coeffs


def test_calibration_save_load():
    """Test saving and loading calibration."""
    extractor1 = EchelleExtractor()
    extractor1.create_mock_calibration()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
        extractor1.save_calibration(temp_path)

    try:
        extractor2 = EchelleExtractor(calibration_file=temp_path)
        assert len(extractor2.orders) == len(extractor1.orders)
        assert extractor2.orders == extractor1.orders
    finally:
        Path(temp_path).unlink()


def test_extract_order():
    """Test extracting a single order."""
    extractor = EchelleExtractor()
    extractor.create_mock_calibration(width=2048, num_orders=1)

    # Create mock image with a bright line
    image = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)
    # Add bright spot on trace (order 50, y ~ 500 + 0.1*x at x=1000)
    image[595:605, 995:1005] += 5000

    wl, flux = extractor.extract_order(image, "50")

    assert len(wl) == 2048
    assert len(flux) == 2048
    assert np.all(wl > 0)
    assert np.any(flux > 0)


def test_extract_spectrum():
    """Test full spectrum extraction."""
    extractor = EchelleExtractor()
    extractor.create_mock_calibration(width=2048, num_orders=3)

    # Create mock image
    image = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)

    # Add bright spots on traces
    image[595:605, 995:1005] += 5000  # Order 50
    image[715:725, 995:1005] += 5000  # Order 49

    wl, intensity = extractor.extract_spectrum(image)

    assert len(wl) == len(intensity)
    assert np.all(wl > 0)
    assert np.any(intensity > 0)
    # Should cover wavelength range
    assert wl.min() < 400
    assert wl.max() > 300


def test_extract_spectrum_merge_methods():
    """Test different merge methods."""
    extractor = EchelleExtractor()
    extractor.create_mock_calibration(width=2048, num_orders=2)

    image = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)

    for method in ["weighted_average", "simple_average", "max"]:
        wl, intensity = extractor.extract_spectrum(image, merge_method=method)
        assert len(wl) == len(intensity)
        assert np.all(intensity >= 0)


def test_extract_spectrum_no_calibration():
    """Test that extraction fails without calibration."""
    extractor = EchelleExtractor()
    image = np.random.normal(100, 10, (1024, 2048))

    with pytest.raises(ValueError, match="No orders calibrated"):
        extractor.extract_spectrum(image)


def test_invalid_calibration_file():
    """Test handling of invalid calibration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json {")
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            EchelleExtractor(calibration_file=temp_path)
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
