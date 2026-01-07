"""
Example: Extracting 1D spectrum from 2D echellogram.

This example demonstrates how to use the EchelleExtractor class to convert
a 2D echellogram image into a 1D spectrum.
"""

import numpy as np
from pathlib import Path

from cflibs.instrument.echelle import EchelleExtractor
from cflibs.io import save_spectrum

# Example 1: Using a calibration file
def example_with_calibration():
    """Extract spectrum using a calibration file."""
    # Load calibration
    calibration_file = Path(__file__).parent / "calibration_example.json"
    extractor = EchelleExtractor(calibration_file=str(calibration_file))
    
    # Load 2D image (in practice, this would come from your camera/FITS file)
    # For this example, create a mock image
    image_2d = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)
    
    # Add some bright spectral lines
    # Order 50 trace at x=1000: y ≈ 500 + 0.1*1000 = 600
    image_2d[595:605, 995:1005] += 5000
    
    # Extract 1D spectrum
    wavelength, intensity = extractor.extract_spectrum(
        image_2d,
        wavelength_step_nm=0.05,
        merge_method='weighted_average'
    )
    
    # Save result
    output_file = Path(__file__).parent / "extracted_spectrum.csv"
    save_spectrum(str(output_file), wavelength, intensity)
    
    print(f"Extracted spectrum: {len(wavelength)} points")
    print(f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
    print(f"Saved to: {output_file}")


# Example 2: Using mock calibration for testing
def example_with_mock_calibration():
    """Extract spectrum using mock calibration (for testing)."""
    # Create extractor with mock calibration
    extractor = EchelleExtractor()
    extractor.create_mock_calibration(
        width=2048,
        num_orders=3,
        wavelength_range=(300.0, 400.0)
    )
    
    # Create mock image with spectral features
    image_2d = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)
    
    # Add bright lines on different orders
    image_2d[595:605, 995:1005] += 5000  # Order 50
    image_2d[715:725, 995:1005] += 5000  # Order 49
    
    # Extract with custom options
    wavelength, intensity = extractor.extract_spectrum(
        image_2d,
        wavelength_step_nm=0.01,  # Higher resolution
        merge_method='max',  # Use maximum in overlaps
        min_valid_pixels=20
    )
    
    print(f"Extracted {len(wavelength)} points using mock calibration")
    print(f"Peak intensity: {intensity.max():.1f}")


# Example 3: Extracting individual orders
def example_extract_single_order():
    """Extract a single order for inspection."""
    extractor = EchelleExtractor()
    extractor.create_mock_calibration()
    
    image_2d = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)
    
    # Extract just order 50
    wl_order, flux_order = extractor.extract_order(
        image_2d,
        order_name='50',
        background_subtract=True
    )
    
    print(f"Order 50: {len(wl_order)} points")
    print(f"Wavelength range: {wl_order.min():.1f} - {wl_order.max():.1f} nm")


if __name__ == "__main__":
    print("Example 1: Using calibration file")
    print("-" * 50)
    example_with_calibration()
    print()
    
    print("Example 2: Using mock calibration")
    print("-" * 50)
    example_with_mock_calibration()
    print()
    
    print("Example 3: Extracting single order")
    print("-" * 50)
    example_extract_single_order()

