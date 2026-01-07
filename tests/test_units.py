"""
Tests for unit conversion utilities.
"""

import pytest
import numpy as np
from cflibs.core import units


def test_temperature_conversion():
    """Test temperature conversions."""
    # K to eV
    T_k = 10000.0
    T_ev = units.convert_temperature(T_k, "K", "eV")
    assert T_ev > 0
    assert T_ev < 1.0  # Should be ~0.86 eV

    # Round trip
    T_k_round = units.convert_temperature(T_ev, "eV", "K")
    assert abs(T_k - T_k_round) < 1e-6

    # Celsius
    T_c = units.convert_temperature(T_k, "K", "C")
    assert abs(T_c - (T_k - 273.15)) < 1e-6


def test_density_conversion():
    """Test density conversions."""
    # cm^-3 to m^-3
    n_cm3 = 1e17
    n_m3 = units.convert_density(n_cm3, "cm^-3", "m^-3")
    assert abs(n_m3 - 1e23) < 1e15

    # Round trip
    n_cm3_round = units.convert_density(n_m3, "m^-3", "cm^-3")
    assert abs(n_cm3 - n_cm3_round) < 1e6


def test_wavelength_conversion():
    """Test wavelength conversions."""
    # nm to m
    wl_nm = 500.0
    wl_m = units.convert_wavelength(wl_nm, "nm", "m")
    assert abs(wl_m - 5e-7) < 1e-9

    # Round trip
    wl_nm_round = units.convert_wavelength(wl_m, "m", "nm")
    assert abs(wl_nm - wl_nm_round) < 1e-6

    # nm to cm^-1 (wavenumber)
    wavenumber = units.convert_wavelength(wl_nm, "nm", "cm^-1")
    assert wavenumber > 0
    # 500 nm should be ~20000 cm^-1
    assert abs(wavenumber - 20000.0) < 100


def test_array_conversion():
    """Test that conversions work with arrays."""
    T_array = np.array([5000, 10000, 15000])
    T_ev_array = units.convert_temperature(T_array, "K", "eV")
    assert len(T_ev_array) == len(T_array)
    assert all(T_ev_array > 0)


if __name__ == "__main__":
    pytest.main([__file__])
