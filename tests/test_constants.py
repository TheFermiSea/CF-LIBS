"""
Tests for physical constants module.
"""

import pytest
from cflibs.core import constants


def test_constants_exist():
    """Test that all expected constants are defined."""
    assert hasattr(constants, "KB")
    assert hasattr(constants, "KB_EV")
    assert hasattr(constants, "H_PLANCK")
    assert hasattr(constants, "C_LIGHT")
    assert hasattr(constants, "M_E")
    assert hasattr(constants, "E_CHARGE")


def test_constant_values():
    """Test that constants have reasonable values."""
    # Boltzmann constant
    assert constants.KB > 0
    assert constants.KB_EV > 0

    # Planck constant
    assert constants.H_PLANCK > 0

    # Speed of light
    assert constants.C_LIGHT > 0
    assert abs(constants.C_LIGHT - 3e8) < 1e6  # ~3e8 m/s

    # Electron mass
    assert constants.M_E > 0
    assert constants.M_E < 1e-30  # Should be very small


def test_conversion_factors():
    """Test conversion factor relationships."""
    # Energy conversions
    assert abs(constants.EV_TO_J * constants.J_TO_EV - 1.0) < 1e-10

    # Temperature conversions
    assert abs(constants.K_TO_EV * constants.EV_TO_K - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
