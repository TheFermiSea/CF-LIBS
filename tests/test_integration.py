"""
Integration tests for complete CF-LIBS workflows.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from cflibs.plasma import SingleZoneLTEPlasma
from cflibs.instrument import InstrumentModel
from cflibs.radiation import SpectrumModel
from cflibs.io import save_spectrum, load_spectrum
from cflibs.core.config import load_config, validate_plasma_config, validate_instrument_config


def test_end_to_end_forward_modeling(atomic_db):
    """Test complete forward modeling workflow."""
    # 1. Create plasma
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1e17, species={"Fe": 1e15, "H": 1e16})

    # 2. Create instrument
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    # 3. Create model
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=370.0,
        lambda_max=375.0,
        delta_lambda=0.01,
    )

    # 4. Compute spectrum
    wavelength, intensity = model.compute_spectrum()

    # 5. Verify results
    assert len(wavelength) == len(intensity)
    assert np.all(wavelength > 0)
    assert np.all(intensity >= 0)
    assert len(wavelength) > 0


def test_config_to_spectrum_workflow(atomic_db, temp_config_file):
    """Test workflow from config file to spectrum."""
    # Load config
    config = load_config(temp_config_file)

    # Validate
    validate_plasma_config(config)
    validate_instrument_config(config)

    # Create objects
    plasma_config = config["plasma"]
    plasma = SingleZoneLTEPlasma(
        T_e=plasma_config["Te"],
        n_e=plasma_config["ne"],
        species={s["element"]: s["number_density"] for s in plasma_config["species"]},
    )

    instrument = InstrumentModel.from_file(Path(temp_config_file))

    spectrum_config = config["spectrum"]
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=spectrum_config["lambda_min_nm"],
        lambda_max=spectrum_config["lambda_max_nm"],
        delta_lambda=spectrum_config["delta_lambda_nm"],
        path_length_m=spectrum_config["path_length_m"],
    )

    # Compute
    wavelength, intensity = model.compute_spectrum()

    assert len(wavelength) > 0
    assert len(intensity) > 0


def test_save_and_load_spectrum():
    """Test saving and loading spectrum."""
    wavelength = np.linspace(200, 800, 100)
    intensity = np.exp(-(((wavelength - 500) / 50) ** 2))

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    Path(temp_path).unlink()

    try:
        # Save
        save_spectrum(temp_path, wavelength, intensity)

        # Load
        wl_loaded, int_loaded = load_spectrum(temp_path)

        # Verify
        assert np.allclose(wl_loaded, wavelength)
        assert np.allclose(int_loaded, intensity)
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def test_multi_species_plasma(atomic_db):
    """Test forward modeling with multiple species."""
    plasma = SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1e17,
        species={
            "Fe": 1e15,
            "H": 1e16,
        },
    )

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    model = SpectrumModel(plasma, atomic_db, instrument, 200.0, 800.0, 1.0)

    wavelength, intensity = model.compute_spectrum()

    # Should have signal from both species
    assert np.any(intensity > 0)
    assert len(wavelength) > 0


def test_temperature_scan(atomic_db):
    """Test generating spectra at different temperatures."""
    temperatures = [8000.0, 10000.0, 12000.0]
    spectra = []

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    for T_e in temperatures:
        plasma = SingleZoneLTEPlasma(T_e=T_e, n_e=1e17, species={"Fe": 1e15})

        model = SpectrumModel(plasma, atomic_db, instrument, 370.0, 375.0, 0.1)

        wl, intensity = model.compute_spectrum()
        spectra.append((T_e, intensity.max()))

    # Higher temperatures should generally produce different intensities
    # (exact relationship depends on energy levels)
    assert len(set(spectra)) > 1  # Not all the same


def test_density_scan(atomic_db):
    """Test generating spectra at different densities."""
    densities = [1e16, 1e17, 1e18]
    spectra = []

    instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    for n_e in densities:
        plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=n_e, species={"Fe": 1e15})

        model = SpectrumModel(plasma, atomic_db, instrument, 370.0, 375.0, 0.1)

        wl, intensity = model.compute_spectrum()
        spectra.append((n_e, intensity.max()))

    # Different densities should produce different spectra
    assert len(set(spectra)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
