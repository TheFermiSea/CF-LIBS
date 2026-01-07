"""
Pytest configuration and shared fixtures for CF-LIBS tests.
"""

import pytest
import numpy as np
import sqlite3
import tempfile
from pathlib import Path

from cflibs.atomic.structures import Transition, EnergyLevel
from cflibs.atomic.database import AtomicDatabase
from cflibs.plasma.state import SingleZoneLTEPlasma


@pytest.fixture
def temp_db():
    """Create a temporary atomic database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute(
        """
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    """
    )

    # Insert test data
    # Fe I (neutral iron)
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Fe', 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000),
               ('Fe', 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500),
               ('Fe', 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200)
    """
    )

    conn.execute(
        """
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('Fe', 1, 9, 0.0),
               ('Fe', 1, 11, 3.33),
               ('Fe', 1, 9, 3.32),
               ('Fe', 1, 7, 3.31)
    """
    )

    conn.execute(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('Fe', 1, 7.87),
               ('Fe', 2, 16.18)
    """
    )

    # H I (hydrogen)
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('H', 1, 656.28, 4.4e7, 0.0, 12.75, 2, 8, 10000),
               ('H', 1, 486.13, 8.4e6, 0.0, 12.75, 2, 8, 2000)
    """
    )

    conn.execute(
        """
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('H', 1, 2, 0.0),
               ('H', 1, 8, 12.75)
    """
    )

    conn.execute(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('H', 1, 13.60)
    """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink()


@pytest.fixture
def atomic_db(temp_db):
    """Create AtomicDatabase instance from temp database."""
    return AtomicDatabase(temp_db)


@pytest.fixture
def sample_plasma():
    """Create a sample plasma state for testing."""
    return SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1e17,
        species={
            "Fe": 1e15,
            "H": 1e16,
        },
    )


@pytest.fixture
def sample_transition():
    """Create a sample transition for testing."""
    return Transition(
        element="Fe",
        ionization_stage=1,
        wavelength_nm=371.99,
        A_ki=1.0e7,
        E_k_ev=3.33,
        E_i_ev=0.0,
        g_k=11,
        g_i=9,
        relative_intensity=1000.0,
    )


@pytest.fixture
def sample_energy_level():
    """Create a sample energy level for testing."""
    return EnergyLevel(element="Fe", ionization_stage=1, energy_ev=3.33, g=11)


@pytest.fixture
def sample_wavelength_grid():
    """Create a sample wavelength grid for testing."""
    return np.linspace(200.0, 800.0, 1000)


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "atomic_database": "libs_production.db",
        "plasma": {
            "model": "single_zone_lte",
            "Te": 10000.0,
            "ne": 1.0e17,
            "species": [
                {"element": "Fe", "number_density": 1.0e15},
                {"element": "H", "number_density": 1.0e16},
            ],
        },
        "instrument": {"resolution_fwhm_nm": 0.05},
        "spectrum": {
            "lambda_min_nm": 200.0,
            "lambda_max_nm": 800.0,
            "delta_lambda_nm": 0.01,
            "path_length_m": 0.01,
        },
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create a temporary YAML config file."""
    import yaml

    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")

    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)

    yield config_path

    Path(config_path).unlink()


@pytest.fixture
def mock_echellogram_image():
    """Create a mock 2D echellogram image for testing."""
    image = np.random.normal(100, 10, (1024, 2048)).astype(np.float32)
    # Add some bright spectral lines
    image[595:605, 995:1005] += 5000
    image[715:725, 995:1005] += 5000
    return image
