"""
Tests for configuration management module.
"""

import pytest
import tempfile
from pathlib import Path
import json

from cflibs.core.config import (
    load_config,
    save_config,
    validate_plasma_config,
    validate_instrument_config,
)


def test_load_config_yaml(temp_config_file):
    """Test loading YAML configuration."""
    config = load_config(temp_config_file)
    assert "plasma" in config
    assert "instrument" in config
    assert config["plasma"]["Te"] == 10000.0


def test_load_config_json(sample_config_dict):
    """Test loading JSON configuration."""
    config_fd, config_path = tempfile.mkstemp(suffix=".json")

    try:
        with open(config_path, "w") as f:
            json.dump(sample_config_dict, f)

        config = load_config(config_path)
        assert "plasma" in config
        assert config["plasma"]["Te"] == 10000.0
    finally:
        Path(config_path).unlink()


def test_load_config_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_load_config_invalid_format():
    """Test loading invalid file format."""
    config_fd, config_path = tempfile.mkstemp(suffix=".txt")

    try:
        with open(config_path, "w") as f:
            f.write("not yaml or json")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_save_config_yaml(sample_config_dict):
    """Test saving YAML configuration."""
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    Path(config_path).unlink()  # Remove temp file

    try:
        save_config(sample_config_dict, config_path)
        assert Path(config_path).exists()

        # Verify it can be loaded back
        loaded = load_config(config_path)
        assert loaded["plasma"]["Te"] == sample_config_dict["plasma"]["Te"]
    finally:
        if Path(config_path).exists():
            Path(config_path).unlink()


def test_save_config_json(sample_config_dict):
    """Test saving JSON configuration."""
    config_fd, config_path = tempfile.mkstemp(suffix=".json")
    Path(config_path).unlink()

    try:
        save_config(sample_config_dict, config_path)
        assert Path(config_path).exists()

        loaded = load_config(config_path)
        assert loaded["plasma"]["Te"] == sample_config_dict["plasma"]["Te"]
    finally:
        if Path(config_path).exists():
            Path(config_path).unlink()


def test_validate_plasma_config_valid(sample_config_dict):
    """Test validating valid plasma configuration."""
    assert validate_plasma_config(sample_config_dict) is True


def test_validate_plasma_config_missing_section():
    """Test validating config without plasma section."""
    config = {"instrument": {}}
    with pytest.raises(ValueError, match="must contain 'plasma' section"):
        validate_plasma_config(config)


def test_validate_plasma_config_missing_fields():
    """Test validating config with missing required fields."""
    config = {"plasma": {"model": "single_zone_lte"}}
    with pytest.raises(ValueError, match="missing required field"):
        validate_plasma_config(config)


def test_validate_plasma_config_invalid_model():
    """Test validating config with invalid model."""
    config = {"plasma": {"model": "invalid_model", "Te": 10000.0, "ne": 1e17}}
    with pytest.raises(ValueError, match="Invalid plasma model"):
        validate_plasma_config(config)


def test_validate_instrument_config_valid(sample_config_dict):
    """Test validating valid instrument configuration."""
    assert validate_instrument_config(sample_config_dict) is True


def test_validate_instrument_config_missing_section():
    """Test validating config without instrument section."""
    config = {"plasma": {}}
    with pytest.raises(ValueError, match="must contain 'instrument' section"):
        validate_instrument_config(config)


def test_validate_instrument_config_missing_resolution():
    """Test validating config with missing resolution."""
    config = {"instrument": {}}
    with pytest.raises(ValueError, match="missing 'resolution_fwhm_nm'"):
        validate_instrument_config(config)


def test_validate_instrument_config_invalid_resolution():
    """Test validating config with invalid resolution."""
    config = {"instrument": {"resolution_fwhm_nm": -0.05}}
    with pytest.raises(ValueError, match="must be positive"):
        validate_instrument_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
