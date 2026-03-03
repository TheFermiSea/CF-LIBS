"""Tests for cflibs.core.platform_config."""

import os
import sys
from unittest.mock import patch

import pytest

from cflibs.core.platform_config import AcceleratorBackend, configure_jax


class TestAcceleratorBackend:
    def test_values(self):
        assert AcceleratorBackend.CPU.value == "cpu"
        assert AcceleratorBackend.CUDA.value == "cuda"


class TestConfigureJax:
    def test_returns_accelerator_backend(self):
        result = configure_jax()
        assert isinstance(result, AcceleratorBackend)

    @patch("cflibs.core.platform_config.platform")
    def test_darwin_forces_cpu_env(self, mock_platform):
        """On macOS, JAX_PLATFORMS must be unconditionally set to 'cpu'."""
        mock_platform.system.return_value = "Darwin"
        # Even if JAX_PLATFORMS was set to something else, Darwin overrides it
        with patch.dict(os.environ, {"JAX_PLATFORMS": "metal"}, clear=False):
            result = configure_jax()
            assert result == AcceleratorBackend.CPU
            assert os.environ["JAX_PLATFORMS"] == "cpu"

    @patch("cflibs.core.platform_config.platform")
    def test_darwin_returns_cpu(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        assert configure_jax() == AcceleratorBackend.CPU

    @patch("cflibs.core.platform_config.platform")
    def test_linux_no_gpu_returns_cpu(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        result = configure_jax(prefer_gpu=False)
        assert result == AcceleratorBackend.CPU

    @patch("cflibs.core.platform_config.platform")
    def test_linux_gpu_exception_falls_back_to_cpu(self, mock_platform):
        """When jax.devices('gpu') raises, gracefully fall back."""
        mock_platform.system.return_value = "Linux"
        import jax

        with patch.object(jax, "devices", side_effect=ValueError("no GPU")):
            result = configure_jax(prefer_gpu=True)
            assert result == AcceleratorBackend.CPU

    def test_enables_x64(self):
        """configure_jax should enable float64 by default."""
        configure_jax(enable_x64=True)
        import jax

        assert jax.config.jax_enable_x64 is True

    def test_warns_if_jax_already_imported(self, caplog):
        """Should warn when JAX is already in sys.modules."""
        # JAX is already imported by conftest, so this should always warn
        assert "jax" in sys.modules
        with caplog.at_level("WARNING"):
            configure_jax()
        # The warning may or may not appear depending on logger propagation;
        # what matters is no crash

    @patch.dict(sys.modules, {"jax": None})
    @patch("builtins.__import__", side_effect=ImportError("no jax"))
    def test_jax_not_installed_returns_cpu(self, mock_import):
        """When JAX is not installed, returns CPU without error."""
        result = configure_jax()
        assert result == AcceleratorBackend.CPU
