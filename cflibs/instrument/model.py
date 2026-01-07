"""
Instrument model for spectrometer response.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from pathlib import Path

from cflibs.core.logging_config import get_logger

logger = get_logger("instrument.model")


@dataclass
class InstrumentModel:
    """
    Model for spectrometer instrument response.

    Attributes
    ----------
    resolution_fwhm_nm : float
        Instrument resolution (FWHM) in nm
    response_curve : Optional[np.ndarray]
        Spectral response curve (wavelength, response) pairs
    wavelength_calibration : Optional[callable]
        Function to convert pixel to wavelength
    """

    resolution_fwhm_nm: float
    response_curve: Optional[np.ndarray] = None
    wavelength_calibration: Optional[callable] = None

    @property
    def resolution_sigma_nm(self) -> float:
        """Gaussian standard deviation for instrument function."""
        return self.resolution_fwhm_nm / 2.355

    @classmethod
    def from_file(cls, config_path: Path) -> "InstrumentModel":
        """
        Load instrument model from configuration file.

        Parameters
        ----------
        config_path : Path
            Path to configuration file

        Returns
        -------
        InstrumentModel
            Instrument model instance
        """
        from cflibs.core.config import load_config

        config = load_config(config_path)

        if "instrument" not in config:
            raise ValueError("Configuration must contain 'instrument' section")

        instr_config = config["instrument"]

        resolution = instr_config.get("resolution_fwhm_nm")
        if resolution is None:
            raise ValueError("Instrument config must specify 'resolution_fwhm_nm'")

        response_file = instr_config.get("response_curve")
        response_curve = None
        if response_file:
            # Load response curve from file
            response_path = Path(response_file)
            if response_path.exists():
                data = np.loadtxt(response_path, delimiter=",")
                response_curve = data
            else:
                logger.warning(f"Response curve file not found: {response_file}")

        return cls(resolution_fwhm_nm=resolution, response_curve=response_curve)

    def apply_response(self, wavelength: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """
        Apply spectral response curve.

        Parameters
        ----------
        wavelength : array
            Wavelength grid in nm
        intensity : array
            Intensity spectrum

        Returns
        -------
        array
            Intensity with response applied
        """
        if self.response_curve is None:
            return intensity

        # Interpolate response curve onto wavelength grid
        from scipy.interpolate import interp1d

        wl_resp = self.response_curve[:, 0]
        resp = self.response_curve[:, 1]

        # Normalize response to max = 1
        resp = resp / resp.max()

        f = interp1d(wl_resp, resp, kind="linear", bounds_error=False, fill_value=0.0)
        response = f(wavelength)

        return intensity * response
