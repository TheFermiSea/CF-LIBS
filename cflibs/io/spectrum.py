"""
I/O utilities for spectra.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from cflibs.core.logging_config import get_logger

logger = get_logger("io.spectrum")


def load_spectrum(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from file.

    Supports CSV files with columns: wavelength, intensity

    Parameters
    ----------
    file_path : str
        Path to spectrum file

    Returns
    -------
    wavelength : array
        Wavelength array in nm
    intensity : array
        Intensity array
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, comment="#")
        # Try common column names
        wl_col = None
        for col in ["wavelength", "wavelength_nm", "wl", "lambda", "lambda_nm"]:
            if col in df.columns:
                wl_col = col
                break

        if wl_col is None:
            raise ValueError("Could not find wavelength column in CSV")

        int_col = None
        for col in ["intensity", "intensity_W_m2_nm_sr", "I", "counts", "signal"]:
            if col in df.columns:
                int_col = col
                break

        if int_col is None:
            raise ValueError("Could not find intensity column in CSV")

        wavelength = df[wl_col].values
        intensity = df[int_col].values

    else:
        # Try numpy loadtxt as fallback
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            raise ValueError("Spectrum file must have at least 2 columns")
        wavelength = data[:, 0]
        intensity = data[:, 1]

    logger.info(f"Loaded spectrum from {file_path}: {len(wavelength)} points")
    return wavelength, intensity


def save_spectrum(
    file_path: str, wavelength: np.ndarray, intensity: np.ndarray, header: str = None
) -> None:
    """
    Save spectrum to file.

    Parameters
    ----------
    file_path : str
        Output file path
    wavelength : array
        Wavelength array in nm
    intensity : array
        Intensity array
    header : str, optional
        Header comment
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        if header is None:
            header = "wavelength_nm,intensity"

        np.savetxt(
            file_path,
            np.column_stack([wavelength, intensity]),
            delimiter=",",
            header=header,
            comments="",
        )
    else:
        # Default to space-separated
        np.savetxt(file_path, np.column_stack([wavelength, intensity]), header=header)

    logger.info(f"Saved spectrum to {file_path}")
