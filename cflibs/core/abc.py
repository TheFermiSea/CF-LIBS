"""
Abstract base classes for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

from cflibs.atomic.structures import Transition, EnergyLevel

if TYPE_CHECKING:
    from cflibs.plasma.state import SingleZoneLTEPlasma


class AtomicDataSource(ABC):
    """
    Abstract interface for atomic data sources.

    This allows plugging in different data sources (SQLite, NIST API, HDF5, etc.)
    without changing the rest of the codebase.
    """

    @abstractmethod
    def get_transitions(
        self,
        element: str,
        ionization_stage: Optional[int] = None,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        min_relative_intensity: Optional[float] = None,
    ) -> List[Transition]:
        """
        Get transitions for an element.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int, optional
            Filter by ionization stage
        wavelength_min : float, optional
            Minimum wavelength in nm
        wavelength_max : float, optional
            Maximum wavelength in nm
        min_relative_intensity : float, optional
            Minimum relative intensity threshold

        Returns
        -------
        List[Transition]
            List of transitions
        """
        pass

    @abstractmethod
    def get_energy_levels(self, element: str, ionization_stage: int) -> List[EnergyLevel]:
        """
        Get energy levels for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage

        Returns
        -------
        List[EnergyLevel]
            List of energy levels
        """
        pass

    @abstractmethod
    def get_ionization_potential(self, element: str, ionization_stage: int) -> Optional[float]:
        """
        Get ionization potential for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage

        Returns
        -------
        float or None
            Ionization potential in eV
        """
        pass

    @abstractmethod
    def get_available_elements(self) -> List[str]:
        """
        Get list of available elements.

        Returns
        -------
        List[str]
            List of element symbols
        """
        pass


class SolverStrategy(ABC):
    """
    Abstract interface for plasma solvers.

    This allows implementing different solver algorithms (Saha-Boltzmann,
    non-LTE solvers, multi-zone solvers, etc.) as interchangeable strategies.
    """

    @abstractmethod
    def solve_ionization_balance(
        self, element: str, T_e_eV: float, n_e_cm3: float, total_density_cm3: float
    ) -> Dict[int, float]:
        """
        Solve ionization balance.

        Parameters
        ----------
        element : str
            Element symbol
        T_e_eV : float
            Electron temperature in eV
        n_e_cm3 : float
            Electron density in cm^-3
        total_density_cm3 : float
            Total element density in cm^-3

        Returns
        -------
        Dict[int, float]
            Dictionary mapping ionization stage to density
        """
        pass

    @abstractmethod
    def solve_level_population(
        self, element: str, ionization_stage: int, stage_density_cm3: float, T_e_eV: float
    ) -> Dict[Tuple[str, int, float], float]:
        """
        Solve level population distribution.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage
        stage_density_cm3 : float
            Stage density in cm^-3
        T_e_eV : float
            Electron temperature in eV

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Dictionary mapping (element, stage, energy) to population
        """
        pass

    @abstractmethod
    def solve_plasma(self, plasma: "SingleZoneLTEPlasma") -> Dict[Tuple[str, int, float], float]:
        """
        Solve complete plasma system.

        Parameters
        ----------
        plasma : SingleZoneLTEPlasma
            Plasma state

        Returns
        -------
        Dict[Tuple[str, int, float], float]
            Complete population dictionary
        """
        pass


class PlasmaModel(ABC):
    """
    Abstract interface for plasma models.

    Allows different plasma models (single-zone LTE, multi-zone, non-LTE, etc.)
    to be used interchangeably.
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate plasma state."""
        pass

    @property
    @abstractmethod
    def T_e_eV(self) -> float:
        """Electron temperature in eV."""
        pass

    @property
    @abstractmethod
    def n_e(self) -> float:
        """Electron density in cm^-3."""
        pass

    @property
    @abstractmethod
    def species(self) -> Dict[str, float]:
        """Species densities in cm^-3."""
        pass


class InstrumentModelInterface(ABC):
    """
    Abstract interface for instrument models.

    Allows different instrument models to be plugged in.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def apply_instrument_function(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> np.ndarray:
        """
        Apply instrument function (convolution).

        Parameters
        ----------
        wavelength : array
            Wavelength grid in nm
        intensity : array
            Intensity spectrum

        Returns
        -------
        array
            Convolved intensity
        """
        pass

    @property
    @abstractmethod
    def resolution_sigma_nm(self) -> float:
        """Instrument resolution (Gaussian sigma) in nm."""
        pass
