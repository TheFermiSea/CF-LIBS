"""
Plasma state representations.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

logger = get_logger("plasma.state")


@dataclass
class PlasmaState:
    """
    Base plasma state representation.

    Attributes
    ----------
    T_e : float
        Electron temperature in K
    T_g : float
        Gas/ion temperature in K (optional, defaults to T_e)
    n_e : float
        Electron density in cm^-3
    species : Dict[str, float]
        Species number densities in cm^-3 (key: element symbol)
    pressure : Optional[float]
        Pressure in atm (optional)
    """

    T_e: float  # K
    n_e: float  # cm^-3
    species: Dict[str, float]  # cm^-3
    T_g: Optional[float] = None
    pressure: Optional[float] = None

    @property
    def T_e_eV(self) -> float:
        """Electron temperature in eV."""
        return self.T_e * KB_EV

    @property
    def T_g_eV(self) -> float:
        """Gas temperature in eV."""
        if self.T_g is None:
            return self.T_e_eV
        return self.T_g * KB_EV


class SingleZoneLTEPlasma(PlasmaState):
    """
    Single-zone LTE plasma model.

    This is the simplest plasma model: a homogeneous, optically thin
    plasma in local thermodynamic equilibrium.
    """

    def __init__(
        self,
        T_e: float,
        n_e: float,
        species: Dict[str, float],
        T_g: Optional[float] = None,
        pressure: Optional[float] = None,
    ):
        """
        Initialize single-zone LTE plasma.

        Parameters
        ----------
        T_e : float
            Electron temperature in K
        n_e : float
            Electron density in cm^-3
        species : Dict[str, float]
            Species number densities in cm^-3
        T_g : float, optional
            Gas temperature in K (defaults to T_e)
        pressure : float, optional
            Pressure in atm
        """
        super().__init__(T_e, n_e, species, T_g, pressure)
        logger.info(
            f"Created SingleZoneLTEPlasma: T_e={T_e:.1f} K, n_e={n_e:.2e} cm^-3, "
            f"species={list(species.keys())}"
        )

    def validate(self) -> bool:
        """
        Validate plasma state.

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If plasma state is invalid
        """
        if self.T_e <= 0:
            raise ValueError("Electron temperature must be positive")

        if self.n_e <= 0:
            raise ValueError("Electron density must be positive")

        if not self.species:
            raise ValueError("At least one species must be specified")

        for element, density in self.species.items():
            if density <= 0:
                raise ValueError(f"Species density for {element} must be positive")

        # Check charge neutrality (rough check)
        # For a simple check, we assume single ionization
        total_charge = sum(self.species.values())
        if abs(self.n_e - total_charge) / max(self.n_e, total_charge) > 0.5:
            logger.warning(
                f"Charge neutrality check: n_e={self.n_e:.2e}, "
                f"estimated charge={total_charge:.2e}"
            )

        return True
