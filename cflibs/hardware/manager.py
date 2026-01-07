"""
Hardware manager for coordinating multiple hardware components.
"""

from typing import Dict, Optional, List, Any
from pathlib import Path

from cflibs.hardware.abc import HardwareComponent
from cflibs.hardware.factory import HardwareFactory
from cflibs.core.logging_config import get_logger

logger = get_logger("hardware.manager")


class HardwareManager:
    """
    Manager for coordinating multiple hardware components.

    This class provides a centralized interface for managing all hardware
    components in the CF-LIBS system, including connection management,
    status monitoring, and coordinated operations.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize hardware manager.

        Parameters
        ----------
        config_path : Path, optional
            Path to hardware configuration file
        """
        self.components: Dict[str, HardwareComponent] = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: Path) -> None:
        """
        Load hardware configuration from file.

        Parameters
        ----------
        config_path : Path
            Path to configuration file
        """
        logger.info(f"Loading hardware configuration from {config_path}")
        components = HardwareFactory.create_from_config(config_path)
        self.components.update(components)
        logger.info(f"Loaded {len(components)} hardware components")

    def add_component(self, name: str, component: HardwareComponent) -> None:
        """
        Add a hardware component.

        Parameters
        ----------
        name : str
            Component name
        component : HardwareComponent
            Component instance
        """
        if name in self.components:
            logger.warning(f"Component '{name}' already exists, replacing")
        self.components[name] = component
        logger.debug(f"Added component: {name}")

    def get_component(self, name: str) -> Optional[HardwareComponent]:
        """
        Get a hardware component by name.

        Parameters
        ----------
        name : str
            Component name

        Returns
        -------
        HardwareComponent or None
            Component instance, or None if not found
        """
        return self.components.get(name)

    def connect_all(self) -> Dict[str, bool]:
        """
        Connect all hardware components.

        Returns
        -------
        dict
            Dictionary mapping component names to connection success status
        """
        results = {}
        for name, component in self.components.items():
            logger.info(f"Connecting {name}...")
            try:
                results[name] = component.connect()
                if results[name]:
                    component.initialize()
            except Exception as e:
                logger.error(f"Error connecting {name}: {e}")
                results[name] = False
        return results

    def disconnect_all(self) -> Dict[str, bool]:
        """
        Disconnect all hardware components.

        Returns
        -------
        dict
            Dictionary mapping component names to disconnection success status
        """
        results = {}
        for name, component in self.components.items():
            logger.info(f"Disconnecting {name}...")
            try:
                results[name] = component.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")
                results[name] = False
        return results

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all hardware components.

        Returns
        -------
        dict
            Dictionary mapping component names to status dictionaries
        """
        return {name: component.get_status() for name, component in self.components.items()}

    def get_ready_components(self) -> List[str]:
        """
        Get list of components that are ready for operation.

        Returns
        -------
        list
            List of component names that are ready
        """
        return [name for name, component in self.components.items() if component.is_ready]

    def get_connected_components(self) -> List[str]:
        """
        Get list of components that are connected.

        Returns
        -------
        list
            List of component names that are connected
        """
        return [name for name, component in self.components.items() if component.is_connected]

    def get_components_by_type(self, component_type: str) -> Dict[str, HardwareComponent]:
        """
        Get components filtered by type.

        Parameters
        ----------
        component_type : str
            Component type ('spectrograph', 'laser', etc.)

        Returns
        -------
        dict
            Dictionary mapping component names to instances
        """
        # Simple heuristic: check if component class name contains type
        return {
            name: comp
            for name, comp in self.components.items()
            if component_type.lower() in comp.__class__.__name__.lower()
        }

    def reset_all(self) -> Dict[str, bool]:
        """
        Reset all hardware components.

        Returns
        -------
        dict
            Dictionary mapping component names to reset success status
        """
        results = {}
        for name, component in self.components.items():
            logger.info(f"Resetting {name}...")
            try:
                results[name] = component.reset()
            except Exception as e:
                logger.error(f"Error resetting {name}: {e}")
                results[name] = False
        return results

    def __enter__(self):
        """Context manager entry."""
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_all()
