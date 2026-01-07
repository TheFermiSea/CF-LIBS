"""
Hardware interfaces for CF-LIBS instrumentation.

This module provides interfaces and placeholders for:
- Spectrographs/detectors
- Lasers
- Motion stages
- Powder flow regulators
- Other hardware components

These interfaces are designed to integrate with custom GUI applications
and hardware control systems.
"""

from cflibs.hardware.abc import (
    HardwareComponent,
    SpectrographInterface,
    LaserInterface,
    MotionStageInterface,
    FlowRegulatorInterface,
)
from cflibs.hardware.spectrograph import SpectrographHardware
from cflibs.hardware.laser import LaserHardware
from cflibs.hardware.stages import MotionStageHardware
from cflibs.hardware.flow import FlowRegulatorHardware
from cflibs.hardware.factory import HardwareFactory
from cflibs.hardware.manager import HardwareManager

__all__ = [
    # Abstract interfaces
    "HardwareComponent",
    "SpectrographInterface",
    "LaserInterface",
    "MotionStageInterface",
    "FlowRegulatorInterface",
    # Implementations
    "SpectrographHardware",
    "LaserHardware",
    "MotionStageHardware",
    "FlowRegulatorHardware",
    # Factories and managers
    "HardwareFactory",
    "HardwareManager",
]
