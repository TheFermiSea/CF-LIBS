"""
Atomic data structures and database interfaces.

This module provides:
- Energy level representations
- Transition probability data structures
- Stark and broadening parameters
- Interfaces to external databases (NIST ASD, Kurucz, etc.)
"""

from cflibs.atomic.structures import EnergyLevel, Transition, SpeciesPhysics, PartitionFunction


# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == "AtomicDatabase":
        from cflibs.atomic.database import AtomicDatabase

        return AtomicDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnergyLevel",
    "Transition",
    "SpeciesPhysics",
    "PartitionFunction",
    "AtomicDatabase",
]
