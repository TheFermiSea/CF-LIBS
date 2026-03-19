"""
Core numerical kernels and utilities.

This module provides:
- Physical constants
- Units and unit conversion
- Configuration and logging
- Caching utilities
- Abstract base classes
- Factory patterns
- Connection pooling
"""

from importlib import import_module

__all__ = [
    # Modules
    "constants",
    "units",
    "config",
    "logging_config",
    # Caching
    "LRUCache",
    "cached_partition_function",
    "cached_transitions",
    "cached_ionization",
    "get_cache_stats",
    "clear_all_caches",
    # Abstract base classes
    "AtomicDataSource",
    "SolverStrategy",
    "PlasmaModel",
    "InstrumentModelInterface",
    # Factories
    "SolverFactory",
    "PlasmaModelFactory",
    "InstrumentFactory",
    # Connection pooling
    "DatabaseConnectionPool",
    "get_pool",
    "close_all_pools",
    # Platform configuration
    "configure_jax",
    "AcceleratorBackend",
]

_MODULE_EXPORTS = {
    "constants": "cflibs.core.constants",
    "units": "cflibs.core.units",
    "config": "cflibs.core.config",
    "logging_config": "cflibs.core.logging_config",
}

_ATTRIBUTE_EXPORTS = {
    "configure_jax": ("cflibs.core.platform_config", "configure_jax"),
    "AcceleratorBackend": ("cflibs.core.platform_config", "AcceleratorBackend"),
    "LRUCache": ("cflibs.core.cache", "LRUCache"),
    "cached_partition_function": ("cflibs.core.cache", "cached_partition_function"),
    "cached_transitions": ("cflibs.core.cache", "cached_transitions"),
    "cached_ionization": ("cflibs.core.cache", "cached_ionization"),
    "get_cache_stats": ("cflibs.core.cache", "get_cache_stats"),
    "clear_all_caches": ("cflibs.core.cache", "clear_all_caches"),
    "AtomicDataSource": ("cflibs.core.abc", "AtomicDataSource"),
    "SolverStrategy": ("cflibs.core.abc", "SolverStrategy"),
    "PlasmaModel": ("cflibs.core.abc", "PlasmaModel"),
    "InstrumentModelInterface": ("cflibs.core.abc", "InstrumentModelInterface"),
    "SolverFactory": ("cflibs.core.factory", "SolverFactory"),
    "PlasmaModelFactory": ("cflibs.core.factory", "PlasmaModelFactory"),
    "InstrumentFactory": ("cflibs.core.factory", "InstrumentFactory"),
    "DatabaseConnectionPool": ("cflibs.core.pool", "DatabaseConnectionPool"),
    "get_pool": ("cflibs.core.pool", "get_pool"),
    "close_all_pools": ("cflibs.core.pool", "close_all_pools"),
}


def __getattr__(name: str):
    """Lazy-load core exports to avoid importing optional stacks eagerly."""
    if name in _MODULE_EXPORTS:
        module = import_module(_MODULE_EXPORTS[name])
        globals()[name] = module
        return module
    if name in _ATTRIBUTE_EXPORTS:
        module_name, attr_name = _ATTRIBUTE_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
