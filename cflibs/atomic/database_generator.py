"""
Atomic database generator for CF-LIBS.

This module provides utilities for generating the atomic database from NIST data.
The database is required for all CF-LIBS calculations.
"""

import sys
from pathlib import Path

# Import the datagen_v2 script functionality
# This allows us to use it as a module or CLI command


def generate_database(
    db_path: str = "libs_production.db",
    elements: list = None,
    max_ionization_stage: int = 2,
    max_upper_energy_ev: float = 12.0,
    min_relative_intensity: float = 50.0,
) -> None:
    """
    Generate atomic database from NIST data.

    This function wraps the datagen_v2.py script functionality to allow
    programmatic database generation.

    Parameters
    ----------
    db_path : str
        Path to output database file
    elements : list, optional
        List of element symbols to include. If None, includes all elements.
    max_ionization_stage : int
        Maximum ionization stage to include (default: 2)
    max_upper_energy_ev : float
        Maximum upper energy level in eV (default: 12.0)
    min_relative_intensity : float
        Minimum relative intensity threshold (default: 50.0)

    Notes
    -----
    This requires the ASDCache library and internet access to fetch NIST data.
    The database generation can take a significant amount of time (hours) for
    all elements.

    See Also
    --------
    datagen_v2.py : Standalone script for database generation
    """
    # Import the datagen_v2 module
    datagen_path = Path(__file__).parent.parent.parent / "datagen_v2.py"

    if not datagen_path.exists():
        raise FileNotFoundError(
            f"datagen_v2.py not found at {datagen_path}. "
            "Please ensure the script is in the project root."
        )

    # For now, we'll recommend using the script directly
    # In the future, we can refactor datagen_v2.py to be importable
    import subprocess

    print(f"Generating atomic database: {db_path}")
    print("Note: This may take a long time. Consider using datagen_v2.py directly.")

    # Run the script
    result = subprocess.run([sys.executable, str(datagen_path)], cwd=str(datagen_path.parent))

    if result.returncode != 0:
        raise RuntimeError("Database generation failed. See error messages above.")


if __name__ == "__main__":
    # Allow running as a module
    import argparse

    parser = argparse.ArgumentParser(description="Generate atomic database for CF-LIBS")
    parser.add_argument(
        "--db-path", default="libs_production.db", help="Path to output database file"
    )

    args = parser.parse_args()
    generate_database(db_path=args.db_path)
