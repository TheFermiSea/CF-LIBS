"""
Atomic database interface for loading and querying atomic data.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from cflibs.atomic.structures import Transition, EnergyLevel, SpeciesPhysics, PartitionFunction
from cflibs.core.logging_config import get_logger
from cflibs.core.abc import AtomicDataSource
from cflibs.core.cache import cached_transitions, cached_ionization
from cflibs.core.pool import get_pool

logger = get_logger("atomic.database")


class AtomicDatabase(AtomicDataSource):
    """
    Interface to atomic data stored in SQLite database.

    The database should have the following tables:
    - `lines`: Spectral line data
    - `energy_levels`: Energy level data
    - `species_physics`: Ionization potentials and species properties
    - `partition_functions`: Partition function coefficients
    """

    def __init__(self, db_path: str):
        """
        Initialize the AtomicDatabase by opening the SQLite file, establishing a connection (preferably pooled), and ensuring the database schema is up-to-date.
        
        Parameters:
            db_path (str): Path to the SQLite database file.
        
        Raises:
            FileNotFoundError: If the specified database file does not exist.
        
        Notes:
            - Attempts to create a connection pool; if pool creation fails, a direct sqlite3 connection is used.
            - Performs schema checks and migrations as needed before returning.
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Atomic database not found: {db_path}")

        self.db_path = db_path
        # Use connection pool for better performance
        try:
            self._pool = get_pool(str(db_path), max_connections=5)
            self._use_pool = True
        except Exception as e:
            # Fallback to direct connection if pool fails
            logger.warning(f"Failed to create connection pool, using direct connection: {e}")
            self.conn = sqlite3.connect(str(db_path))
            self._use_pool = False

        # Verify and migrate schema if needed
        self._check_and_migrate_schema()
        logger.info(f"Connected to atomic database: {db_path}")

    @contextmanager
    def _get_connection(self):
        """
        Provide a database connection for executing queries.
        
        Returns:
            sqlite3.Connection: A connection obtained from the pool when pooling is enabled; otherwise the persistent direct connection.
        """
        if self._use_pool:
            with self._pool.get_connection() as conn:
                yield conn
        else:
            yield self.conn

    def _check_and_migrate_schema(self):
        """
        Ensure the database schema matches expectations and apply migrations when needed.
        
        Attempts to perform required schema migration using an internal database connection and re-raises any exception encountered.
        
        Raises:
            Exception: If schema migration fails.
        """
        try:
            with self._get_connection() as conn:
                self._perform_migration(conn)
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise

    def _perform_migration(self, conn: sqlite3.Connection):
        """Perform the actual migration steps."""
        cursor = conn.cursor()

        # 1. Check lines table columns
        cursor.execute("PRAGMA table_info(lines)")
        columns = {row[1] for row in cursor.fetchall()}

        required_line_cols = {
            "stark_w": "REAL",
            "stark_alpha": "REAL",
            "stark_shift": "REAL",
            "is_resonance": "INTEGER",
        }

        for col, dtype in required_line_cols.items():
            if col not in columns:
                logger.info(f"Migrating schema: Adding {col} to lines table")
                cursor.execute(f"ALTER TABLE lines ADD COLUMN {col} {dtype}")

                # Backfill is_resonance if we just added it
                if col == "is_resonance":
                    logger.info("Backfilling is_resonance based on ei_ev")
                    # SQLite doesn't strictly support boolean, so 1/0
                    # Assuming ei_ev exists and is populated
                    cursor.execute("UPDATE lines SET is_resonance = 1 WHERE ei_ev < 0.01")
                    cursor.execute(
                        "UPDATE lines SET is_resonance = 0 WHERE ei_ev >= 0.01 OR ei_ev IS NULL"
                    )

        # 2. Check partition_functions table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='partition_functions'"
        )
        if not cursor.fetchone():
            logger.info("Migrating schema: Creating partition_functions table")
            cursor.execute(
                """
                CREATE TABLE partition_functions (
                    element TEXT,
                    sp_num INTEGER,
                    a0 REAL,
                    a1 REAL,
                    a2 REAL,
                    a3 REAL,
                    a4 REAL,
                    t_min REAL,
                    t_max REAL,
                    source TEXT,
                    PRIMARY KEY (element, sp_num)
                )
            """
            )

        # 3. Check species_physics table for atomic_mass
        cursor.execute("PRAGMA table_info(species_physics)")
        physics_columns = {row[1] for row in cursor.fetchall()}

        if "atomic_mass" not in physics_columns:
            logger.info("Migrating schema: Adding atomic_mass to species_physics table")
            cursor.execute("ALTER TABLE species_physics ADD COLUMN atomic_mass REAL")

        conn.commit()

    @cached_transitions
    def get_transitions(
        self,
        element: str,
        ionization_stage: Optional[int] = None,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        min_relative_intensity: Optional[float] = None,
    ) -> List[Transition]:
        """
        Retrieve spectral transitions for a given element, optionally filtered by ionization stage, wavelength range, and minimum relative intensity.
        
        Parameters:
            element (str): Element symbol.
            ionization_stage (int, optional): Ionization stage where 1 = neutral, 2 = singly ionized, etc.
            wavelength_min (float, optional): Minimum wavelength in nanometers (inclusive).
            wavelength_max (float, optional): Maximum wavelength in nanometers (inclusive).
            min_relative_intensity (float, optional): Minimum relative intensity threshold; rows with `rel_int` below this value are excluded.
        
        Returns:
            List[Transition]: List of Transition objects matching the provided filters. Missing numeric fields are converted to sensible defaults or `None` where applicable.
        """
        # Check if new columns exist in the actual query execution (though schema check should have fixed it)
        # We select all relevant columns.
        query = """
            SELECT 
                element, sp_num, wavelength_nm, aki, ek_ev, ei_ev, 
                gk, gi, rel_int,
                stark_w, stark_alpha, stark_shift, is_resonance
            FROM lines
            WHERE element = ?
        """
        params = [element]

        if ionization_stage is not None:
            query += " AND sp_num = ?"
            params.append(ionization_stage)

        if wavelength_min is not None:
            query += " AND wavelength_nm >= ?"
            params.append(wavelength_min)

        if wavelength_max is not None:
            query += " AND wavelength_nm <= ?"
            params.append(wavelength_max)

        if min_relative_intensity is not None:
            query += " AND rel_int >= ?"
            params.append(min_relative_intensity)

        query += " ORDER BY wavelength_nm"

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error querying transitions: {e}")
            return []

        transitions = []
        for _, row in df.iterrows():
            # Handle potential missing columns if something went wrong, defaulting to None
            stark_w = (
                float(row["stark_w"]) if "stark_w" in row and pd.notna(row["stark_w"]) else None
            )
            stark_alpha = (
                float(row["stark_alpha"])
                if "stark_alpha" in row and pd.notna(row["stark_alpha"])
                else None
            )
            stark_shift = (
                float(row["stark_shift"])
                if "stark_shift" in row and pd.notna(row["stark_shift"])
                else None
            )
            is_resonance = (
                bool(row["is_resonance"])
                if "is_resonance" in row and pd.notna(row["is_resonance"])
                else False
            )

            trans = Transition(
                element=row["element"],
                ionization_stage=int(row["sp_num"]),
                wavelength_nm=float(row["wavelength_nm"]),
                A_ki=float(row["aki"]),
                E_k_ev=float(row["ek_ev"]),
                E_i_ev=float(0.0)
                if pd.isna(row.get("ei_ev", 0.0))
                else float(row.get("ei_ev", 0.0)),
                g_k=int(row["gk"]),
                g_i=int(1) if pd.isna(row.get("gi", 1)) else int(row.get("gi", 1)),
                relative_intensity=(
                    float(row.get("rel_int", 0.0)) if pd.notna(row.get("rel_int")) else None
                ),
                stark_w=stark_w,
                stark_alpha=stark_alpha,
                stark_shift=stark_shift,
                is_resonance=is_resonance,
            )
            transitions.append(trans)

        logger.debug(f"Retrieved {len(transitions)} transitions for {element}")
        return transitions

    def get_energy_levels(self, element: str, ionization_stage: int) -> List[EnergyLevel]:
        """
        Retrieve energy levels for the specified element and ionization stage.
        
        Parameters:
            element (str): Element symbol (e.g., "Fe").
            ionization_stage (int): Ionization stage number (sp_num in the database).
        
        Returns:
            List[EnergyLevel]: EnergyLevel objects ordered by increasing energy_ev.
        """
        query = """
            SELECT g_level, energy_ev
            FROM energy_levels
            WHERE element = ? AND sp_num = ?
            ORDER BY energy_ev
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(element, ionization_stage))

        levels = []
        for _, row in df.iterrows():
            level = EnergyLevel(
                element=element,
                ionization_stage=ionization_stage,
                energy_ev=float(row["energy_ev"]),
                g=int(row["g_level"]),
            )
            levels.append(level)

        logger.debug(f"Retrieved {len(levels)} energy levels for {element} {ionization_stage}")
        return levels

    @cached_ionization
    def get_ionization_potential(self, element: str, ionization_stage: int) -> Optional[float]:
        """
        Get ionization potential for a species.

        Parameters
        ----------
        element : str
            Element symbol
        ionization_stage : int
            Ionization stage (1=neutral, 2=singly ionized, etc.)

        Returns
        -------
        float or None
            Ionization potential in eV, or None if not found
        """
        query = """
            SELECT ip_ev
            FROM species_physics
            WHERE element = ? AND sp_num = ?
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element, ionization_stage))
            res = cur.fetchone()
            return float(res[0]) if res else None

    def get_atomic_mass(self, element: str) -> Optional[float]:
        """
        Return the atomic mass for the given element in atomic mass units.
        
        Parameters:
            element (str): Element symbol (e.g., 'Fe').
        
        Returns:
            float or None: Atomic mass in amu if present in the database, otherwise None.
        """
        # Usually atomic mass is per element, not per species, but stored in species_physics which is (element, sp_num).
        # We can grab it from any sp_num or specifically sp_num=0 or 1.
        # Let's query any record for this element.
        query = """
            SELECT atomic_mass
            FROM species_physics
            WHERE element = ? AND atomic_mass IS NOT NULL
            LIMIT 1
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element,))
            res = cur.fetchone()
            return float(res[0]) if res else None

    def get_partition_coefficients(
        self, element: str, ionization_stage: int
    ) -> Optional[PartitionFunction]:
        """
        Retrieve the partition function coefficients, valid temperature range, and source for a given species.
        
        Parameters:
        	element (str): Element symbol (e.g., "Fe").
        	ionization_stage (int): Ionization stage (sp_num) for the species.
        
        Returns:
        	PartitionFunction or None: A PartitionFunction with coefficients [a0, a1, a2, a3, a4], t_min, t_max, and source if an entry exists; `None` if no partition function is found for the given element and ionization stage.
        """
        query = """
            SELECT a0, a1, a2, a3, a4, t_min, t_max, source
            FROM partition_functions
            WHERE element = ? AND sp_num = ?
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element, ionization_stage))
            res = cur.fetchone()

        if not res:
            return None

        return PartitionFunction(
            element=element,
            ionization_stage=ionization_stage,
            coefficients=[res[0], res[1], res[2], res[3], res[4]],
            t_min=res[5],
            t_max=res[6],
            source=res[7],
        )

    def get_species_physics(self, element: str, ionization_stage: int) -> Optional[SpeciesPhysics]:
        """
        Retrieve the ionization potential and atomic mass for a species.
        
        Returns:
            A SpeciesPhysics containing the species' ionization potential (in eV) and atomic mass, or `None` if no record is found.
        """
        query = """
            SELECT ip_ev, atomic_mass
            FROM species_physics
            WHERE element = ? AND sp_num = ?
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (element, ionization_stage))
            res = cur.fetchone()

        if not res:
            return None

        ip_ev = float(res[0])
        atomic_mass = float(res[1]) if res[1] is not None else None

        return SpeciesPhysics(
            element=element,
            ionization_stage=ionization_stage,
            ionization_potential_ev=ip_ev,
            atomic_mass=atomic_mass,
        )

    def get_stark_parameters(
        self, element: str, ionization_stage: int, wavelength_nm: float, tolerance_nm: float = 0.01
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Return Stark broadening parameters for the line nearest the target wavelength within a tolerance.
        
        Parameters:
            element (str): Chemical element symbol to search.
            ionization_stage (int): Ionization stage (sp_num) of the element.
            wavelength_nm (float): Target wavelength in nanometers.
            tolerance_nm (float): Maximum allowed absolute difference (nm) between stored line wavelength and the target.
        
        Returns:
            tuple: (stark_w, stark_alpha, stark_shift) where each value is a float if present in the database, `None` if that parameter is missing, or `(None, None, None)` if no matching line is found.
        """
        query = """
            SELECT stark_w, stark_alpha, stark_shift
            FROM lines
            WHERE element = ? AND sp_num = ?
            AND ABS(wavelength_nm - ?) < ?
            ORDER BY ABS(wavelength_nm - ?) ASC
            LIMIT 1
        """
        params = (element, ionization_stage, wavelength_nm, tolerance_nm, wavelength_nm)

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            res = cur.fetchone()

        if not res:
            return (None, None, None)

        stark_w = float(res[0]) if res[0] is not None else None
        stark_alpha = float(res[1]) if res[1] is not None else None
        stark_shift = float(res[2]) if res[2] is not None else None

        return (stark_w, stark_alpha, stark_shift)

    def get_available_elements(self) -> List[str]:
        """
        Return the distinct element symbols present in the database, ordered alphabetically.
        
        Returns:
            A list of element symbols (strings) available in the `lines` table, sorted by element name.
        """
        query = "SELECT DISTINCT element FROM lines ORDER BY element"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            return df["element"].tolist()

    def close(self):
        """
        Close the database connection held by this AtomicDatabase instance.
        
        If the instance was created with a connection pool, this method does not close the shared pool; it only releases the instance's reference. If the instance uses a direct sqlite3 connection, that connection is closed.
        """
        if self._use_pool:
            # Note: Pool is shared, so we don't close it here
            # Use close_all_pools() if needed
            logger.debug("Database connection pool reference released")
        else:
            self.conn.close()
            logger.debug("Database connection closed")

    def __getstate__(self):
        """Pickle support: exclude connection/pool."""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        state.pop("_pool", None)
        state.pop("conn", None)
        return state

    def __setstate__(self, state):
        """Unpickle support: restore connection/pool."""
        self.__dict__.update(state)
        # Re-initialize connection/pool
        try:
            self._pool = get_pool(str(self.db_path), max_connections=5)
            self._use_pool = True
        except Exception:
            self.conn = sqlite3.connect(str(self.db_path))
            self._use_pool = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Pool is managed globally, no cleanup needed
        pass