"""
Connection pooling and resource management.
"""

from typing import Dict
from contextlib import contextmanager
import threading
from queue import Queue, Empty
import sqlite3

from cflibs.core.logging_config import get_logger

logger = get_logger("core.pool")


class DatabaseConnectionPool:
    """
    Thread-safe connection pool for SQLite databases.

    This improves performance by reusing database connections instead of
    creating new ones for each query.
    """

    def __init__(self, db_path: str, max_connections: int = 5, timeout: float = 5.0):
        """
        Initialize connection pool.

        Parameters
        ----------
        db_path : str
            Path to SQLite database
        max_connections : int
            Maximum number of connections in pool
        timeout : float
            Timeout in seconds for acquiring connection
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self._created = 0
        self._lock = threading.Lock()

        logger.info(f"Initialized connection pool for {db_path} " f"(max={max_connections})")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row
        # Optimize for read-heavy workloads
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64 MB cache
        return conn

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager).

        Yields
        ------
        sqlite3.Connection
            Database connection

        Example
        -------
        >>> with pool.get_connection() as conn:
        ...     cursor = conn.execute("SELECT * FROM lines")
        """
        conn = None
        try:
            # Try to get from pool
            try:
                conn = self._pool.get(timeout=self.timeout)
            except Empty:
                # Pool empty, create new if under limit
                with self._lock:
                    if self._created < self.max_connections:
                        conn = self._create_connection()
                        self._created += 1
                        logger.debug(
                            f"Created new connection ({self._created}/{self.max_connections})"
                        )
                    else:
                        # Wait for available connection
                        conn = self._pool.get(timeout=self.timeout)

            yield conn

        finally:
            # Return to pool
            if conn is not None:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    # Pool full, close connection
                    conn.close()
                    with self._lock:
                        self._created -= 1

    def close_all(self) -> None:
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break

        with self._lock:
            self._created = 0

        logger.info("All connections closed")


# Global pool registry
_pools: Dict[str, DatabaseConnectionPool] = {}


def get_pool(db_path: str, **kwargs) -> DatabaseConnectionPool:
    """
    Get or create a connection pool for a database.

    Parameters
    ----------
    db_path : str
        Path to database
    **kwargs
        Additional arguments for pool creation

    Returns
    -------
    DatabaseConnectionPool
        Connection pool instance
    """
    if db_path not in _pools:
        _pools[db_path] = DatabaseConnectionPool(db_path, **kwargs)

    return _pools[db_path]


def close_all_pools() -> None:
    """Close all connection pools."""
    for pool in _pools.values():
        pool.close_all()
    _pools.clear()
