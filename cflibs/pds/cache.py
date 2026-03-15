"""
PDS data download and local cache layer.

Fetches PDS data products into a deterministic local cache layout.
Re-running acquisition is idempotent and avoids unnecessary downloads.

Cache layout:
    <cache_dir>/
        chemcam/
            sol00069/
                CL5_398755580RCE_....csv
        supercam/
            sol00082/
                SC3_0082_....csv
"""

from __future__ import annotations

import shutil
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Union

from cflibs.core.logging_config import get_logger
from cflibs.pds.corpus import CorpusEntry

logger = get_logger("pds.cache")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cflibs" / "pds"

# Minimum size (bytes) for a cached file to be considered valid.
# PDS CSV files are at least a few KB; anything smaller is likely
# a truncated download or an HTML error page.
_MIN_VALID_SIZE = 256


class PDSCache:
    """Local cache for PDS data products.

    Parameters
    ----------
    cache_dir : Path, str, or None
        Root directory for cached files. Defaults to ``~/.cache/cflibs/pds/``.
    """

    def __init__(self, cache_dir: Union[Path, str, None] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, entry: CorpusEntry) -> Path:
        """Deterministic subdirectory for a corpus entry."""
        instrument = entry.instrument.value  # "chemcam" or "supercam"
        sol_dir = f"sol{entry.sol:05d}"
        d = self.cache_dir / instrument / sol_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _entry_filename(self, entry: CorpusEntry) -> str:
        """Filename for a cached entry."""
        return f"{entry.product_id}.csv"

    def cached_path(self, entry: CorpusEntry) -> Path:
        """Return the local cache path for an entry (may not exist yet)."""
        return self._entry_dir(entry) / self._entry_filename(entry)

    def is_cached(self, entry: CorpusEntry) -> bool:
        """Check if the entry is in the local cache with valid content.

        A file is considered valid only if it exists, exceeds the minimum
        size threshold, and does not look like an HTML error page.
        """
        p = self.cached_path(entry)
        if not p.exists():
            return False
        size = p.stat().st_size
        if size < _MIN_VALID_SIZE:
            return False
        # Reject HTML error pages (PDS servers return these on 404)
        try:
            head = p.read_bytes()[:64]
            if b"<html" in head.lower() or b"<!doctype" in head.lower():
                return False
        except OSError:
            return False
        return True

    def fetch(self, entry: CorpusEntry, timeout: int = 60) -> Path:
        """Download a PDS data product if not already cached.

        Parameters
        ----------
        entry : CorpusEntry
            Corpus entry to fetch.
        timeout : int
            HTTP timeout in seconds.

        Returns
        -------
        Path
            Local path to the cached file.

        Raises
        ------
        RuntimeError
            If the download fails.
        """
        dest = self.cached_path(entry)
        if self.is_cached(entry):
            logger.debug("Cache hit: %s", dest)
            return dest

        url = self._build_url(entry)
        logger.info("Downloading %s → %s", url, dest)

        try:
            response = urllib.request.urlopen(url, timeout=timeout)
            with open(dest, "wb") as f:
                f.write(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Failed to download PDS product {entry.product_id}: {exc}") from exc

        logger.info("Downloaded %d bytes → %s", dest.stat().st_size, dest.name)
        return dest

    def fetch_all(self, entries: list[CorpusEntry], timeout: int = 60) -> dict[str, Path]:
        """Download all entries, returning a mapping of entry_id → local path.

        Skips entries that are already cached. Returns only successfully
        fetched entries; logs warnings for failures.
        """
        results: dict[str, Path] = {}
        for entry in entries:
            try:
                path = self.fetch(entry, timeout=timeout)
                results[entry.entry_id] = path
            except RuntimeError as exc:
                logger.warning("Skipping %s: %s", entry.entry_id, exc)
        return results

    def _build_url(self, entry: CorpusEntry) -> str:
        """Construct the full PDS download URL for an entry."""
        base = entry.pds_base_url.rstrip("/")
        rel = entry.relative_path.strip("/")
        filename = self._entry_filename(entry)
        return f"{base}/{rel}/{filename}"

    def clear(self, entry: Optional[CorpusEntry] = None) -> None:
        """Remove cached files.

        Parameters
        ----------
        entry : CorpusEntry, optional
            If given, remove only this entry. Otherwise clear all.
        """
        if entry is not None:
            p = self.cached_path(entry)
            if p.exists():
                p.unlink()
                logger.info("Removed cached file: %s", p)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared PDS cache: %s", self.cache_dir)

    def status(self) -> dict[str, int]:
        """Report cache statistics."""
        n_files = sum(1 for _ in self.cache_dir.rglob("*") if _.is_file())
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())
        return {"n_files": n_files, "total_bytes": total_bytes}
