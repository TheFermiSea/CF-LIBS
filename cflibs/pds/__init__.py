"""
PDS (Planetary Data System) ingestion for ChemCam and SuperCam LIBS data.

This package provides:
- Corpus definitions for curated evaluation datasets
- Download and caching of PDS data products
- Parsers for ChemCam CL5 and SuperCam calibrated spectra
- Mapping into the internal validation schema
"""

from cflibs.pds.corpus import PDSCorpus, CorpusEntry
from cflibs.pds.cache import PDSCache
from cflibs.pds.chemcam import ChemCamParser
from cflibs.pds.supercam import SuperCamParser

__all__ = [
    "PDSCorpus",
    "CorpusEntry",
    "PDSCache",
    "ChemCamParser",
    "SuperCamParser",
]
