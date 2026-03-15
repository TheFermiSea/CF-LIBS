"""
Validation schema for PDS-backed LIBS datasets.

Maps parsed ChemCam and SuperCam products into a common validation
dataset schema that records provenance, expected-element context,
and instrument-response assumptions. Connects the PDS ingestion layer
to the existing round-trip validation and benchmark infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.pds.corpus import CorpusEntry
from cflibs.pds.chemcam import ChemCamSpectrum
from cflibs.pds.supercam import SuperCamSpectrum

logger = get_logger("pds.validation")


@dataclass
class PDSValidationDataset:
    """A parsed PDS spectrum mapped into the validation schema.

    This unified representation works for both ChemCam and SuperCam
    data, recording provenance, expected elements, and instrument
    context. It can be consumed by the real-data validator and
    benchmark infrastructure.

    Attributes
    ----------
    entry_id : str
        Corpus entry identifier for traceability.
    instrument : str
        Instrument name ("chemcam" or "supercam").
    target_name : str
        Observation target name.
    sol : int
        Mars sol number.
    wavelength : np.ndarray
        Wavelength array in nm.
    intensity : np.ndarray
        Calibrated intensity array.
    expected_elements : Dict[str, Optional[float]]
        Elements expected in this target, with known weight fractions
        where independently measured.
    spectrometer_ranges : List[Tuple[int, int]]
        Index ranges for each spectrometer segment.
    provenance : Dict
        Traceability information (source URL, product ID, etc.).
    ground_truth_quality : str
        Honest assessment of ground truth reliability:
        "quantified" (known compositions), "qualitative" (expected
        elements only), or "unknown".
    notes : str
        Mars-specific assumptions or caveats.
    """

    entry_id: str
    instrument: str
    target_name: str
    sol: int
    wavelength: np.ndarray
    intensity: np.ndarray
    expected_elements: Dict[str, Optional[float]]
    spectrometer_ranges: List[Tuple[int, int]] = field(default_factory=list)
    provenance: Dict = field(default_factory=dict)
    ground_truth_quality: str = "unknown"
    notes: str = ""

    @property
    def n_quantified_elements(self) -> int:
        """Number of elements with known weight fractions."""
        return sum(1 for v in self.expected_elements.values() if v is not None)

    @property
    def quantified_elements(self) -> Dict[str, float]:
        """Only elements with known weight fractions."""
        return {k: v for k, v in self.expected_elements.items() if v is not None}

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Overall wavelength range in nm."""
        return (float(self.wavelength.min()), float(self.wavelength.max()))


def map_chemcam_to_validation(
    spectrum: ChemCamSpectrum,
    entry: CorpusEntry,
) -> PDSValidationDataset:
    """Map a parsed ChemCam spectrum into the validation schema."""
    provenance = {
        "product_id": spectrum.product_id,
        "pds_base_url": entry.pds_base_url,
        "relative_path": entry.relative_path,
        "instrument": "chemcam",
        "n_shots": spectrum.n_shots,
        **spectrum.metadata,
    }

    quality = "quantified" if entry.expected_elements else "unknown"
    has_quantified = any(v is not None for v in entry.expected_elements.values())
    if has_quantified:
        quality = "quantified"
    elif entry.expected_elements:
        quality = "qualitative"

    return PDSValidationDataset(
        entry_id=entry.entry_id,
        instrument="chemcam",
        target_name=entry.target_name,
        sol=entry.sol,
        wavelength=spectrum.wavelength,
        intensity=spectrum.intensity,
        expected_elements=dict(entry.expected_elements),
        spectrometer_ranges=spectrum.spectrometer_ranges,
        provenance=provenance,
        ground_truth_quality=quality,
        notes=entry.notes,
    )


def map_supercam_to_validation(
    spectrum: SuperCamSpectrum,
    entry: CorpusEntry,
) -> PDSValidationDataset:
    """Map a parsed SuperCam spectrum into the validation schema."""
    provenance = {
        "product_id": spectrum.product_id,
        "pds_base_url": entry.pds_base_url,
        "relative_path": entry.relative_path,
        "instrument": "supercam",
        "n_shots": spectrum.n_shots,
        **spectrum.metadata,
    }

    quality = "unknown"
    has_quantified = any(v is not None for v in entry.expected_elements.values())
    if has_quantified:
        quality = "quantified"
    elif entry.expected_elements:
        quality = "qualitative"

    return PDSValidationDataset(
        entry_id=entry.entry_id,
        instrument="supercam",
        target_name=entry.target_name,
        sol=entry.sol,
        wavelength=spectrum.wavelength,
        intensity=spectrum.intensity,
        expected_elements=dict(entry.expected_elements),
        spectrometer_ranges=spectrum.spectrometer_ranges,
        provenance=provenance,
        ground_truth_quality=quality,
        notes=entry.notes,
    )
