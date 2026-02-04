"""
Core data structures for LIBS benchmark datasets.

This module defines the standardized data structures for representing
LIBS spectra with certified compositions and instrumental conditions,
enabling reproducible cross-study comparison.

Attributes
----------
SUPPORTED_ELEMENTS : List[str]
    List of elements commonly analyzed in LIBS benchmarks

References
----------
- NIST Standard Reference Materials program
- IUPAC Technical Report (2008): Harmonization of quality assurance schemes
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("benchmark.dataset")


# Common elements in LIBS analysis
SUPPORTED_ELEMENTS = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
]


class SampleType(Enum):
    """Type of sample material for benchmark spectra."""

    CRM = "crm"  # Certified Reference Material
    SYNTHETIC = "synthetic"  # Synthetic/laboratory prepared
    FIELD = "field"  # Field sample with ICP-validated composition
    SIMULATED = "simulated"  # Computationally generated


class MatrixType(Enum):
    """Matrix type classification for LIBS samples."""

    METAL_ALLOY = "metal_alloy"
    METAL_PURE = "metal_pure"
    GLASS = "glass"
    CERAMIC = "ceramic"
    SOIL = "soil"
    GEOLOGICAL = "geological"
    POLYMER = "polymer"
    BIOLOGICAL = "biological"
    LIQUID = "liquid"
    AEROSOL = "aerosol"
    OTHER = "other"


@dataclass
class InstrumentalConditions:
    """
    Instrumental and experimental conditions for LIBS measurement.

    This dataclass captures the key experimental parameters that affect
    LIBS spectra, enabling proper comparison between datasets acquired
    under different conditions.

    Attributes
    ----------
    laser_wavelength_nm : float
        Laser wavelength in nm (e.g., 1064, 532, 266)
    laser_energy_mj : float
        Laser pulse energy in millijoules
    laser_pulse_width_ns : float
        Laser pulse duration in nanoseconds
    repetition_rate_hz : float
        Laser repetition rate in Hz
    spot_diameter_um : float
        Laser spot diameter on sample in micrometers
    fluence_j_cm2 : float, optional
        Calculated fluence in J/cm^2
    gate_delay_us : float
        Detection gate delay after laser pulse in microseconds
    gate_width_us : float
        Detection gate width in microseconds
    spectrometer_type : str
        Spectrometer type (e.g., "Echelle", "Czerny-Turner")
    spectral_range_nm : Tuple[float, float]
        Wavelength range (min, max) in nm
    spectral_resolution_nm : float
        Spectral resolution in nm
    detector_type : str
        Detector type (e.g., "ICCD", "CCD", "EMCCD")
    accumulations : int
        Number of accumulated spectra
    atmosphere : str
        Measurement atmosphere (e.g., "air", "argon", "helium", "vacuum")
    pressure_mbar : float
        Ambient pressure in mbar (1013.25 for atmospheric)
    standoff_distance_m : float, optional
        Standoff distance for remote LIBS in meters
    notes : str
        Additional experimental notes
    """

    laser_wavelength_nm: float
    laser_energy_mj: float
    laser_pulse_width_ns: float = 10.0
    repetition_rate_hz: float = 10.0
    spot_diameter_um: float = 100.0
    fluence_j_cm2: Optional[float] = None
    gate_delay_us: float = 1.0
    gate_width_us: float = 10.0
    spectrometer_type: str = "Echelle"
    spectral_range_nm: Tuple[float, float] = (200.0, 900.0)
    spectral_resolution_nm: float = 0.05
    detector_type: str = "ICCD"
    accumulations: int = 1
    atmosphere: str = "air"
    pressure_mbar: float = 1013.25
    standoff_distance_m: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        """Calculate derived parameters."""
        if self.fluence_j_cm2 is None and self.spot_diameter_um > 0:
            # Fluence = Energy / Area
            # Convert spot diameter from um to cm
            spot_cm = self.spot_diameter_um * 1e-4
            area_cm2 = np.pi * (spot_cm / 2) ** 2
            self.fluence_j_cm2 = (self.laser_energy_mj * 1e-3) / area_cm2

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "laser_wavelength_nm": self.laser_wavelength_nm,
            "laser_energy_mj": self.laser_energy_mj,
            "laser_pulse_width_ns": self.laser_pulse_width_ns,
            "repetition_rate_hz": self.repetition_rate_hz,
            "spot_diameter_um": self.spot_diameter_um,
            "fluence_j_cm2": self.fluence_j_cm2,
            "gate_delay_us": self.gate_delay_us,
            "gate_width_us": self.gate_width_us,
            "spectrometer_type": self.spectrometer_type,
            "spectral_range_nm": list(self.spectral_range_nm),
            "spectral_resolution_nm": self.spectral_resolution_nm,
            "detector_type": self.detector_type,
            "accumulations": self.accumulations,
            "atmosphere": self.atmosphere,
            "pressure_mbar": self.pressure_mbar,
            "standoff_distance_m": self.standoff_distance_m,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "InstrumentalConditions":
        """Create from dictionary representation."""
        # Handle spectral_range_nm conversion from list
        if "spectral_range_nm" in d and isinstance(d["spectral_range_nm"], list):
            d = d.copy()
            d["spectral_range_nm"] = tuple(d["spectral_range_nm"])
        return cls(**d)


@dataclass
class SampleMetadata:
    """
    Metadata about the sample material.

    Attributes
    ----------
    sample_id : str
        Unique sample identifier
    sample_type : SampleType
        Type of sample (CRM, synthetic, field, simulated)
    matrix_type : MatrixType
        Matrix classification
    crm_name : str, optional
        CRM designation (e.g., "NIST SRM 1261a")
    crm_source : str, optional
        CRM source organization
    preparation : str
        Sample preparation method
    surface_condition : str
        Surface condition (e.g., "polished", "as-received")
    measurement_date : str, optional
        ISO format date of measurement
    laboratory : str, optional
        Laboratory where measurement was performed
    doi : str, optional
        DOI of associated publication
    provenance : str
        Data provenance and traceability information
    """

    sample_id: str
    sample_type: SampleType = SampleType.CRM
    matrix_type: MatrixType = MatrixType.METAL_ALLOY
    crm_name: Optional[str] = None
    crm_source: Optional[str] = None
    preparation: str = "polished"
    surface_condition: str = "polished"
    measurement_date: Optional[str] = None
    laboratory: Optional[str] = None
    doi: Optional[str] = None
    provenance: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "sample_id": self.sample_id,
            "sample_type": self.sample_type.value,
            "matrix_type": self.matrix_type.value,
            "crm_name": self.crm_name,
            "crm_source": self.crm_source,
            "preparation": self.preparation,
            "surface_condition": self.surface_condition,
            "measurement_date": self.measurement_date,
            "laboratory": self.laboratory,
            "doi": self.doi,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SampleMetadata":
        """Create from dictionary representation."""
        d = d.copy()
        if "sample_type" in d and isinstance(d["sample_type"], str):
            d["sample_type"] = SampleType(d["sample_type"])
        if "matrix_type" in d and isinstance(d["matrix_type"], str):
            d["matrix_type"] = MatrixType(d["matrix_type"])
        return cls(**d)


@dataclass
class BenchmarkSpectrum:
    """
    A single LIBS spectrum with certified composition and metadata.

    This is the fundamental unit of a benchmark dataset. Each spectrum
    contains the spectral data, ground truth composition with uncertainties,
    and full documentation of instrumental conditions.

    Attributes
    ----------
    spectrum_id : str
        Unique identifier for this spectrum
    wavelength_nm : np.ndarray
        Wavelength array in nanometers
    intensity : np.ndarray
        Intensity array (arbitrary units, but consistent within dataset)
    intensity_uncertainty : np.ndarray, optional
        Per-pixel intensity uncertainty
    true_composition : Dict[str, float]
        Ground truth elemental composition (mass fractions, sum to 1.0)
    composition_uncertainty : Dict[str, float]
        Composition uncertainty (1-sigma, mass fraction)
    conditions : InstrumentalConditions
        Instrumental and experimental conditions
    metadata : SampleMetadata
        Sample and measurement metadata
    plasma_temperature_K : float, optional
        Estimated plasma temperature if known
    electron_density_cm3 : float, optional
        Estimated electron density if known
    quality_flag : int
        Quality flag (0=good, 1=marginal, 2=poor, -1=excluded)

    Notes
    -----
    Compositions are expressed as mass fractions (weight percent / 100)
    and should sum to approximately 1.0 within uncertainty.
    """

    spectrum_id: str
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    true_composition: Dict[str, float]
    conditions: InstrumentalConditions
    metadata: SampleMetadata
    intensity_uncertainty: Optional[np.ndarray] = None
    composition_uncertainty: Dict[str, float] = field(default_factory=dict)
    plasma_temperature_K: Optional[float] = None
    electron_density_cm3: Optional[float] = None
    quality_flag: int = 0

    def __post_init__(self):
        """Validate and convert arrays."""
        self.wavelength_nm = np.asarray(self.wavelength_nm)
        self.intensity = np.asarray(self.intensity)
        if self.intensity_uncertainty is not None:
            self.intensity_uncertainty = np.asarray(self.intensity_uncertainty)

        # Validate array shapes
        if len(self.wavelength_nm) != len(self.intensity):
            raise ValueError(
                f"Wavelength ({len(self.wavelength_nm)}) and intensity "
                f"({len(self.intensity)}) arrays must have same length"
            )

        if self.intensity_uncertainty is not None:
            if len(self.intensity_uncertainty) != len(self.intensity):
                raise ValueError("Intensity uncertainty must match intensity array length")

        # Validate composition sums approximately to 1
        total = sum(self.true_composition.values())
        if abs(total - 1.0) > 0.05:
            logger.warning(
                f"Spectrum {self.spectrum_id}: composition sums to {total:.4f}, "
                "expected ~1.0"
            )

    @property
    def elements(self) -> List[str]:
        """List of elements in composition."""
        return sorted(self.true_composition.keys())

    @property
    def n_points(self) -> int:
        """Number of spectral points."""
        return len(self.wavelength_nm)

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Wavelength range (min, max) in nm."""
        return float(self.wavelength_nm.min()), float(self.wavelength_nm.max())

    def get_composition_with_uncertainty(
        self, element: str
    ) -> Tuple[float, float]:
        """
        Get composition and uncertainty for an element.

        Parameters
        ----------
        element : str
            Element symbol

        Returns
        -------
        value : float
            Mass fraction
        uncertainty : float
            1-sigma uncertainty
        """
        value = self.true_composition.get(element, 0.0)
        uncertainty = self.composition_uncertainty.get(element, 0.0)
        return value, uncertainty

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "spectrum_id": self.spectrum_id,
            "wavelength_nm": self.wavelength_nm.tolist(),
            "intensity": self.intensity.tolist(),
            "intensity_uncertainty": (
                self.intensity_uncertainty.tolist()
                if self.intensity_uncertainty is not None
                else None
            ),
            "true_composition": self.true_composition,
            "composition_uncertainty": self.composition_uncertainty,
            "conditions": self.conditions.to_dict(),
            "metadata": self.metadata.to_dict(),
            "plasma_temperature_K": self.plasma_temperature_K,
            "electron_density_cm3": self.electron_density_cm3,
            "quality_flag": self.quality_flag,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkSpectrum":
        """Create from dictionary representation."""
        return cls(
            spectrum_id=d["spectrum_id"],
            wavelength_nm=np.array(d["wavelength_nm"]),
            intensity=np.array(d["intensity"]),
            intensity_uncertainty=(
                np.array(d["intensity_uncertainty"])
                if d.get("intensity_uncertainty") is not None
                else None
            ),
            true_composition=d["true_composition"],
            composition_uncertainty=d.get("composition_uncertainty", {}),
            conditions=InstrumentalConditions.from_dict(d["conditions"]),
            metadata=SampleMetadata.from_dict(d["metadata"]),
            plasma_temperature_K=d.get("plasma_temperature_K"),
            electron_density_cm3=d.get("electron_density_cm3"),
            quality_flag=d.get("quality_flag", 0),
        )


@dataclass
class DataSplit:
    """
    A train/test or cross-validation split of benchmark spectra.

    Attributes
    ----------
    name : str
        Split identifier (e.g., "default", "fold_1")
    train_ids : List[str]
        Spectrum IDs in training set
    test_ids : List[str]
        Spectrum IDs in test set
    validation_ids : List[str], optional
        Spectrum IDs in validation set
    description : str
        Description of split rationale
    random_seed : int, optional
        Random seed used for split (for reproducibility)
    """

    name: str
    train_ids: List[str]
    test_ids: List[str]
    validation_ids: Optional[List[str]] = None
    description: str = ""
    random_seed: Optional[int] = None

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_ids)

    @property
    def test_size(self) -> int:
        """Number of test samples."""
        return len(self.test_ids)

    @property
    def validation_size(self) -> int:
        """Number of validation samples."""
        return len(self.validation_ids) if self.validation_ids else 0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "train_ids": self.train_ids,
            "test_ids": self.test_ids,
            "validation_ids": self.validation_ids,
            "description": self.description,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DataSplit":
        """Create from dictionary representation."""
        return cls(**d)


@dataclass
class BenchmarkDataset:
    """
    A complete benchmark dataset with spectra, splits, and metadata.

    This is the main container for benchmark data. It includes all spectra,
    predefined train/test splits, and dataset-level metadata for
    reproducible evaluation.

    Attributes
    ----------
    name : str
        Dataset name (e.g., "nist_steel_crm_2024")
    version : str
        Dataset version string
    spectra : List[BenchmarkSpectrum]
        List of benchmark spectra
    splits : Dict[str, DataSplit]
        Named train/test splits
    elements : List[str]
        Target elements in this dataset
    description : str
        Dataset description
    citation : str
        How to cite this dataset
    license : str
        Data license (e.g., "CC-BY-4.0")
    created_date : str
        ISO format creation date
    contributors : List[str]
        List of contributors/authors

    Example
    -------
    >>> dataset = BenchmarkDataset.from_json("nist_steel.json")
    >>> print(f"Dataset: {dataset.name}, {dataset.n_spectra} spectra")
    >>> train, test = dataset.get_split("default")
    >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """

    name: str
    version: str
    spectra: List[BenchmarkSpectrum]
    elements: List[str]
    splits: Dict[str, DataSplit] = field(default_factory=dict)
    description: str = ""
    citation: str = ""
    license: str = "CC-BY-4.0"
    created_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    contributors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Build index and validate."""
        self._id_to_spectrum: Dict[str, BenchmarkSpectrum] = {
            s.spectrum_id: s for s in self.spectra
        }

        # Validate splits reference valid spectrum IDs
        valid_ids = set(self._id_to_spectrum.keys())
        for split_name, split in self.splits.items():
            for id_ in split.train_ids + split.test_ids:
                if id_ not in valid_ids:
                    raise ValueError(
                        f"Split '{split_name}' references unknown spectrum ID: {id_}"
                    )
            if split.validation_ids:
                for id_ in split.validation_ids:
                    if id_ not in valid_ids:
                        raise ValueError(
                            f"Split '{split_name}' references unknown spectrum ID: {id_}"
                        )

    @property
    def n_spectra(self) -> int:
        """Number of spectra in dataset."""
        return len(self.spectra)

    @property
    def spectrum_ids(self) -> List[str]:
        """List of all spectrum IDs."""
        return list(self._id_to_spectrum.keys())

    def get_spectrum(self, spectrum_id: str) -> BenchmarkSpectrum:
        """
        Get spectrum by ID.

        Parameters
        ----------
        spectrum_id : str
            Spectrum identifier

        Returns
        -------
        BenchmarkSpectrum
            The requested spectrum

        Raises
        ------
        KeyError
            If spectrum ID not found
        """
        if spectrum_id not in self._id_to_spectrum:
            raise KeyError(f"Spectrum '{spectrum_id}' not found in dataset")
        return self._id_to_spectrum[spectrum_id]

    def get_split(
        self, split_name: str = "default"
    ) -> Tuple[List[BenchmarkSpectrum], List[BenchmarkSpectrum]]:
        """
        Get train/test split by name.

        Parameters
        ----------
        split_name : str
            Name of the split (default: "default")

        Returns
        -------
        train : List[BenchmarkSpectrum]
            Training spectra
        test : List[BenchmarkSpectrum]
            Test spectra

        Raises
        ------
        KeyError
            If split not found
        """
        if split_name not in self.splits:
            available = list(self.splits.keys())
            raise KeyError(
                f"Split '{split_name}' not found. Available: {available}"
            )

        split = self.splits[split_name]
        train = [self._id_to_spectrum[id_] for id_ in split.train_ids]
        test = [self._id_to_spectrum[id_] for id_ in split.test_ids]
        return train, test

    def get_split_with_validation(
        self, split_name: str = "default"
    ) -> Tuple[
        List[BenchmarkSpectrum],
        List[BenchmarkSpectrum],
        Optional[List[BenchmarkSpectrum]],
    ]:
        """
        Get train/validation/test split by name.

        Parameters
        ----------
        split_name : str
            Name of the split

        Returns
        -------
        train : List[BenchmarkSpectrum]
            Training spectra
        test : List[BenchmarkSpectrum]
            Test spectra
        validation : List[BenchmarkSpectrum] or None
            Validation spectra (if defined)
        """
        train, test = self.get_split(split_name)
        split = self.splits[split_name]

        if split.validation_ids:
            validation = [self._id_to_spectrum[id_] for id_ in split.validation_ids]
        else:
            validation = None

        return train, test, validation

    def create_random_split(
        self,
        name: str,
        train_fraction: float = 0.7,
        test_fraction: float = 0.3,
        validation_fraction: float = 0.0,
        random_seed: int = 42,
        stratify_by: Optional[str] = None,
    ) -> DataSplit:
        """
        Create a new random train/test/validation split.

        Parameters
        ----------
        name : str
            Name for the new split
        train_fraction : float
            Fraction of data for training (default: 0.7)
        test_fraction : float
            Fraction of data for testing (default: 0.3)
        validation_fraction : float
            Fraction of data for validation (default: 0.0)
        random_seed : int
            Random seed for reproducibility
        stratify_by : str, optional
            Element to stratify by (maintains composition distribution)

        Returns
        -------
        DataSplit
            The created split (also added to self.splits)
        """
        # Validate fractions
        total = train_fraction + test_fraction + validation_fraction
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Fractions must sum to 1.0, got {total}")

        rng = np.random.default_rng(random_seed)
        ids = list(self._id_to_spectrum.keys())

        if stratify_by is not None:
            # Stratified split based on composition quartiles
            compositions = [
                self._id_to_spectrum[id_].true_composition.get(stratify_by, 0.0)
                for id_ in ids
            ]
            # Create quartile bins
            quartiles = np.percentile(compositions, [25, 50, 75])
            bins = np.digitize(compositions, quartiles)

            train_ids, test_ids, val_ids = [], [], []

            for bin_idx in range(4):
                bin_ids = [id_ for id_, b in zip(ids, bins) if b == bin_idx]
                rng.shuffle(bin_ids)

                n = len(bin_ids)
                n_train = int(n * train_fraction)
                n_val = int(n * validation_fraction)

                train_ids.extend(bin_ids[:n_train])
                val_ids.extend(bin_ids[n_train : n_train + n_val])
                test_ids.extend(bin_ids[n_train + n_val :])
        else:
            # Simple random split
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(n * train_fraction)
            n_val = int(n * validation_fraction)

            train_ids = ids[:n_train]
            val_ids = ids[n_train : n_train + n_val] if validation_fraction > 0 else []
            test_ids = ids[n_train + n_val :]

        split = DataSplit(
            name=name,
            train_ids=train_ids,
            test_ids=test_ids,
            validation_ids=val_ids if val_ids else None,
            description=f"Random split with seed={random_seed}",
            random_seed=random_seed,
        )

        self.splits[name] = split
        return split

    def create_kfold_splits(
        self,
        n_folds: int = 5,
        random_seed: int = 42,
        stratify_by: Optional[str] = None,
    ) -> List[DataSplit]:
        """
        Create k-fold cross-validation splits.

        Parameters
        ----------
        n_folds : int
            Number of folds (default: 5)
        random_seed : int
            Random seed for reproducibility
        stratify_by : str, optional
            Element to stratify by

        Returns
        -------
        List[DataSplit]
            List of k DataSplit objects (also added to self.splits)
        """
        rng = np.random.default_rng(random_seed)
        ids = list(self._id_to_spectrum.keys())
        rng.shuffle(ids)

        # Create fold assignments
        fold_assignments = np.array_split(ids, n_folds)

        splits = []
        for fold_idx in range(n_folds):
            test_ids = list(fold_assignments[fold_idx])
            train_ids = []
            for i, fold_ids in enumerate(fold_assignments):
                if i != fold_idx:
                    train_ids.extend(fold_ids)

            split = DataSplit(
                name=f"fold_{fold_idx + 1}",
                train_ids=train_ids,
                test_ids=test_ids,
                description=f"Fold {fold_idx + 1} of {n_folds}-fold CV, seed={random_seed}",
                random_seed=random_seed,
            )
            self.splits[split.name] = split
            splits.append(split)

        return splits

    def filter_by_quality(self, max_flag: int = 0) -> "BenchmarkDataset":
        """
        Create a filtered dataset keeping only spectra with quality_flag <= max_flag.

        Parameters
        ----------
        max_flag : int
            Maximum quality flag to include (default: 0 = good only)

        Returns
        -------
        BenchmarkDataset
            Filtered dataset
        """
        filtered_spectra = [s for s in self.spectra if s.quality_flag <= max_flag]

        # Update splits to remove excluded spectra
        filtered_ids = {s.spectrum_id for s in filtered_spectra}
        new_splits = {}
        for name, split in self.splits.items():
            new_train = [id_ for id_ in split.train_ids if id_ in filtered_ids]
            new_test = [id_ for id_ in split.test_ids if id_ in filtered_ids]
            new_val = None
            if split.validation_ids:
                new_val = [id_ for id_ in split.validation_ids if id_ in filtered_ids]

            new_splits[name] = DataSplit(
                name=split.name,
                train_ids=new_train,
                test_ids=new_test,
                validation_ids=new_val,
                description=split.description + f" (filtered quality<={max_flag})",
                random_seed=split.random_seed,
            )

        return BenchmarkDataset(
            name=self.name,
            version=self.version,
            spectra=filtered_spectra,
            elements=self.elements,
            splits=new_splits,
            description=self.description,
            citation=self.citation,
            license=self.license,
            created_date=self.created_date,
            contributors=self.contributors,
        )

    def get_composition_matrix(
        self, spectrum_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get composition matrix for all or selected spectra.

        Parameters
        ----------
        spectrum_ids : List[str], optional
            Spectrum IDs to include (default: all)

        Returns
        -------
        compositions : np.ndarray
            Shape (n_spectra, n_elements) composition matrix
        elements : List[str]
            Element names for each column
        ids : List[str]
            Spectrum IDs for each row
        """
        if spectrum_ids is None:
            spectra = self.spectra
        else:
            spectra = [self._id_to_spectrum[id_] for id_ in spectrum_ids]

        # Use dataset's element list for consistent ordering
        n_spectra = len(spectra)
        n_elements = len(self.elements)

        compositions = np.zeros((n_spectra, n_elements))
        ids = []

        for i, spec in enumerate(spectra):
            ids.append(spec.spectrum_id)
            for j, element in enumerate(self.elements):
                compositions[i, j] = spec.true_composition.get(element, 0.0)

        return compositions, self.elements, ids

    def summary(self) -> str:
        """Generate a human-readable summary of the dataset."""
        lines = [
            f"Benchmark Dataset: {self.name} v{self.version}",
            f"  Description: {self.description}",
            f"  Spectra: {self.n_spectra}",
            f"  Elements: {', '.join(self.elements)}",
            f"  Splits: {list(self.splits.keys())}",
            f"  License: {self.license}",
        ]

        if self.citation:
            lines.append(f"  Citation: {self.citation}")

        # Composition statistics
        comp_matrix, elements, _ = self.get_composition_matrix()
        lines.append("  Composition ranges:")
        for i, elem in enumerate(elements):
            col = comp_matrix[:, i]
            if col.max() > 0:
                lines.append(
                    f"    {elem}: {col.min()*100:.2f}% - {col.max()*100:.2f}%"
                )

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "spectra": [s.to_dict() for s in self.spectra],
            "elements": self.elements,
            "splits": {name: split.to_dict() for name, split in self.splits.items()},
            "description": self.description,
            "citation": self.citation,
            "license": self.license,
            "created_date": self.created_date,
            "contributors": self.contributors,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkDataset":
        """Create from dictionary representation."""
        spectra = [BenchmarkSpectrum.from_dict(s) for s in d["spectra"]]
        splits = {
            name: DataSplit.from_dict(split)
            for name, split in d.get("splits", {}).items()
        }

        return cls(
            name=d["name"],
            version=d["version"],
            spectra=spectra,
            elements=d["elements"],
            splits=splits,
            description=d.get("description", ""),
            citation=d.get("citation", ""),
            license=d.get("license", "CC-BY-4.0"),
            created_date=d.get("created_date", ""),
            contributors=d.get("contributors", []),
        )
