import os
os.environ["JAX_PLATFORMS"] = "cpu"

import xarray as xr
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.alias_identifier import ALIASIdentifier

db_path = "ASD_da/libs_production.db"
db = AtomicDatabase(db_path)

ds = xr.open_dataset('data/steel_245nm.nc')
data = ds['__xarray_dataarray_variable__'].values
wavelength = ds.coords['Wavelength'].values
spec = data[data.shape[0]//2, data.shape[1]//2, :]

elements = ["Fe", "Cr", "Ni", "Mn", "Cu", "Ti", "Si"]
expected = ["Fe", "Cr", "Ni", "Mn"]

print("Running ALIAS with intensity_threshold_factor=3.0")
identifier = ALIASIdentifier(db, elements=elements, intensity_threshold_factor=3.0)
result = identifier.identify(wavelength, spec)

print("\nDetailed results for steel_245nm:")
for e in result.all_elements:
    m = e.metadata
    print(f"{e.element:<4}: score={e.score:.3f}, k_sim={m['k_sim']:.3f}, k_rate={m['k_rate']:.3f}, k_shift={m['k_shift']:.3f}, N_X={m['N_X']}")