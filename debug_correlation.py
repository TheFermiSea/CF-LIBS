import os
os.environ["JAX_PLATFORMS"] = "cpu"

import xarray as xr
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.correlation_identifier import CorrelationIdentifier

db_path = "ASD_da/libs_production.db"
db = AtomicDatabase(db_path)

ds = xr.open_dataset('data/steel_245nm.nc')
data = ds['__xarray_dataarray_variable__'].values
wavelength = ds.coords['Wavelength'].values
spec = data[data.shape[0]//2, data.shape[1]//2, :]

elements = ["Fe", "Cr", "Ni", "Mn", "Cu", "Ti", "Si"]
expected = ["Fe", "Cr", "Ni", "Mn"]

print("Running Correlation (classic mode)")
identifier = CorrelationIdentifier(db, elements=elements)
result = identifier.identify(wavelength, spec, mode="classic")

print("\nResults for steel_245nm (Correlation):")
for e in result.all_elements:
    status = "DETECTED" if e.detected else "NOT DETECTED"
    print(f"{e.element:<4}: {status:<15} score={e.score:.3f}")