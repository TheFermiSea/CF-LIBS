import os
os.environ["JAX_PLATFORMS"] = "cpu"
import xarray as xr
import h5py
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.correlation_identifier import CorrelationIdentifier

db_path = "ASD_da/libs_production.db"
db = AtomicDatabase(db_path)

def run_dataset(name, path, expected):
    print("\nProcessing " + name + "...")
    if path.endswith('.nc'):
        ds = xr.open_dataset(path)
        data = ds['__xarray_dataarray_variable__'].values
        wl = ds.coords['Wavelength'].values
    else:
        with h5py.File(path, 'r') as f:
            wl = f['Wavelength'][:]
            data = f['__xarray_dataarray_variable__'][:]
    
    if data.ndim == 3:
        spec = data[data.shape[0]//2, data.shape[1]//2, :]
    else:
        spec = data
    
    elements = ["Fe", "Ni", "Cr", "Mn", "Cu", "Ti", "Si"]
    identifier = CorrelationIdentifier(db, elements=elements)
    result = identifier.identify(wl, spec, mode="classic")
    
    for e in result.all_elements:
        status = "DETECTED" if e.detected else "NOT DETECTED"
        match = "*" if (e.detected and e.element in expected) or (not e.detected and e.element not in expected) else ""
        print("{:<4}: {:<15} score={:.3f} {}".format(e.element, status, e.score, match))

run_dataset("Fe_245nm", "data/Fe_245nm", ["Fe"])
run_dataset("Ni_245nm", "data/Ni_245nm", ["Ni"])
run_dataset("steel_245nm", "data/steel_245nm.nc", ["Fe", "Cr", "Ni", "Mn"])