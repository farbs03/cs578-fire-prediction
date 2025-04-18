#!/usr/bin/env python3
"""
check_bad.py

Check  dataset for NaN and Inf values, report basic statistics per variable,
and surface any potential bad data.

"""

import sys
import numpy as np
import xarray as xr

def main():
    input_path = "C:\\Users\\Jay\\Documents\\cs578-fire-prediction\\earth_data_processing\\spread_fire_data.nc"
    try:
        ds = xr.open_dataset(input_path)
    except Exception as e:
        print(f"Error opening '{input_path}': {e}")
        sys.exit(1)

    print(f"Opened dataset: {input_path}")
    print(f"Dimensions: {ds.dims}\n")

    for var in ds.data_vars:
        data = ds[var]
        arr = data.values
        total = arr.size

        # Count NaNs and Infs
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())

        # Compute finite min/max
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin = vmax = float("nan")

        print(f"Variable: '{var}'")
        print(f"  dtype         : {data.dtype}")
        print(f"  shape         : {data.shape}")
        print(f"  NaNs          : {n_nan} ({100 * n_nan / total:.2f}%)")
        print(f"  Infs          : {n_inf} ({100 * n_inf / total:.2f}%)")
        print(f"  min (finite)  : {vmin}")
        print(f"  max (finite)  : {vmax}\n")

    ds.close()

if __name__ == "__main__":
    main()
