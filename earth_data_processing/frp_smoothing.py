#!/usr/bin/env python3
"""
frp_smoothing.py
Spread each FRP “spike” via a Gaussian kernel, and convert all NaNs → 0.
"""
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

# === User Settings ===
INPUT_FILE  = "cropped_fire_pred_dataset.nc"
OUTPUT_FILE = "spread_fire_data.nc"
RADIUS_KM   = 10.0
# ======================

def main():
    ds = xr.open_dataset(INPUT_FILE)
    if "frp" not in ds:
        sys.exit("Error: 'frp' variable not found.")

    frp = ds["frp"]
    total = frp.size
    n_nan = int(frp.isnull().sum().item())
    print(f"Original NaNs: {n_nan}/{total} = {100*n_nan/total:.2f}%")

    # 1) Turn NaNs → 0 so they don't “block” the blur
    frp_filled = frp.fillna(0)

    # 2) Compute σ in grid‑cells
    lat = ds["lat"]; lon = ds["lon"]
    dlat = float(lat.diff("lat").mean())
    dlon = float(lon.diff("lon").mean())
    mean_lat = float(lat.mean())
    r_deg = RADIUS_KM * 1e3 / 111e3
    sigma_lat = r_deg / dlat
    sigma_lon = r_deg / (dlon * np.cos(np.deg2rad(mean_lat)))
    print(f"Sigma (cells): lat={sigma_lat:.2f}, lon={sigma_lon:.2f}")

    # 3) Gaussian blur (no time‑blurring)
    if "time" in frp_filled.dims:
        core_dims = [["lat","lon"]]
        smoothed = xr.apply_ufunc(
            lambda arr: gaussian_filter(arr,
                                        sigma=(sigma_lat, sigma_lon),
                                        mode="constant", cval=0),
            frp_filled,
            input_core_dims=core_dims,
            output_core_dims=core_dims,
            vectorize=True,
        )
    else:
        smoothed = xr.DataArray(
            gaussian_filter(frp_filled.values,
                            sigma=(sigma_lat, sigma_lon),
                            mode="constant", cval=0),
            coords=frp_filled.coords,
            dims=frp_filled.dims,
        )

    # 4) Ensure no NaNs remain
    frp_spread = smoothed.fillna(0)

    # 5) Replace original and save
    ds["frp"] = frp_spread
    ds.to_netcdf(OUTPUT_FILE)
    print(f"Finished. Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
