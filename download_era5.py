"""Download daily maximum 10 m wind over Ireland from ERA5, run sanity checks, save one file.

Reads hourly ERA5 straight from ECMWF's ARCO Zarr store (geo-chunked: built for long time
series over a small area), so there is no CDS queue: the whole 1985-2024 record takes ~2 min.
Needs your CDS API key in ~/.cdsapirc (see README). The ARCO service is in beta.

Output: data/era5_ireland_wind_daily_max.nc with variables
  gust  (time, latitude, longitude)  daily max 10 m wind gust [m/s]
  speed (time, latitude, longitude)  daily max 10 m wind speed [m/s], from hourly u10, v10
Box: lat 49.75-57.5, lon -12.75 to -5.0 = 32 x 32 points at 0.25 deg. Days are UTC.
Latitude runs south -> north (49.75 ... 57.5): use origin="lower" in imshow.
Values are in physical units; normalisation happens in dataset.py with training-set stats.

    python download_era5.py                       # 1985-2024
    python download_era5.py --start 2024 --end 2024 --out data/test.nc
"""
import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr

ARCO_URL = "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
BOX = dict(latitude=slice(49.75, 57.5), longitude=slice(-12.75, -5.0))


def cds_key():
    lines = (Path.home() / ".cdsapirc").read_text().splitlines()
    return next(l.split(":", 1)[1].strip() for l in lines if l.startswith("key"))


def daily_max(hourly):
    return hourly.resample(time="1D").max()


def build(start, end):
    ds = xr.open_zarr(ARCO_URL, consolidated=True,
                      storage_options={"headers": {"Authorization": f"Bearer {cds_key()}"}})
    ds = ds.sel(**BOX)

    out = {}
    t = time.time()
    # fg10 stamped T is the max over (T-1h, T], so shift back 1 h before grouping into days.
    # This reproduces the CDS daily-statistics product exactly (checked on Jan 2024).
    gust = ds["fg10"].sel(time=slice(f"{start}-01-01T01", f"{end + 1}-01-01T00")).load()
    out["gust"] = daily_max(gust.assign_coords(time=gust.time - np.timedelta64(1, "h")))
    print(f"gust done in {time.time() - t:.0f}s", flush=True)

    t = time.time()
    hours = slice(f"{start}-01-01T00", f"{end}-12-31T23")  # instantaneous: stamps are the time itself
    u, v = ds["u10"].sel(time=hours).load(), ds["v10"].sel(time=hours).load()
    out["speed"] = daily_max(np.hypot(u, v))  # speed per hour first, then max: max|u,v| != |max u, max v|
    print(f"speed done in {time.time() - t:.0f}s", flush=True)

    ds = xr.Dataset(out).astype("float32")
    ds["gust"].attrs = {"long_name": "daily max 10 m wind gust", "units": "m s-1"}
    ds["speed"].attrs = {"long_name": "daily max 10 m wind speed (from hourly u10, v10)", "units": "m s-1"}
    return ds


def sanity_check(ds):
    checks = {
        "grid is 32x32": ds.sizes["latitude"] == 32 and ds.sizes["longitude"] == 32,
        "no NaNs": not any(bool(ds[v].isnull().any()) for v in ds.data_vars),
        "days are contiguous": bool((ds.time.diff("time") == np.timedelta64(1, "D")).all()),
        "values in 0-80 m/s": all(float(ds[v].min()) >= 0 and float(ds[v].max()) < 80 for v in ds.data_vars),
        "gust >= speed on >99% of pixels": float((ds.gust >= ds.speed).mean()) > 0.99,
    }
    print(f"{ds.sizes['time']} days, {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
    for v in ds.data_vars:
        print(f"  {v}: mean {float(ds[v].mean()):.1f}, max {float(ds[v].max()):.1f} m/s")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=1985)
    p.add_argument("--end", type=int, default=2024)
    p.add_argument("--out", type=Path, default=Path("data/era5_ireland_wind_daily_max.nc"))
    args = p.parse_args()

    ds = build(args.start, args.end)
    ok = sanity_check(ds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # store as int16 at 0.01 m/s + zlib: 115 MB -> 33 MB, so it fits in git; loads back as float32
    enc = {v: {"dtype": "int16", "scale_factor": np.float32(0.01), "_FillValue": np.int16(-32768),
               "zlib": True, "complevel": 5} for v in ds.data_vars}
    ds.to_netcdf(args.out, encoding=enc)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MB)" + ("" if ok else "  -- some checks FAILED"))


if __name__ == "__main__":
    main()
