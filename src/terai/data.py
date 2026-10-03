"""Real-data loading and the daily multimodal panel.

All inputs are the real files in data/raw.  Nothing in this module is synthetic.

Panel conventions
-----------------
* Day index ``d`` counts days from PANEL_START.
* For every modality m and district i we keep an *acquisition calendar*: the set of
  days on which a new product exists (daily for IMD/POWER, irregular for Sentinel-1
  and SMAP).  A product acquired on day d becomes available at the hub on day
  d + LATENCY[m].
* ``feat[m]`` has shape (n_days, n_districts, f_m) and holds the feature vector of the
  product acquired on day d (NaN where no acquisition exists).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

F_NAMES = {
    "IMD": ["log_rain_d", "log_rain_3d", "log_rain_7d", "tmax", "tmin"],
    "POWER": ["t2m", "dtr", "rh2m", "log_prec", "sw_down", "ws2m"],
    "S1": ["vv_db", "vh_db", "vv_minus_vh", "dvv"],
    "SMAP": ["sm", "vwc", "tsurf_c", "retrieved_frac"],
}


@dataclass
class Panel:
    days: pd.DatetimeIndex                 # calendar of the panel
    feat: dict                             # m -> array (n_days, n_dist, f_m)
    avail: dict                            # m -> bool array (n_days, n_dist): acquisition exists
    rain_mm: np.ndarray                    # (n_days, n_dist) IMD daily rain (mm)
    rain_event: np.ndarray                 # (n_days, n_dist) rain >= threshold (0/1)
    sm_target: np.ndarray                  # (n_days,) regional SMAP soil moisture (NaN if none)
    era5_sm: np.ndarray                    # (n_days,) ERA5-Land swvl1 at Dehradun cell (NaN if none)
    norm: dict                             # m -> (mean, std) computed on the training period
    power_identical_pairs: list            # districts whose POWER series are identical

    @property
    def n_days(self):
        return len(self.days)

    def day_index(self, date) -> int:
        return int((pd.Timestamp(date) - self.days[0]).days)


def _read_imd() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(C.DATA_RAW, "imd_power_s1", "uttarakhand_imd_gridded.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def _read_power() -> dict:
    out = {}
    for f in glob.glob(os.path.join(C.DATA_RAW, "imd_power_s1", "*_power.csv")):
        d = pd.read_csv(f)
        d["date"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d")
        out[d["district"].iloc[0]] = d.set_index("date")
    return out


def _read_s1() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(C.DATA_RAW, "imd_power_s1", "uttarakhand_sentinel1_vvvh.csv"))
    df["date"] = pd.to_datetime(df["date"])
    n0 = len(df)
    df = df.dropna(subset=["VV", "VH"]).copy()
    df.attrs["n_raw"] = n0
    # several districts have two acquisitions on one day (ascending/descending passes): average
    df = df.groupby(["district", "date"], as_index=False)[["VV", "VH"]].mean()
    return df


def _read_smap() -> pd.DataFrame:
    f = glob.glob(os.path.join(C.DATA_RAW, "smap", "*Statistics.csv"))
    f = [x for x in f if "QA" not in x][0]
    s = pd.read_csv(f)
    s["Date"] = pd.to_datetime(s["Date"])
    piv = {}
    for key, name in [("AM_soil_moisture", "sm_am"), ("PM_soil_moisture_pm", "sm_pm"),
                      ("AM_vegetation_water_content", "vwc"), ("AM_surface_temperature", "tsurf")]:
        g = s[s["Dataset"].str.endswith(key)].set_index("Date")
        piv[name] = g["Mean"]
        if name in ("sm_am", "sm_pm"):
            piv[name + "_count"] = g["Count"]
    return pd.DataFrame(piv)


def _read_era5_sm() -> pd.Series:
    import xarray as xr
    files = sorted(glob.glob(os.path.join(C.DATA_RAW, "era5", "*.nc")))
    ds = xr.concat([xr.open_dataset(p).load() for p in files], dim="valid_time")
    c = ds.sel(latitude=30.3, longitude=78.0, method="nearest")
    return c["swvl1"].to_series().resample("D").mean()


def build_panel() -> Panel:
    days = pd.date_range(C.PANEL_START, C.PANEL_END, freq="D")
    nd, nn = len(days), len(C.DISTRICTS)
    feat, avail = {}, {}

    # --- IMD gridded rainfall and temperature (daily, all districts) -----------------
    imd = _read_imd()
    rain_full = imd.pivot_table(index="date", columns="district", values="rain_mm")[C.DISTRICTS]
    tmax = imd.pivot_table(index="date", columns="district", values="tmax_C")[C.DISTRICTS]
    tmin = imd.pivot_table(index="date", columns="district", values="tmin_C")[C.DISTRICTS]
    r3 = rain_full.rolling(3).sum()
    r7 = rain_full.rolling(7).sum()
    f = np.stack([np.log1p(rain_full.reindex(days).values), np.log1p(r3.reindex(days).values),
                  np.log1p(r7.reindex(days).values), tmax.reindex(days).values,
                  tmin.reindex(days).values], axis=-1)
    feat["IMD"] = f.astype(np.float32)
    avail["IMD"] = ~np.isnan(f).any(-1)
    rain_mm = rain_full.reindex(days).values.astype(np.float32)

    # --- NASA POWER (daily; Haridwar not provided) -------------------------------------
    pw = _read_power()
    f = np.full((nd, nn, 6), np.nan, np.float32)
    for j, dname in enumerate(C.DISTRICTS):
        if dname not in pw:
            continue
        p = pw[dname].reindex(days)
        f[:, j, :] = np.stack([p.T2M, p.T2M_MAX - p.T2M_MIN, p.RH2M, np.log1p(p.PRECTOTCORR),
                               p.ALLSKY_SFC_SW_DWN, p.WS2M], -1)
    feat["POWER"] = f
    avail["POWER"] = ~np.isnan(f).any(-1)
    cols = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS2M"]
    ident = []
    names = sorted(pw)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if np.array_equal(pw[names[a]][cols].values, pw[names[b]][cols].values):
                ident.append((names[a], names[b]))

    # --- Sentinel-1 VV/VH point extractions (irregular) --------------------------------
    s1 = _read_s1()
    f = np.full((nd, nn, 4), np.nan, np.float32)
    for j, dname in enumerate(C.DISTRICTS):
        g = s1[s1.district == dname].sort_values("date")
        g = g[(g.date >= days[0]) & (g.date <= days[-1])]
        dvv = g.VV.diff().fillna(0.0).values
        idx = ((g.date - days[0]).dt.days).values
        f[idx, j, 0] = g.VV.values
        f[idx, j, 1] = g.VH.values
        f[idx, j, 2] = (g.VV - g.VH).values
        f[idx, j, 3] = dvv
    feat["S1"] = f
    avail["S1"] = ~np.isnan(f).any(-1)

    # --- SMAP enhanced L3 (regional statistics, irregular) -----------------------------
    sm = _read_smap().reindex(days)
    sm_mean = sm[["sm_am", "sm_pm"]].mean(axis=1)
    frac = (sm[["sm_am_count", "sm_pm_count"]].max(axis=1) / 722.0)
    vwc = sm["vwc"].ffill()
    tsurf = (sm["tsurf"] - 273.15).ffill()
    f1 = np.stack([sm_mean.values, vwc.values, tsurf.values, frac.values], -1).astype(np.float32)
    has = ~np.isnan(sm_mean.values)
    f1[~has] = np.nan
    # regional product: replicate across the district axis (one product, shared)
    feat["SMAP"] = np.repeat(f1[:, None, :], nn, axis=1)
    avail["SMAP"] = np.repeat(has[:, None], nn, axis=1)
    sm_target = sm_mean.values.astype(np.float32)

    # --- ERA5-Land (validation only) ----------------------------------------------------
    era5 = _read_era5_sm().reindex(days).values.astype(np.float32)

    rain_event = (rain_mm >= C.RAIN_EVENT_MM).astype(np.float32)

    # --- normalisation statistics on the training period only ---------------------------
    tr = (days >= pd.Timestamp(C.PANEL_START)) & (days <= pd.Timestamp(C.TRAIN_END))
    norm = {}
    for m in C.MODALITIES:
        x = feat[m][tr].reshape(-1, feat[m].shape[-1])
        mu = np.nanmean(x, 0)
        sd = np.nanstd(x, 0) + 1e-6
        norm[m] = (mu.astype(np.float32), sd.astype(np.float32))

    return Panel(days=days, feat=feat, avail=avail, rain_mm=rain_mm, rain_event=rain_event,
                 sm_target=sm_target, era5_sm=era5, norm=norm, power_identical_pairs=ident)


def held_acquisition(panel: Panel, m: str, j: int, t: int, refresh_day: int) -> int:
    """Newest acquisition day of modality m for district j that was available when the
    agent last refreshed on decision day ``refresh_day`` (or -1 if none)."""
    lim = refresh_day - C.LATENCY[m]
    if lim < 0:
        return -1
    a = panel.avail[m][: lim + 1, j]
    idx = np.flatnonzero(a)
    return int(idx[-1]) if len(idx) else -1


def latest_available(panel: Panel, m: str) -> np.ndarray:
    """L[t, j] = newest acquisition day of modality m available on decision day t (-1 if none)."""
    nd, nn = panel.avail[m].shape
    out = np.full((nd, nn), -1, dtype=np.int32)
    for j in range(nn):
        last = -1
        for t in range(nd):
            d = t - C.LATENCY[m]
            if d >= 0 and panel.avail[m][d, j]:
                last = d
            out[t, j] = last
    return out


def dataset_summary(panel: Panel) -> dict:
    """Facts about the real data that are quoted in the manuscript."""
    s1 = _read_s1()
    days = panel.days
    out = {
        "panel_days": int(panel.n_days),
        "smap_days_any": int(np.isfinite(panel.sm_target).sum()),
        "s1_raw_rows": int(pd.read_csv(os.path.join(C.DATA_RAW, "imd_power_s1",
                                                    "uttarakhand_sentinel1_vvvh.csv")).shape[0]),
        "s1_valid_acq": {d: int(panel.avail["S1"][:, j].sum()) for j, d in enumerate(C.DISTRICTS)},
        "power_identical_pairs": panel.power_identical_pairs,
        "rain_event_rate": {d: float(np.nanmean(panel.rain_event[:, j])) for j, d in enumerate(C.DISTRICTS)},
        "era5_smap_pearson": float(pd.Series(panel.era5_sm).corr(pd.Series(panel.sm_target))),
        "era5_smap_n": int((np.isfinite(panel.era5_sm) & np.isfinite(panel.sm_target)).sum()),
    }
    return out
