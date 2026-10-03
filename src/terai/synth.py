"""Calibrated synthetic data used by the network simulations.

After examining the original datasets we derived a synthetic dataset whose statistics
are fitted to the real records available to us.  The synthetic data drive *network*
workloads only; every forecasting/estimation result is computed on real data.

Components
----------
1. Agricultural IoT flow records (Farm-Flow-calibrated): per traffic class (Normal and
   eight attack types), a Gaussian copula over log-transformed flow features fitted to
   the 545 unique real Farm-Flow flow records that were available.
2. Telemetry timing: report inter-arrival times fitted (log-normal) to the 99 real
   intervals of the Mendeley in-field IoT log; telemetry sensor channels and payload
   follow the same log.
3. Satellite product payloads: raster sizes computed from the real Landsat 8/9
   Collection-2 MTL metadata (lines x samples x bands x bytes) and the Sentinel-1 IW GRD
   10 m grid applied to the district boxes.
4. Fidelity checks: per-feature two-sample Kolmogorov-Smirnov statistics, correlation
   matrix distance, and a classifier two-sample test (real vs synthetic ROC-AUC).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

FLOW_FEATURES = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
                 "orig_ip_bytes", "resp_ip_bytes", "fwd_pkts_payload.avg", "flow_iat.avg",
                 "flow_pkts_per_sec"]


def load_farmflow():
    d = pd.read_csv(os.path.join(C.DATA_RAW, "farmflow", "farmflow_available_raw_flows.csv"), low_memory=False)
    for c in FLOW_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[FLOW_FEATURES] = d[FLOW_FEATURES].fillna(0.0).clip(lower=0.0)
    return d


class GaussianCopulaFlows:
    """Per-class Gaussian copula with empirical marginals on log1p(features)."""

    def __init__(self, df: pd.DataFrame, seed=0):
        self.rng = np.random.default_rng(seed)
        self.classes = sorted(df["traffic"].unique())
        self.counts = df["traffic"].value_counts().to_dict()
        self.models = {}
        # features whose real values are all integers (counts, bytes) are rounded after sampling
        vals = df[FLOW_FEATURES].values.astype(float)
        self.integer = [bool(np.all(np.abs(vals[:, k] - np.round(vals[:, k])) < 1e-9)) for k in range(len(FLOW_FEATURES))]
        for c in self.classes:
            x = np.log1p(df.loc[df.traffic == c, FLOW_FEATURES].values.astype(float))
            n = len(x)
            # normal scores of the empirical ranks (ties broken randomly)
            u = (stats.rankdata(x + 1e-9 * self.rng.standard_normal(x.shape), axis=0) - 0.5) / n
            z = stats.norm.ppf(u)
            corr = np.corrcoef(z, rowvar=False) if n > 2 else np.eye(x.shape[1])
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            # nearest positive-definite correction
            w, V = np.linalg.eigh(corr)
            corr = (V * np.clip(w, 1e-6, None)) @ V.T
            dd = np.sqrt(np.diag(corr)); corr = corr / np.outer(dd, dd)
            self.models[c] = (np.sort(x, axis=0), corr)

    def sample(self, n_per_class: dict):
        rows = []
        for c, n in n_per_class.items():
            xs, corr = self.models[c]
            z = self.rng.multivariate_normal(np.zeros(len(FLOW_FEATURES)), corr, size=n)
            u = stats.norm.cdf(z)
            # inverse empirical CDF with linear interpolation between order statistics
            q = np.linspace(0, 1, xs.shape[0])
            x = np.column_stack([np.interp(u[:, k], q, xs[:, k]) for k in range(xs.shape[1])])
            v = np.expm1(x)
            for k, isint in enumerate(self.integer):
                if isint:
                    v[:, k] = np.round(v[:, k])
            df = pd.DataFrame(v, columns=FLOW_FEATURES)
            df["traffic"] = c
            rows.append(df)
        return pd.concat(rows, ignore_index=True)


def fidelity(real: pd.DataFrame, syn: pd.DataFrame, seed=0):
    """KS statistic per feature, correlation distance, classifier two-sample AUC."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    out = {"ks": {}, "ks_p": {}}
    for f in FLOW_FEATURES:
        r = stats.ks_2samp(np.log1p(real[f]), np.log1p(syn[f]))
        out["ks"][f] = float(r.statistic); out["ks_p"][f] = float(r.pvalue)
    cr = np.corrcoef(np.log1p(real[FLOW_FEATURES].values), rowvar=False)
    cs = np.corrcoef(np.log1p(syn[FLOW_FEATURES].values), rowvar=False)
    out["corr_frobenius"] = float(np.linalg.norm(np.nan_to_num(cr) - np.nan_to_num(cs)) /
                                  np.linalg.norm(np.nan_to_num(cr)))
    # classifier two-sample test on a balanced sample
    n = min(len(real), len(syn))
    rs = real.sample(n, random_state=seed); ss = syn.sample(n, random_state=seed)
    X = np.log1p(np.vstack([rs[FLOW_FEATURES].values, ss[FLOW_FEATURES].values]))
    y = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(RandomForestClassifier(200, random_state=seed), X, y, cv=5, method="predict_proba")[:, 1]
    out["c2st_auc"] = float(roc_auc_score(y, p))
    return out


def mendeley_intervals():
    """Real inter-arrival times (s) of the Mendeley IoT log (matched by entry id)."""
    f = os.path.join(C.DATA_RAW, "mendeley", "Soil Moisture.xlsx")
    d = pd.read_excel(f, header=1)
    d.columns = ["ts", "entry_id", "value"]
    ts = pd.to_datetime(d["ts"].astype(str).str[:19])
    d = d.assign(ts=ts).sort_values("entry_id")
    dt = d["ts"].diff().dt.total_seconds().dropna().values
    return dt[dt > 0]


def fit_lognormal(x):
    s, loc, scale = stats.lognorm.fit(x, floc=0)
    ks = stats.kstest(x, "lognorm", args=(s, loc, scale))
    return {"sigma": float(s), "scale": float(scale), "median": float(scale),
            "mean": float(scale * np.exp(s ** 2 / 2)), "ks": float(ks.statistic), "ks_p": float(ks.pvalue)}


def landsat_payloads():
    """Raster payloads of the real Landsat scenes from their MTL metadata."""
    out = []
    for f in sorted(glob.glob(os.path.join(C.DATA_RAW, "landsat_mtl", "*_MTL.txt"))):
        meta = {}
        for line in open(f):
            if "=" in line:
                k, v = [x.strip() for x in line.split("=", 1)]
                meta.setdefault(k, v.strip('"'))
        lines, samples = int(meta["REFLECTIVE_LINES"]), int(meta["REFLECTIVE_SAMPLES"])
        cell = float(meta["GRID_CELL_SIZE_REFLECTIVE"])
        sr_bands = 7
        nbytes = lines * samples * sr_bands * 2               # UINT16 surface reflectance
        out.append({"product": meta["LANDSAT_PRODUCT_ID"], "date": meta["DATE_ACQUIRED"],
                    "cloud_cover": float(meta["CLOUD_COVER"]), "lines": lines, "samples": samples,
                    "cell_m": cell, "sr_bytes": nbytes,
                    "bytes_per_km2_per_band": 1e6 / cell ** 2 * 2})
    return out


def box_area_km2(lat, half_deg=0.1):
    return (2 * half_deg * 111.32) * (2 * half_deg * 111.32 * np.cos(np.radians(lat)))


def sentinel1_subset_bytes(lat, pixel_m=10.0, pols=2, bytes_per=2, half_deg=0.1):
    return box_area_km2(lat, half_deg) * 1e6 / pixel_m ** 2 * pols * bytes_per
