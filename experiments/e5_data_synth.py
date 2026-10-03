"""E5 - Real-data facts quoted in the paper and fidelity of the calibrated synthetic data.
Output: results/raw/e5_data_synth.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, synth as SY  # noqa: E402


def main():
    P = D.build_panel()
    out = {"data": D.dataset_summary(P)}
    out["data"]["s1_valid_rows"] = int(sum(out["data"]["s1_valid_acq"].values()))
    raw = pd.read_csv(os.path.join(C.DATA_RAW, "imd_power_s1", "uttarakhand_sentinel1_vvvh.csv"))
    out["data"]["s1_empty_rows"] = int(raw[["VV", "VH"]].isna().any(axis=1).sum())
    q = pd.read_csv(glob.glob(os.path.join(C.DATA_RAW, "smap", "*AM-retrieval-qual-flag-Statistics-QA.csv"))[0]).fillna(0)
    q = q[q["7"] < 722]
    out["data"]["smap_overpass_days_am"] = int(len(q))
    out["data"]["smap_recommended_frac"] = float((q["0"] / 722).mean())
    out["data"]["smap_retrieved_frac"] = float(((q["0"] + q["1"] + q["9"]) / 722).mean())
    imd = pd.read_csv(os.path.join(C.DATA_RAW, "imd_power_s1", "uttarakhand_imd_gridded.csv"))
    out["data"]["imd_rows"] = int(len(imd))
    # ---------------- synthetic flows
    d = SY.load_farmflow()
    out["farmflow_records"] = int(len(d))
    out["farmflow_class_counts"] = {k: int(v) for k, v in d.traffic.value_counts().items()}
    g = SY.GaussianCopulaFlows(d, seed=0)
    out["integer_features"] = [f for f, i in zip(SY.FLOW_FEATURES, g.integer) if i]
    fids = []
    for s in range(5):
        g = SY.GaussianCopulaFlows(d, seed=s)
        fids.append(SY.fidelity(d, g.sample(dict(g.counts)), seed=s))
    out["fidelity"] = {"ks": {f: float(np.mean([x["ks"][f] for x in fids])) for f in SY.FLOW_FEATURES},
                       "ks_p": {f: float(np.mean([x["ks_p"][f] for x in fids])) for f in SY.FLOW_FEATURES},
                       "corr_frobenius": float(np.mean([x["corr_frobenius"] for x in fids])),
                       "c2st_auc": float(np.mean([x["c2st_auc"] for x in fids])),
                       "c2st_auc_sd": float(np.std([x["c2st_auc"] for x in fids]))}
    iv = SY.mendeley_intervals()
    out["mendeley"] = {"n_intervals": int(len(iv)), "median_s": float(np.median(iv)), "mean_s": float(np.mean(iv)),
                       "min_s": float(iv.min()), "max_s": float(iv.max())}
    syn = SY.GaussianCopulaFlows(d, 0).sample({"Normal": 20000})
    out["telemetry_bytes"] = {"real_median": float(d[d.traffic == "Normal"].orig_ip_bytes.median()),
                              "synthetic_median": float(syn.orig_ip_bytes.median())}
    out["landsat"] = SY.landsat_payloads()
    out["s1_subset_mb"] = {dn: float(SY.sentinel1_subset_bytes(lat) / 1e6) for dn, (lat, lon) in C.DISTRICT_LATLON.items()}
    os.makedirs(C.RES_RAW, exist_ok=True)
    json.dump(out, open(os.path.join(C.RES_RAW, "e5_data_synth.json"), "w"), indent=1)
    print(json.dumps(out["fidelity"], indent=1))
    print(out["data"])


if __name__ == "__main__":
    main()
