"""E2 - Measured value of freshness of every modality (real data, deployed estimator).

For each modality m and staleness a, the held product of m is the newest acquisition
at least a days old (all other modalities fresh); we report the RMSE increase over the
all-fresh estimate on the 2024 test days and on the training days.  We also measure the
value of refreshing one district's IMD report alone versus all districts jointly, and
the rank correlation between the innovation feature used by the agents and the actual
reduction of the squared error.
Output: results/raw/e2_voi.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, env as E, forecast as F  # noqa: E402

AGES = [1, 2, 3, 5, 7, 10, 14, 21, 30]


def stale_held(S, days, fresh, m, a, districts=None):
    h = {k: v.copy() for k, v in fresh.items()}
    js = range(len(C.DISTRICTS)) if districts is None else districts
    for b, t in enumerate(days):
        for j in js:
            acq = S.acq[m][j]
            c = acq[acq <= t - a]
            h[m][b, j] = c[-1] if len(c) else -1
    if m == "SMAP":
        h[m][:] = h[m][:, :1]
    return h


def main():
    P = D.build_panel(); S = F.Sampler(P)
    est = E.load_estimators(S)
    f = lambda t, h: np.mean([e(t, h) for e in est], 0)
    out = {"ages": AGES}
    for period, (a0, a1) in {"test": (C.TEST_START, C.TEST_END), "train": (C.TRAIN_START, C.TRAIN_END)}.items():
        days = S.day_range(a0, a1)
        y = P.sm_target[days]; ok = np.isfinite(y)
        fresh = S.fresh_held(days)
        base = f(days, fresh)
        r0 = float(np.sqrt(((base[ok] - y[ok]) ** 2).mean()))
        res = {"rmse_fresh": r0, "curves": {}}
        for m in C.MODALITIES:
            res["curves"][m] = []
            for a in AGES:
                a_eff = max(a, C.LATENCY[m])
                p = f(days, stale_held(S, days, fresh, m, a_eff))
                res["curves"][m].append(float(np.sqrt(((p[ok] - y[ok]) ** 2).mean())) - r0)
        # all district reports (IMD+POWER+S1) stale together
        res["curves"]["all_district"] = []
        for a in AGES:
            h = fresh
            for m in ["IMD", "POWER", "S1"]:
                h = stale_held(S, days, h, m, max(a, C.LATENCY[m]))
            p = f(days, h)
            res["curves"]["all_district"].append(float(np.sqrt(((p[ok] - y[ok]) ** 2).mean())) - r0)
        # single-district IMD vs all-district IMD at 7 days
        single = {}
        for j, dn in enumerate(C.DISTRICTS):
            p = f(days, stale_held(S, days, fresh, "IMD", 7, [j]))
            single[dn] = float(np.sqrt(((p[ok] - y[ok]) ** 2).mean())) - r0
        res["imd7_single_district"] = single
        res["imd7_all_districts"] = res["curves"]["IMD"][AGES.index(7)]
        out[period] = res
    # innovation as a VoI proxy (test days): innovation of district j's IMD report when
    # all IMD reports are 3 days stale vs actual squared-error reduction from delivering it
    days = S.day_range(C.TEST_START, C.TEST_END)
    y = P.sm_target[days]; ok = np.isfinite(y)
    fresh = S.fresh_held(days)
    stale = stale_held(S, days, fresh, "IMD", 4)
    base = f(days, stale)
    innov, gain = [], []
    for j in range(len(C.DISTRICTS)):
        h = {k: v.copy() for k, v in stale.items()}
        h["IMD"][:, j] = fresh["IMD"][:, j]
        p = f(days, h)
        innov.append(np.abs(p - base)[ok]); gain.append(((base - y) ** 2 - (p - y) ** 2)[ok])
    innov = np.concatenate(innov); gain = np.concatenate(gain)
    rho = stats.spearmanr(innov, gain)
    out["innovation_proxy"] = {"spearman_innov_vs_gain": float(rho.correlation), "p": float(rho.pvalue),
                               "spearman_innov_vs_abs_gain": float(stats.spearmanr(innov, np.abs(gain)).correlation),
                               "mean_gain_top_quartile": float(gain[innov >= np.quantile(innov, 0.75)].mean()),
                               "mean_gain_bottom_quartile": float(gain[innov <= np.quantile(innov, 0.25)].mean())}
    os.makedirs(C.RES_RAW, exist_ok=True)
    json.dump(out, open(os.path.join(C.RES_RAW, "e2_voi.json"), "w"), indent=1)
    print(json.dumps(out["test"]["curves"], indent=0)[:1500]); print(out["innovation_proxy"])


if __name__ == "__main__":
    main()
