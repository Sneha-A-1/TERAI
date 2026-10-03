"""E0 - Cross-fitted estimators for the *training* environment.

The multi-agent policies are trained on replayed 2022-2023 days.  If those days were
scored with an estimator that was fitted on the same days, estimation errors (and hence
the cost of staleness) would be optimistically small.  We therefore split the training
period into 5 folds of 30-day blocks and, for every fold, fit SCAF ensembles on the
other folds only; the training environment scores day t with the ensemble that never
saw day t.  The test environment (2024) uses the ensemble fitted on the full training
period (saved by e1_forecast.py).
Output: data/processed/crossfit_estimators.pkl  (list over folds of model lists),
        data/processed/crossfit_folds.npy        (fold id per panel day, -1 outside)
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, forecast as F  # noqa: E402

K = 5
SEEDS = [0, 1, 2, 3]


def main():
    t0 = time.time()
    P = D.build_panel(); S = F.Sampler(P)
    days = S.day_range(C.TRAIN_START, C.TRAIN_END)
    fold = np.full(P.n_days, -1, int)
    fold[days] = ((days - days[0]) // 30) % K
    out = []
    for k in range(K):
        tr = days[fold[days] != k]
        rng = np.random.default_rng(100 + k)
        blocks = (tr - tr[0]) // 7
        ub = np.unique(blocks)
        vb = rng.choice(ub, size=int(round(0.2 * len(ub))), replace=False)
        isv = np.isin(blocks, vb)
        models = [F.train_scaf(S, "cloud", seed=s, train_days=tr[~isv], val_days=tr[isv]) for s in SEEDS]
        out.append([{"state": r.model.state_dict(), "sm_mu": r.sm_mu, "sm_sd": r.sm_sd, "variant": "cloud",
                     "mods": list(r.model.mods), "d": r.model.d, "use_gate": r.model.use_gate} for r in models])
        print("fold", k, "%.0fs" % (time.time() - t0), flush=True)
    os.makedirs(C.DATA_PROC, exist_ok=True)
    pickle.dump(out, open(os.path.join(C.DATA_PROC, "crossfit_estimators.pkl"), "wb"))
    np.save(os.path.join(C.DATA_PROC, "crossfit_folds.npy"), fold)


if __name__ == "__main__":
    main()
