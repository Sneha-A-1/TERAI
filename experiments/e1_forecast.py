"""E1 - Regional soil-moisture nowcasting on the real 2024 test period.

Compares the staleness-conditioned attention fusion forecaster (SCAF) with persistence,
climatology, early-fusion ridge / gradient boosting / MLP, late fusion, the no-gate
ablation and leave-one-modality-out ablations.  Every model is trained on the
2022-10-15..2023-12-31 period only (SCAF uses an internal validation split for early
stopping) and evaluated on 2024-01-01..2024-10-31 under (a) fresh inputs and
(b) random staleness patterns (5 fixed draws).
Outputs: results/raw/e1_forecast.json and data/processed/estimator_ensemble.pkl
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, forecast as F  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

N_SEEDS = int(os.environ.get("E1_SEEDS", 10))
N_ABL = int(os.environ.get("E1_ABL_SEEDS", 5))
TAG = os.environ.get("E1_TAG", "")


def metrics(pred, y, thr=0.20):
    ok = np.isfinite(y)
    e = pred[ok] - y[ok]
    ss = ((y[ok] - y[ok].mean()) ** 2).sum()
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "r2": float(1 - (e ** 2).sum() / ss), "bias": float(e.mean()),
            "decision_err": float(((pred[ok] < thr) != (y[ok] < thr)).mean()), "n": int(ok.sum())}


def main():
    t0 = time.time()
    P = D.build_panel(); S = F.Sampler(P)
    tr_all = S.day_range(C.TRAIN_START, C.TRAIN_END)
    te = S.day_range(C.TEST_START, C.TEST_END)
    y_te = S.targets(te)[1]
    regimes = {"fresh": [S.inputs(te, S.fresh_held(te))]}
    regimes["stale"] = [S.inputs(te, S.random_held(te, np.random.default_rng(900 + k))) for k in range(5)]

    # augmented training design for the scikit-learn baselines (same staleness process)
    rng = np.random.default_rng(0)
    tt = np.concatenate([tr_all] * 8)
    held = S.random_held(tt, rng); fresh = S.fresh_held(tt)
    sel = rng.random(len(tt)) < 0.3
    for m in held:
        held[m][sel] = fresh[m][sel]
    inp_tr = S.inputs(tt, held)
    y_tr = S.targets(tt)[1]
    ok_tr = np.isfinite(y_tr)
    Xtr = F.regional_features(inp_tr)
    base_tr = inp_tr["sm_base"]

    res = {"meta": {"n_train_days": int(len(tr_all)), "n_test_days": int(len(te)),
                    "n_test_targets": int(np.isfinite(y_te).sum()), "seeds": N_SEEDS}}

    def evaluate(predict_fn):
        out = {}
        for rg, inps in regimes.items():
            ms = [metrics(predict_fn(inp), y_te) for inp in inps]
            out[rg] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        return out

    # ---- reference baselines -------------------------------------------------------
    res["Persistence"] = [evaluate(lambda inp: inp["sm_base"])]
    months_tr = P.days[tr_all + C.HORIZON].month.values
    ytr0 = P.sm_target[tr_all + C.HORIZON]
    clim = np.array([np.nanmean(ytr0[months_tr == mo]) if (months_tr == mo).any() else np.nanmean(ytr0)
                     for mo in range(13)])
    mo_te = P.days[te + C.HORIZON].month.values
    res["Climatology"] = [evaluate(lambda inp: clim[mo_te])]

    # residual targets for the regression baselines (same persistence prior as SCAF)
    r_tr = (y_tr - base_tr)
    rid = Ridge(alpha=1.0).fit(Xtr[ok_tr], r_tr[ok_tr])
    res["Ridge (early fusion)"] = [evaluate(lambda inp: inp["sm_base"] + rid.predict(F.regional_features(inp)))]

    # late fusion: one ridge per modality (own features + calendar), predictions averaged
    cols = {}
    off = 0
    for m in C.MODALITIES:
        f = len(D.F_NAMES[m])
        w = f + (2 if m == "SMAP" else 1)
        cols[m] = list(range(off, off + w)); off += w
    cal_cols = list(range(off, off + 2))
    late = {m: Ridge(alpha=1.0).fit(Xtr[ok_tr][:, cols[m] + cal_cols], r_tr[ok_tr]) for m in C.MODALITIES}
    res["Late fusion (ridge)"] = [evaluate(lambda inp: inp["sm_base"] + np.mean(
        [late[m].predict(F.regional_features(inp)[:, cols[m] + cal_cols]) for m in C.MODALITIES], 0))]

    res["Gradient boosting (early fusion)"] = []
    res["MLP (early fusion)"] = []
    for s in range(N_SEEDS):
        hg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                           l2_regularization=1.0, random_state=s).fit(Xtr[ok_tr], r_tr[ok_tr])
        res["Gradient boosting (early fusion)"].append(
            evaluate(lambda inp, hg=hg: inp["sm_base"] + hg.predict(F.regional_features(inp))))
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-3, learning_rate_init=3e-3,
                           max_iter=400, early_stopping=True, random_state=s).fit(Xtr[ok_tr], r_tr[ok_tr] / S.sm_sd)
        res["MLP (early fusion)"].append(
            evaluate(lambda inp, mlp=mlp: inp["sm_base"] + S.sm_sd * mlp.predict(F.regional_features(inp))))
    print("baselines done %.0fs" % (time.time() - t0), flush=True)

    # ---- SCAF and ablations ---------------------------------------------------------
    trd, vad = F.split_train_val(S)
    scaf_models = []
    res["SCAF (proposed)"] = []
    res["SCAF without staleness gate"] = []
    for s in range(N_SEEDS):
        r = F.train_scaf(S, "cloud", seed=s, train_days=trd, val_days=vad)
        scaf_models.append(r)
        res["SCAF (proposed)"].append(evaluate(lambda inp, r=r: F.predict_scaf(r, inp)["sm"]))
        r2 = F.train_scaf(S, "cloud", seed=s, train_days=trd, val_days=vad, use_gate=False)
        res["SCAF without staleness gate"].append(evaluate(lambda inp, r2=r2: F.predict_scaf(r2, inp)["sm"]))
        print("scaf seed", s, "%.0fs" % (time.time() - t0), flush=True)
    for drop in C.MODALITIES:
        key = "SCAF without %s" % drop
        res[key] = []
        mods = [m for m in C.MODALITIES if m != drop]
        for s in range(N_ABL):
            r = F.train_scaf(S, "cloud", seed=s, train_days=trd, val_days=vad, mods=mods)
            res[key].append(evaluate(lambda inp, r=r: F.predict_scaf(r, inp)["sm"]))
        print(key, "%.0fs" % (time.time() - t0), flush=True)

    # ensemble used inside the multi-agent environment (first three seeds)
    os.makedirs(C.DATA_PROC, exist_ok=True)
    F.save_models(scaf_models[:3], os.path.join(C.DATA_PROC, TAG + "estimator_ensemble.pkl"))
    F.save_models(scaf_models, os.path.join(C.DATA_PROC, TAG + "scaf_all_seeds.pkl"))
    ens = lambda inp: np.mean([F.predict_scaf(r, inp)["sm"] for r in scaf_models[:3]], 0)
    res["SCAF ensemble (environment estimator)"] = [evaluate(ens)]
    # attention weights of the regional query (interpretability), fresh test inputs
    import torch
    with torch.no_grad():
        out = scaf_models[0].model(F.to_torch(regimes["fresh"][0]))
    w = out["attn_reg"][:, 0, :].numpy().mean(0)
    names = []
    for m in scaf_models[0].model.mods:
        names += ([m] if m == "SMAP" else ["%s:%s" % (m, d) for d in C.DISTRICTS])
    names.append("calendar")
    res["attention_regional"] = dict(zip(names, map(float, w)))
    res["meta"]["runtime_s"] = time.time() - t0
    os.makedirs(C.RES_RAW, exist_ok=True)
    json.dump(res, open(os.path.join(C.RES_RAW, TAG + "e1_forecast.json"), "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
