"""E4 - Policy analysis of the trained retrieval policies (no retraining).

Loads the policies trained in E3 (data/processed/policies/e3_*.pt) and evaluates
1. generalisation: a continuous replay of the *training* period (2022-10-15..2023-12-31)
   versus the unseen 2024 season, for CoVIR, CoVIR without BC initialisation, QMIX and
   the main heuristics;
2. slot-budget sensitivity on the 2024 season, K_H in {2,3,4,6,8}, and rain-fade margin
   sensitivity (clear-sky Es/N0 in {12, 2, 0, -1} dB), both without retraining;
3. decision time of the decentralised actor versus the number of agents (replicated
   observations), of the Whittle index versus the number of sources, and of one day of
   SCAF ensemble inference including the innovation features.
Output: results/raw/e4_policy.json
"""
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, env as E, forecast as F, marl as M, policies as PO  # noqa: E402

SEEDS = list(range(int(os.environ.get("E4_SEEDS", 5))))
BUDGETS = [2, 3, 4, 6, 8]
CLEAR_SNR = [12.0, 2.0, 0.0, -1.0]      # clear-sky Es/N0 (dB); 12 dB is the default
POL_DIR = os.path.join(C.DATA_PROC, "policies")
_G = {}


def _init():
    P = D.build_panel(); S = F.Sampler(P)
    _G.update(P=P, S=S, est=E.load_estimators(S), cfg=E.SlowConfig())


def replay_env(period, seed, cfg=None):
    cfg = cfg or _G["cfg"]
    env = E.SlowEnv(_G["P"], _G["S"], _G["est"], cfg, B=4, mode="test", seed=2000 + seed)
    if period == "train":
        env.lo = _G["P"].day_index(C.TRAIN_START); env.hi = _G["P"].day_index(C.TRAIN_END)
    return env


def evaluate(pol, seed, period, cfg=None):
    env = replay_env(period, seed, cfg)
    infos, R = PO.rollout(env, pol)
    return PO.summarize(infos, R, env)


def load_policy(kind, variant, seed):
    st = torch.load(os.path.join(POL_DIR, "e3_%s_%s_seed%d.pt" % (kind, variant, seed)))
    if kind == "qmix":
        ag = M.QMIXAgent(21, 105, M.QMIXConfig(), seed=seed)
        ag.q.load_state_dict(st["q"]); ag.mix.load_state_dict(st["mix"])
    else:
        ag = M.PPOAgent(st["kind"], 21, 105, M.PPOConfig(), seed=seed)
        ag.actor.load_state_dict(st["actor"]); ag.critic.load_state_dict(st["critic"])
    return M.GreedyWrap(ag, "%s-%s" % (kind, variant))


LEARNED = {"CoVIR (proposed)": ("covir", "full"), "CoVIR without BC": ("covir", "no_bc"),
           "QMIX": ("qmix", "full")}
HEUR = {"VoI-greedy": lambda: PO.VoIGreedy(0.05), "VoI-threshold (decentralized)": lambda: PO.VoIThresholdDecentral(0.05),
        "Whittle index [Kadota]": PO.Whittle, "Max-age (Greedy)": PO.MaxAge,
        "IMD+POWER daily": lambda: PO.Fixed(3, "IMD+POWER daily")}


def job(args):
    if not _G:
        _init()
    name, seed = args
    mk = (lambda: load_policy(*LEARNED[name], seed)) if name in LEARNED else HEUR[name]
    out = {"name": name, "seed": seed, "train_period": evaluate(mk(), seed, "train"),
           "test": evaluate(mk(), seed, "test"), "budget": {}}
    for k in BUDGETS:
        cfg = copy.deepcopy(_G["cfg"]); cfg.k_ntn = k
        out["budget"][str(k)] = evaluate(mk(), seed, "test", cfg)
    out["link"] = {}
    for snr in CLEAR_SNR:                  # smaller clear-sky Es/N0 = smaller rain-fade margin
        cfg = copy.deepcopy(_G["cfg"]); cfg.ntn_clear_snr_db = snr
        out["link"][str(snr)] = evaluate(mk(), seed, "test", cfg)
    return out


def timing():
    _init()
    ag = M.PPOAgent("covir", 21, 105, M.PPOConfig(), seed=0)
    res = {"actor_ms_vs_agents": {}, "whittle_ms_vs_sources": {}}
    # one day of SCAF ensemble inference: the estimate plus the 10 innovation variants
    S = _G["S"]; est = _G["est"]
    env = replay_env("test", 0); env.reset()
    t0 = time.time()
    for _ in range(50):
        env._new_day()
    res["scaf_day_ms"] = (time.time() - t0) / 50 * 1000 / env.B
    for n in [5, 10, 20, 40, 80, 160]:
        o = torch.randn(1, n, 21); m = torch.ones(1, n, 8, dtype=torch.bool)
        t0 = time.time()
        for _ in range(200):
            with torch.no_grad():
                ag.actor(o).masked_fill(~m, -1e9).argmax(-1)
        res["actor_ms_vs_agents"][str(n)] = (time.time() - t0) / 200 * 1000
    for n in [15, 30, 60, 120, 240]:
        rng = np.random.default_rng(0)
        h = rng.integers(1, 30, n); p = rng.uniform(0.6, 1.0, n)
        t0 = time.time()
        for _ in range(50):
            left, served = 4, set()
            while left > 0:
                idx = [PO.Whittle.index(p[i], h[i], 4) if i not in served else -1 for i in range(n)]
                i = int(np.argmax(idx)); served.add(i); left -= 1
        res["whittle_ms_vs_sources"][str(n)] = (time.time() - t0) / 50 * 1000
    return res


def main():
    t0 = time.time()
    out = {"timing": timing(), "runs": []}
    jobs = [(n, s) for n in list(LEARNED) + list(HEUR) for s in SEEDS]
    with Pool(2, initializer=_init) as pool:
        for r in pool.imap_unordered(job, jobs):
            out["runs"].append(r)
            print(r["name"], r["seed"], "train-period rmse %.4f R %.4f | test rmse %.4f R %.4f" % (
                r["train_period"]["rmse"], r["train_period"]["reward"], r["test"]["rmse"], r["test"]["reward"]),
                "%.0fs" % (time.time() - t0), flush=True)
            json.dump(out, open(os.path.join(C.RES_RAW, "e4_policy.json"), "w"))
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(os.path.join(C.RES_RAW, "e4_policy.json"), "w"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
