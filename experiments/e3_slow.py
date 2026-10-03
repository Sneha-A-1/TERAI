"""E3 - Slow-timescale multi-agent retrieval scheduling (main comparison + ablations).

Training simulator: replay of 2022-10-15..2023-12-31 scored with the deployed estimator
(the same SCAF ensemble used at test time) and the regret reward (Proposition 1).
Model selection: every learned method is evaluated greedily every 10 PPO updates (QMIX:
every 300 joint steps) on a fixed validation copy of the *training* simulator, and the
best feasible checkpoint is kept (no 2024 information is used).
Test environment: 2024-01-01..2024-10-31 (305 decision days).
Every learned method gets the same interaction budget (200 PPO updates x 8 envs x 60
days; QMIX 6000 x 16 joint transitions).  Learned methods: seeds 0..9; each trained
policy is evaluated on 4 copies of the test environment (independent rain timing, slot
outcomes, contention).  Heuristics: environment seeds 0..9 (4 copies each).

CoVIR (proposed) = behaviour-cloning initialisation from the decentralised VoI threshold
rule + critic warm-up + constrained PPO fine-tuning with a centralised critic and
primal-dual multipliers + checkpoint selection.
Ablations (seeds 0..4): without BC initialisation, without checkpoint selection, without
Lagrangian (fixed penalties), without VoI features (and therefore without BC),
decentralised critic, cross-fitted training simulator.
Output: results/raw/e3_slow.json, data/processed/policies/e3_*.pt
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, env as E, forecast as F, marl as M, policies as PO  # noqa: E402

SEEDS = list(range(int(os.environ.get("E3_SEEDS", 10))))
ABL_SEEDS = list(range(int(os.environ.get("E3_ABL_SEEDS", 5))))
UPDATES = int(os.environ.get("E3_UPDATES", 200))
QSTEPS = int(os.environ.get("E3_QSTEPS", 6000))
OUT = os.environ.get("E3_OUT", os.path.join(C.RES_RAW, "e3_slow.json"))
REUSE_HEUR = os.environ.get("E3_REUSE_HEUR")      # optional: path of an earlier E3 file (identical heuristics)
POL_DIR = os.path.join(C.DATA_PROC, "policies")
TAU_TEACHER = 0.05                                # threshold of the BC teacher (= tuned VoI threshold)
BC_WARMUP = 10
SEL_EVERY = 10

_G = {}


def _init():
    P = D.build_panel(); S = F.Sampler(P)
    _G.update(P=P, S=S, test_est=E.load_estimators(S), cf=E.load_crossfit(S), cfg=E.SlowConfig())


def make_train_env(B, seed, use_voi=True, crossfit=False):
    est = _G["cf"] if crossfit else _G["test_est"]
    return E.SlowEnv(_G["P"], _G["S"], est, _G["cfg"], B=B, mode="train", seed=seed, use_voi_features=use_voi)


def make_val_env(use_voi=True, crossfit=False):
    """Fixed validation copy of the training simulator (common random numbers)."""
    return make_train_env(16, 9000, use_voi, crossfit)


def make_test_env(seed, use_voi=True):
    return E.SlowEnv(_G["P"], _G["S"], _G["test_est"], _G["cfg"], B=4, mode="test", seed=1000 + seed,
                     use_voi_features=use_voi)


def eval_policy(pol, seed, use_voi=True):
    env = make_test_env(seed, use_voi)
    infos, R = PO.rollout(env, pol)
    out = PO.summarize(infos, R, env)
    est = np.array([i["est"] for i in infos])            # (T,B)
    out["est_trace_copy0"] = est[:, 0].tolist()
    return out


# learned variants: (kind, use BC, use selection, lagrangian, use VoI features, cross-fit simulator)
VARIANTS = {
    "full":          ("covir", True, True, None, True, False),
    "no_bc":         ("covir", False, True, None, True, False),
    "no_select":     ("covir", True, False, None, True, False),
    "no_lagrangian": ("covir", True, True, False, True, False),
    "no_voi":        ("covir", False, True, None, False, False),
    "decentral_critic": ("ippo", True, True, True, True, False),
    "crossfit_sim":  ("covir", True, True, None, True, True),
}


def train_eval(job):
    if not _G:
        _init()
    kind, seed, variant = job
    t0 = time.time()
    if kind == "covir":
        kind_, use_bc, use_sel, lag, use_voi, cfit = VARIANTS[variant]
    else:                                   # baselines: fixed penalties, no BC, same selection protocol
        kind_, use_bc, use_sel, lag, use_voi, cfit = kind, False, True, None, True, False
    sel = {"make_env": lambda: make_val_env(use_voi, cfit), "every": SEL_EVERY} if use_sel else None
    bc = None
    if kind_ == "qmix":
        ag = M.QMIXAgent(21, 105, M.QMIXConfig(env_steps=QSTEPS), seed=seed)
        if sel:
            sel["every"] = 300
        hist = ag.train(lambda B: make_train_env(B, 50 + seed, use_voi, cfit), select=sel)
        pol = M.GreedyWrap(ag, "QMIX")
    else:
        ag = M.PPOAgent(kind_, 21, 105, M.PPOConfig(updates=UPDATES), seed=seed, lagrangian=lag)
        warm = 0
        if use_bc:
            bc = M.bc_pretrain(ag, lambda B: make_train_env(B, 300 + seed, use_voi, cfit),
                               PO.VoIThresholdDecentral(TAU_TEACHER), rollouts=40, seed=seed)
            ag.cfg = M.PPOConfig(updates=UPDATES, lr=1e-4, ent_start=0.005, ent_end=0.0005)
            ag.opt = torch.optim.Adam(list(ag.actor.parameters()) + list(ag.critic.parameters()), lr=ag.cfg.lr)
            warm = BC_WARMUP
        hist = ag.train(lambda B: make_train_env(B, 50 + seed, use_voi, cfit), critic_warmup=warm, select=sel)
        pol = M.GreedyWrap(ag, kind_)
    res = eval_policy(pol, seed, use_voi)
    res["train_time_s"] = time.time() - t0
    env = make_test_env(seed, use_voi); obs = env.reset()
    t1 = time.time()
    for _ in range(50):
        pol.act(env, obs)
    res["decision_ms"] = (time.time() - t1) / 50 * 1000 / env.B
    res["history"] = hist
    res["bc"] = None if bc is None else {"bc_accuracy": bc["bc_accuracy"], "bc_samples": bc["bc_samples"]}
    res["selected_update"] = getattr(ag, "selected_update", None)
    res["selection_log"] = getattr(ag, "selection_log", None)
    if kind_ != "qmix":
        res["final_lambda"] = [ag.lam_o, ag.lam_d]
    os.makedirs(POL_DIR, exist_ok=True)
    state = {"q": ag.q.state_dict(), "mix": ag.mix.state_dict()} if kind_ == "qmix" else \
        {"actor": ag.actor.state_dict(), "critic": ag.critic.state_dict(), "lam": [ag.lam_o, ag.lam_d], "kind": kind_}
    torch.save(state, os.path.join(POL_DIR, "e3_%s_%s_seed%d.pt" % (kind, variant, seed)))
    return (kind, seed, variant, res)


HEUR = ["Random", "Round-robin", "Max-age (Greedy)", "Whittle index [Kadota]", "VoI-greedy",
        "VoI-threshold (decentralized)", "Myopic optimum", "IMD+POWER daily", "Never refresh"]


def heuristic_eval(job):
    if not _G:
        _init()
    name, seed, tau = job
    pol = {"Random": lambda: PO.Random(seed), "Round-robin": PO.RoundRobin, "Max-age (Greedy)": PO.MaxAge,
           "Whittle index [Kadota]": PO.Whittle, "VoI-greedy": lambda: PO.VoIGreedy(tau),
           "VoI-threshold (decentralized)": lambda: PO.VoIThresholdDecentral(tau),
           "Myopic optimum": PO.MyopicOpt, "IMD+POWER daily": lambda: PO.Fixed(3, "IMD+POWER daily"),
           "Never refresh": lambda: PO.Fixed(0, "Never refresh")}[name]()
    t1 = time.time()
    res = eval_policy(pol, seed)
    res["decision_ms"] = (time.time() - t1) / 305 * 1000 / 4
    return (name, seed, None, res)


def tune_tau():
    """Threshold of the VoI rules, tuned on the training simulator (centralised VoI-greedy)."""
    _init()
    best, bt, scores = -1e9, 0.0, {}
    for tau in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        rs = []
        for s in range(3):
            env = make_train_env(8, 700 + s)
            infos, R = PO.rollout(env, PO.VoIGreedy(tau))
            rs.append(R.mean())
        scores[tau] = float(np.mean(rs))
        if scores[tau] > best:
            best, bt = scores[tau], tau
    return bt, scores


def main():
    t0 = time.time()
    tau, tau_scores = tune_tau()
    assert abs(tau - TAU_TEACHER) < 1e-9, "BC teacher threshold must equal the tuned threshold"
    print("tau", tau, tau_scores, flush=True)
    results = {"meta": {"tau": tau, "tau_scores": tau_scores, "seeds": SEEDS, "abl_seeds": ABL_SEEDS,
                        "updates": UPDATES, "qmix_steps": QSTEPS, "bc_warmup": BC_WARMUP,
                        "select_every": SEL_EVERY}, "runs": []}
    jobs_h = [(h, s, tau) for h in HEUR for s in SEEDS]
    if REUSE_HEUR and os.path.exists(REUSE_HEUR):
        old = json.load(open(REUSE_HEUR))
        assert abs(old["meta"]["tau"] - tau) < 1e-9
        for r in old["runs"]:
            if r[2] is None and r[0] in HEUR and r[1] in SEEDS:
                results["runs"].append(r)
        have = {(r[0], r[1]) for r in results["runs"]}
        jobs_h = [j for j in jobs_h if (j[0], j[1]) not in have]
        results["meta"]["heuristics_reused_from"] = os.path.basename(REUSE_HEUR)
    jobs_l = [(k, s, "full") for k in ["covir", "ippo", "central", "qmix"] for s in SEEDS]
    jobs_l += [("covir", s, v) for v in ["no_bc", "no_select", "no_lagrangian", "no_voi", "decentral_critic",
                                          "crossfit_sim"] for s in ABL_SEEDS]
    with Pool(2, initializer=_init) as pool:
        for r in pool.imap_unordered(heuristic_eval, jobs_h):
            results["runs"].append(r)
        print("heuristics done %.0fs" % (time.time() - t0), flush=True)
        json.dump(results, open(OUT, "w"))
        for r in pool.imap_unordered(train_eval, jobs_l):
            results["runs"].append(r)
            print("done", r[0], r[1], r[2], "rmse %.4f E %.2f R %.4f sel %s" % (
                r[3]["rmse"], r[3]["energy_wh_day"], r[3]["reward"], r[3]["selected_update"]),
                "%.0fs" % (time.time() - t0), flush=True)
            json.dump(results, open(OUT, "w"))
    results["meta"]["runtime_s"] = time.time() - t0
    json.dump(results, open(OUT, "w"))
    print("all done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
