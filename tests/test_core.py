"""Unit tests: run with  python -m pytest -q tests  (or python tests/test_core.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, data as D, forecast as F, env as E, policies as PO, channel as CH, ran as R  # noqa: E402


def test_whittle_index_closed_form():
    # Kadota et al. (2018), Eq. (57): C(h) = p*alpha*h*(h + (1+(1-p)^T)/(1-(1-p)^T))
    p, h, T = 0.5, 3, 2
    q = (1 - p) ** T
    assert abs(PO.Whittle.index(p, h, T) - p * h * (h + (1 + q) / (1 - q))) < 1e-12
    # error-free channel: index = h*(h+1)
    assert abs(PO.Whittle.index(1.0, 4, 3) - 4 * 5) < 1e-12


def test_rain_attenuation_monotone():
    a = [CH.rain_attenuation_db("Almora", r, 20.0, 40.0) for r in [0, 1, 5, 10, 25, 50]]
    assert a[0] == 0.0 and all(x < y for x, y in zip(a, a[1:]))


def test_estimator_equivalence_and_budget():
    P = D.build_panel(); S = F.Sampler(P)
    r = F.train_scaf(S, "cloud", seed=0, epochs=3)
    te = S.day_range(C.TEST_START, C.TEST_END)
    held = S.random_held(te, np.random.default_rng(0))
    inp = S.inputs(te, held)
    a = F.predict_scaf(r, inp)["sm"]
    b = F.NumpySCAF(r)(inp)["sm"]
    c = F.FastSCAF(r, S)(te, held)
    assert np.abs(a - b).max() < 1e-5 and np.abs(a - c).max() < 1e-5
    env = E.SlowEnv(P, S, [F.FastSCAF(r, S)], E.SlowConfig(), B=2, mode="test", seed=0)
    env.reset()
    for _ in range(20):
        _, _, _, info = env.step(np.full((2, 5), 7))
        assert (info["ntn_slots"] <= env.cfg.k_ntn).all() and (info["terr_slots"] <= env.cfg.k_terr).all()


def test_rma_pathloss_increasing():
    rng = np.random.default_rng(0)
    d = np.array([100.0, 1000.0, 3000.0])
    pl = np.mean([R.rma_pathloss_db(d, 3.5, 35, 1.5, 5, 20, np.random.default_rng(s))[0] for s in range(200)], 0)
    assert pl[0] < pl[1] < pl[2]


def test_bc_and_checkpoint_selection():
    """The decentralised actor can represent the VoI threshold rule (behaviour cloning), and
    checkpoint selection only considers updates after the critic warm-up."""
    from terai import marl as M
    P = D.build_panel(); S = F.Sampler(P)
    est = E.load_estimators(S)
    cfg = E.SlowConfig()
    ag = M.PPOAgent("covir", 21, 105, M.PPOConfig(updates=3, n_envs=2), seed=0)
    bc = M.bc_pretrain(ag, lambda B: E.SlowEnv(P, S, est, cfg, B=B, mode="train", seed=1),
                       PO.VoIThresholdDecentral(0.05), rollouts=3, epochs=400, lr=3e-3, seed=0)
    assert bc["bc_accuracy"] > 0.9
    ag.train(lambda B: E.SlowEnv(P, S, est, cfg, B=B, mode="train", seed=2), critic_warmup=1,
             select={"make_env": lambda: E.SlowEnv(P, S, est, cfg, B=2, mode="train", seed=9), "every": 1})
    assert ag.selected_update in (1, 2) and len(ag.selection_log) == 2


if __name__ == "__main__":
    for f in [test_whittle_index_closed_form, test_rain_attenuation_monotone, test_rma_pathloss_increasing,
              test_estimator_equivalence_and_budget, test_bc_and_checkpoint_selection]:
        f(); print("ok", f.__name__)
