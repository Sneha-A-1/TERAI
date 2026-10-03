"""Non-learning policies for the slow-timescale environment.

Centralised planners return a joint action that respects the slot budgets.

* Random, RoundRobin, MaxAge (= the Greedy policy of Kadota et al., 2018)
* Whittle: the Whittle's-index policy of Kadota et al. (IEEE/ACM ToN 26(6), 2018,
  Eq. (57)):  C_i(h) = p_i * alpha_i * h * [ h + (1 + (1-p_i)^T) / (1 - (1-p_i)^T) ],
  with T slots per frame.  Each (district, modality) report is a client; a frame is a
  day; h counts frames since the report was last delivered; alpha_i = 1 (AoI-optimal
  weighting, as in the original paper); p_i is today's slot success probability.
  Slots are filled one at a time with the highest-index undelivered client, as in the
  original policy; a client needing c slots (SAR) is served only if c slots remain.
* VoIGreedy: myopic value-of-information knapsack using the innovation features
  (event-triggered estimation), ordered by value per slot, threshold tau.
* MyopicOpt: exhaustive search of the per-day knapsack maximising
  sum(innovation) - w_E * energy (exact optimum of the myopic proxy).
"""
from __future__ import annotations

import itertools

import numpy as np

from .env import AGENT_MODS, IS_NTN, N, N_ACT

COST = None


def _sources(env, b):
    """List of schedulable sources (j, k, cost) for env copy b."""
    out = []
    for j in range(N):
        for k, m in enumerate(AGENT_MODS):
            if env.avail[b, j, k]:
                out.append((j, k, env.cfg.slot_cost[m]))
    return out


def _to_actions(selected, B):
    a = np.zeros((B, N), int)
    for b, sel in enumerate(selected):
        for (j, k) in sel:
            a[b, j] |= (1 << k)
    return a


def _battery_ok(env, b, j, slots):
    if not IS_NTN[j]:
        return True
    need = slots * env.cfg.slot_s * env.cfg.ntn_terminal_w / 3600.0
    return env.battery[b, j] >= need


def _fill(env, b, order):
    """Greedy fill of both pools following ``order`` (list of (j,k,c))."""
    cap = {True: env.cfg.k_ntn, False: env.cfg.k_terr}
    used = {True: 0, False: 0}
    per_agent = {}
    sel = []
    for (j, k, c) in order:
        pool = bool(IS_NTN[j])
        tot = per_agent.get(j, 0) + c
        if used[pool] + c <= cap[pool] and _battery_ok(env, b, j, tot):
            used[pool] += c; per_agent[j] = tot; sel.append((j, k))
    return sel


class Random:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        m = env.action_mask()
        a = np.zeros((env.B, N), int)
        for b in range(env.B):
            for j in range(N):
                a[b, j] = self.rng.choice(np.flatnonzero(m[b, j]))
        return a, False


class RoundRobin:
    name = "Round-robin"

    def __init__(self):
        self.ptr = 0

    def act(self, env, obs):
        sel = []
        for b in range(env.B):
            src = _sources(env, b)
            if src:
                r = self.ptr % len(src)
                src = src[r:] + src[:r]
            sel.append(_fill(env, b, src))
        self.ptr += 1
        return _to_actions(sel, env.B), True


def _frames_since(env, b, j, k):
    m = AGENT_MODS[k]
    from . import config as C
    return max(int(env.t[b] - env.held[m][b, j]) - C.LATENCY[m] + 1, 1)


class MaxAge:
    name = "Max-age (Greedy)"

    def act(self, env, obs):
        sel = []
        for b in range(env.B):
            src = sorted(_sources(env, b), key=lambda s: -_frames_since(env, b, s[0], s[1]))
            sel.append(_fill(env, b, src))
        return _to_actions(sel, env.B), True


class Whittle:
    """Kadota et al. (2018) Whittle's index policy, slot-by-slot within each pool."""
    name = "Whittle index [Kadota]"

    def __init__(self, alpha=None):
        self.alpha = alpha

    @staticmethod
    def index(p, h, T, alpha=1.0):
        p = min(max(p, 1e-6), 1.0)
        q = (1.0 - p) ** T
        return p * alpha * h * (h + (1.0 + q) / max(1.0 - q, 1e-9))

    def act(self, env, obs):
        sel = []
        for b in range(env.B):
            chosen = []
            for pool, cap in [(True, env.cfg.k_ntn), (False, env.cfg.k_terr)]:
                cand = [s for s in _sources(env, b) if bool(IS_NTN[s[0]]) == pool]
                left = cap
                per_agent = {}
                served = set()
                while left > 0:
                    best, bi = None, -1.0
                    for (j, k, c) in cand:
                        if (j, k) in served or c > left:
                            continue
                        if not _battery_ok(env, b, j, per_agent.get(j, 0) + c):
                            continue
                        h = _frames_since(env, b, j, k)
                        idx = self.index(env.p_succ[b, j], h, cap, 1.0 if self.alpha is None else self.alpha[k])
                        if idx > bi:
                            best, bi = (j, k, c), idx
                    if best is None:
                        break
                    j, k, c = best
                    served.add((j, k)); per_agent[j] = per_agent.get(j, 0) + c
                    chosen.append((j, k)); left -= c
            sel.append(chosen)
        return _to_actions(sel, env.B), True


def _energy_of(env, j, k):
    cfg = env.cfg
    m = AGENT_MODS[k]
    c = cfg.slot_cost[m]
    e = c * cfg.slot_s * cfg.ntn_terminal_w / 3600.0 if IS_NTN[j] else c * cfg.terr_wh_per_slot
    return e + (cfg.s1_proc_wh if m == "S1" else cfg.report_wh)


class VoIGreedy:
    name = "VoI-greedy"

    def __init__(self, tau=0.0):
        self.tau = tau

    def act(self, env, obs):
        sel = []
        for b in range(env.B):
            src = [s for s in _sources(env, b) if env.innov[b, s[0], s[1]] > self.tau]
            src.sort(key=lambda s: -env.innov[b, s[0], s[1]] / s[2])
            sel.append(_fill(env, b, src))
        return _to_actions(sel, env.B), True


class MyopicOpt:
    """Exact per-day optimum of  sum(innovation^2) - w_E E / E_ref  under the budgets."""
    name = "Myopic optimum"

    def act(self, env, obs):
        cfg = env.cfg
        sel = []
        for b in range(env.B):
            src = _sources(env, b)
            best, bv = [], 0.0
            for r in range(len(src) + 1):
                for comb in itertools.combinations(src, r):
                    used = {True: 0, False: 0}; per = {}
                    ok = True
                    for (j, k, c) in comb:
                        used[bool(IS_NTN[j])] += c; per[j] = per.get(j, 0) + c
                    if used[True] > cfg.k_ntn or used[False] > cfg.k_terr:
                        continue
                    if not all(_battery_ok(env, b, j, s) for j, s in per.items()):
                        continue
                    v = sum(env.innov[b, j, k] ** 2 - cfg.w_energy * _energy_of(env, j, k) / cfg.e_ref_wh
                            for (j, k, c) in comb)
                    if v > bv:
                        bv, best = v, [(j, k) for (j, k, c) in comb]
            sel.append(best)
        return _to_actions(sel, env.B), True


class Fixed:
    def __init__(self, action, name):
        self.a, self.name = action, name

    def act(self, env, obs):
        return np.full((env.B, N), self.a), False


def rollout(env, policy, start=None):
    """Run one full episode on every copy; return per-step infos and rewards."""
    obs = env.reset(start)
    infos, rewards = [], []
    while not env.done.all():
        a, central = policy.act(env, obs)
        obs, r, d, info = env.step(a, centralized_schedule=central)
        infos.append(info); rewards.append(r)
    return infos, np.array(rewards)


def summarize(infos, rewards, env):
    se = np.array([i["se"] for i in infos]); ht = np.array([i["has_target"] for i in infos])
    g = lambda k: np.array([i[k] for i in infos])
    out = {
        "rmse": float(np.sqrt(se.sum() / ht.sum())),
        "decision_err": float(g("decision_err").sum() / ht.sum()),
        "energy_wh_day": float(g("energy_wh").mean()),
        "carbon_g_day": float(g("carbon_g").mean()),
        "overflow_slots_day": float(g("overflow").mean()),
        "fail_per_day": float(g("fail").mean()),
        "ntn_slots_day": float(g("ntn_slots").mean()),
        "depleted_days": float(g("depleted").sum() / env.B),
        "s1_deliv_day": float(g("s1_deliv").mean()),
        "imd_deliv_day": float(g("imd_deliv").mean()),
        "pw_deliv_day": float(g("pw_deliv").mean()),
        "reward": float(rewards.mean()),
    }
    return out


class VoIThresholdDecentral:
    """Decentralised event-triggered rule: every agent independently requests each new
    report whose innovation exceeds tau; requests contend for the pool budget
    (random admission).  Decentralised counterpart of VoI-greedy (no coordination)."""
    name = "VoI-threshold (decentralized)"

    def __init__(self, tau=0.05):
        self.tau = tau

    def act(self, env, obs):
        m = env.action_mask()
        a = np.zeros((env.B, N), int)
        for b in range(env.B):
            for j in range(N):
                bits = 0
                for k in range(3):
                    if env.avail[b, j, k] and env.innov[b, j, k] > self.tau:
                        bits |= (1 << k)
                while bits and not m[b, j, bits]:
                    bits &= bits - 1          # drop the lowest bit until battery-feasible
                a[b, j] = bits
        return a, False
