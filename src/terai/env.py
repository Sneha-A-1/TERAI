"""Trace-driven slow-timescale (daily) multi-agent environment.

Five district edge agents decide, every day, which of their district observations to
deliver to the regional cloud estimator:  IMD-type rain/temperature report, POWER-type
weather report, and Sentinel-1 SAR features (which require the raw SAR subset to be
downloaded and processed at the edge).  SMAP is ingested by the cloud directly when it
becomes available (exogenous).

Coupling between agents:
  * hill agents share K_N transmission slots per day on a LEO beam (plains share K_T
    terrestrial slots); requests beyond the budget are dropped (decentralised access);
  * hill terminals run on a small solar-charged battery (real NASA POWER irradiance);
  * NTN slots fail with a probability set by rain attenuation (real IMD rainfall);
  * the team reward is the regional soil-moisture estimation error of the cloud
    estimator (real SMAP target) plus energy.

All environment randomness (rain timing within the window, slot success, contention
order) is seeded; the data are replayed from the real panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .channel import BackhaulDay
from .forecast import FastSCAF, Sampler, load_models

AGENT_MODS = ["IMD", "POWER", "S1"]
N = len(C.DISTRICTS)
IS_NTN = np.array([C.BACKHAUL[d] == "ntn" for d in C.DISTRICTS])
N_ACT = 8                                           # bitmask over AGENT_MODS


def action_bits(a: int):
    return [(a >> k) & 1 for k in range(3)]


@dataclass
class SlowConfig(C.EnvConfig):
    k_ntn: int = 4                                  # LEO slots/day for the agricultural slice (hills)
    k_terr: int = 6                                 # terrestrial slots/day (plains)
    slot_cost: dict = field(default_factory=lambda: {"IMD": 1, "POWER": 1, "S1": 2})
    slot_s: float = 120.0                           # duration of one transmission slot (s)
    ntn_terminal_w: float = 60.0                    # satellite terminal power while active (W)
    ntn_standby_w: float = 0.5                      # terminal standby power (W)
    pv_peak_w: float = 10.0                         # PV module rating of the terminal kit (W)
    pv_pr: float = 0.8                              # PV performance ratio
    battery_wh: float = 40.0                        # terminal battery capacity (Wh)
    s1_proc_wh: float = 12.5                        # edge processing of one SAR subset (Wh)
    report_wh: float = 0.2                          # collecting one station report over the RAN (Wh)
    terr_wh_per_slot: float = 0.05                  # terrestrial transport energy per slot (Wh)
    e_ref_wh: float = 20.0                          # energy normalisation in the reward
    sm_threshold: float = 0.20                      # irrigation refill point (m3/m3)
    lam_overflow: float = 0.2                       # fixed penalty per dropped slot (baselines)
    lam_depletion: float = 1.0                      # fixed penalty per depleted terminal-day


class SlowEnv:
    """Vectorised over B independent copies (different seeds / start days)."""

    def __init__(self, panel, sampler: Sampler, estimators, cfg: SlowConfig, B: int,
                 mode: str = "train", seed: int = 0, use_voi_features: bool = True):
        """estimators: list of FastSCAF (used for every day) or a dict
        {"folds": array(day->fold), "models": [list of FastSCAF per fold]} for the
        cross-fitted training environment."""
        self.P, self.S, self.cfg, self.B = panel, sampler, cfg, B
        self.est = estimators
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.use_voi = use_voi_features
        self.lo = panel.day_index(C.TRAIN_START if mode == "train" else C.TEST_START)
        self.hi = panel.day_index(C.TRAIN_END if mode == "train" else C.TEST_END)
        self.latest = {m: sampler.latest[m] for m in C.MODALITIES}
        self.sigma = float(np.nanstd(panel.sm_target[sampler.day_range(C.TRAIN_START, C.TRAIN_END)]))
        # real solar irradiance (MJ m-2 day-1 -> peak-sun-hours); Haridwar is a grid site
        pw = panel.feat["POWER"][:, :, 4]
        self.psh = np.where(np.isnan(pw), 0.0, pw / 3.6)
        self.obs_dim = 21
        self.state_dim = self.obs_dim * N
        # S1 value table (expected squared-error reduction per refresh); filled by voi.py
        self.s1_voi = np.zeros(C.A_MAX + 1)

    # ------------------------------------------------------------------ helpers
    def estimate(self, t, held):
        # estimators are FastSCAF objects (table-driven SCAF regional head)
        if isinstance(self.est, dict):                 # cross-fitted: day t scored by its fold
            out = np.zeros(len(t))
            fo = self.est["folds"][t]
            for k in np.unique(fo):
                sel = fo == k
                hk = {m: v[sel] for m, v in held.items()}
                out[sel] = np.mean([e(t[sel], hk) for e in self.est["models"][k]], 0)
            return out
        return np.mean([e(t, held) for e in self.est], 0)

    def _held_full(self):
        h = {m: self.held[m] for m in AGENT_MODS}
        h["SMAP"] = np.repeat(self.latest["SMAP"][self.t][:, :1], N, axis=1)
        return h

    def reset(self, start=None):
        B = self.B
        if self.mode == "train":
            self.t = self.rng.integers(self.lo, self.hi - self.cfg.episode_days, size=B)
            self.end = self.t + self.cfg.episode_days
        else:
            self.t = np.full(B, self.lo if start is None else start)
            self.end = np.full(B, self.hi + 1)
        self.held = {m: self.latest[m][self.t].copy() for m in AGENT_MODS}
        if self.mode == "train":
            # start from a random information state (ages up to A_MAX, as at test time)
            for m in AGENT_MODS:
                for b in range(B):
                    for j in range(N):
                        if self.held[m][b, j] < 0:
                            continue
                        acq = self.S.acq[m][j]
                        cand = acq[(acq >= self.t[b] - C.A_MAX - 5) & (acq <= self.held[m][b, j])]
                        if len(cand) and self.rng.random() < 0.8:
                            self.held[m][b, j] = int(self.rng.choice(cand))
        self.battery = np.where(IS_NTN[None], self.rng.uniform(0.5, 1.0, (B, N)) * self.cfg.battery_wh
                                if self.mode == "train" else 0.75 * self.cfg.battery_wh, np.inf)
        self.last_overflow = np.zeros((B, N))
        self.done = np.zeros(B, bool)
        self._new_day()
        return self.obs()

    def _new_day(self):
        B = self.B
        self.bh = [BackhaulDay(self.cfg, self.P.rain_mm[self.t[b]], self.rng) for b in range(B)]
        self.p_succ = np.stack([d.p_success for d in self.bh])            # (B,N)
        self.avail = np.zeros((B, N, 3), bool)
        for k, m in enumerate(AGENT_MODS):
            self.avail[:, :, k] = self.latest[m][self.t] > self.held[m]
        self.base_est = self.estimate(self.t, self._held_full())
        self.innov = np.zeros((B, N, 3))
        if self.use_voi:
            self._innovation()

    def _innovation(self):
        """|change of the regional estimate| if agent j delivered its new IMD/POWER report
        (computable at the edge: the edge holds its own new report and the broadcast
        cloud state).  S1 uses the expected value table (content unknown before download)."""
        B = self.B
        ts, helds, idx = [], {m: [] for m in C.MODALITIES}, []
        hf = self._held_full()
        for k, m in enumerate(AGENT_MODS[:2]):
            for j in range(N):
                h = {mm: hf[mm].copy() for mm in hf}
                h[m][:, j] = np.where(self.avail[:, j, k], self.latest[m][self.t, j], h[m][:, j])
                for mm in C.MODALITIES:
                    helds[mm].append(h[mm])
                ts.append(self.t)
                idx.append((j, k))
        tt = np.concatenate(ts)
        hh = {mm: np.concatenate(helds[mm]) for mm in C.MODALITIES}
        est = self.estimate(tt, hh).reshape(len(idx), B)
        for r, (j, k) in enumerate(idx):
            self.innov[:, j, k] = np.abs(est[r] - self.base_est) / self.sigma * self.avail[:, j, k]
        ages = np.clip(self.t[:, None] - self.held["S1"], 0, C.A_MAX)
        self.innov[:, :, 2] = self.s1_voi[ages] * self.avail[:, :, 2]

    def obs(self):
        B, cfg = self.B, self.cfg
        o = np.zeros((B, N, self.obs_dim), np.float32)
        for k, m in enumerate(AGENT_MODS):
            a = np.clip(self.t[:, None] - self.held[m], 0, C.A_MAX)
            o[:, :, k] = np.log1p(a) / np.log1p(C.A_MAX)
        o[:, :, 3:6] = self.avail
        sa = np.clip(self.t - self.latest["SMAP"][self.t, 0], 0, C.A_MAX)
        o[:, :, 6] = (np.log1p(sa) / np.log1p(C.A_MAX))[:, None]
        o[:, :, 7] = self.p_succ
        o[:, :, 8] = np.where(IS_NTN[None], self.battery / cfg.battery_wh, 1.0)
        r = self.P.rain_mm[self.t]
        r3 = self.P.rain_mm[np.maximum(self.t - 2, 0)] + self.P.rain_mm[np.maximum(self.t - 1, 0)] + r
        o[:, :, 9] = np.log1p(r) / 5
        o[:, :, 10] = np.log1p(r3) / 5
        o[:, :, 11] = IS_NTN[None]
        o[:, :, 12:17] = np.eye(N)[None]
        o[:, :, 17] = self.last_overflow
        if self.use_voi:
            o[:, :, 18:21] = np.clip(self.innov * 5, 0, 5)
        return o

    def state(self, o=None):
        o = self.obs() if o is None else o
        return o.reshape(self.B, -1)

    def action_mask(self):
        """(B,N,8) feasible actions: only new products, battery for every slot requested."""
        B, cfg = self.B, self.cfg
        mask = np.ones((B, N, N_ACT), bool)
        cost = np.array([cfg.slot_cost[m] for m in AGENT_MODS])
        for a in range(N_ACT):
            bits = np.array(action_bits(a), bool)
            if bits.any():
                ok = (self.avail[:, :, bits]).all(-1)
                need = (cost * bits).sum() * cfg.slot_s * cfg.ntn_terminal_w / 3600.0
                ok &= np.where(IS_NTN[None], self.battery >= need, True)
                mask[:, :, a] = ok
        return mask

    # ------------------------------------------------------------------ step
    def step(self, actions, centralized_schedule=False):
        """actions: (B,N) ints in [0,8).  If centralized_schedule, the joint action is
        assumed feasible w.r.t. the slot budgets (planner baselines)."""
        B, cfg = self.B, self.cfg
        info = {k: np.zeros(B) for k in ["se", "has_target", "energy_wh", "overflow", "fail",
                                         "granted", "depleted", "decision_err", "est", "ntn_slots",
                                         "terr_slots", "carbon_g", "s1_deliv", "imd_deliv", "pw_deliv"]}
        mask = self.action_mask()
        a = np.where(mask[np.arange(B)[:, None], np.arange(N)[None], actions], actions, 0)
        overflow = np.zeros((B, N))
        for b in range(B):
            req = []
            for j in range(N):
                for k, bit in enumerate(action_bits(int(a[b, j]))):
                    if bit:
                        req.append((j, k, cfg.slot_cost[AGENT_MODS[k]]))
            granted = []
            for pool, cap in [(True, cfg.k_ntn), (False, cfg.k_terr)]:
                items = [r for r in req if IS_NTN[r[0]] == pool]
                if not centralized_schedule:
                    self.rng.shuffle(items)
                used = 0
                for (j, k, c) in items:
                    if used + c <= cap:
                        granted.append((j, k, c)); used += c
                    else:
                        overflow[b, j] += c
                if pool:
                    info["ntn_slots"][b] = used
                else:
                    info["terr_slots"][b] = used
            e = 0.0; e_grid = 0.0
            for (j, k, c) in granted:
                m = AGENT_MODS[k]
                ok = self.rng.random() < self.p_succ[b, j] ** c
                if IS_NTN[j]:
                    ej = c * cfg.slot_s * cfg.ntn_terminal_w / 3600.0
                    self.battery[b, j] -= ej
                    e += ej
                else:
                    e_grid += c * cfg.terr_wh_per_slot
                proc = cfg.s1_proc_wh if m == "S1" else cfg.report_wh
                e += proc; e_grid += proc if not IS_NTN[j] else 0.0
                if ok:
                    self.held[m][b, j] = self.latest[m][self.t[b], j]
                    info[{"IMD": "imd_deliv", "POWER": "pw_deliv", "S1": "s1_deliv"}[m]][b] += 1
                else:
                    info["fail"][b] += 1
            info["granted"][b] = len(granted)
            info["energy_wh"][b] = e + sum(c * cfg.terr_wh_per_slot for (j, k, c) in granted if not IS_NTN[j])
            info["carbon_g"][b] = e_grid / 1000.0 * C.EnergyConfig().grid_kgco2_per_kwh * 1000.0
        info["overflow"] = overflow.sum(1)
        self.last_overflow = overflow / max(cfg.k_ntn, 1)
        # estimate with the delivered data, score against the real SMAP target
        est = self.estimate(self.t, self._held_full())
        tgt = self.P.sm_target[self.t]
        has = np.isfinite(tgt)
        se = np.where(has, (est - np.nan_to_num(tgt)) ** 2, 0.0)
        info["se"], info["has_target"], info["est"] = se, has.astype(float), est
        hf = {m: self.latest[m][self.t].copy() for m in AGENT_MODS}
        hf["SMAP"] = np.repeat(self.latest["SMAP"][self.t][:, :1], N, axis=1)
        est_f = self.estimate(self.t, hf)
        se_f = np.where(has, (est_f - np.nan_to_num(tgt)) ** 2, 0.0)
        info["se_fresh"] = se_f
        dec_e = (est < cfg.sm_threshold) != (np.nan_to_num(tgt) < cfg.sm_threshold)
        info["decision_err"] = np.where(has, dec_e, 0.0)
        # batteries: standby drain and solar harvest (real irradiance)
        harvest = cfg.pv_peak_w * cfg.pv_pr * self.psh[self.t]
        stand = cfg.ntn_standby_w * 24.0
        nb = self.battery - stand + harvest
        dep = IS_NTN[None] & (nb <= 0)
        info["depleted"] = dep.sum(1)
        self.battery = np.where(IS_NTN[None], np.clip(nb, 0.0, cfg.battery_wh), np.inf)
        # reward (shared by all agents)
        # reward = -(estimation regret w.r.t. always-fresh data) - energy; the fresh-data error
        # does not depend on the actions, so this control variate leaves the optimum unchanged
        r = -((se - se_f) / self.sigma ** 2) - cfg.w_energy * info["energy_wh"] / cfg.e_ref_wh
        self.last_info = info
        self.done = (self.t + 1) >= self.end
        self.t = np.minimum(self.t + 1, self.P.n_days - 1)
        if not self.done.all():
            self._new_day()
        return self.obs(), r, self.done.copy(), info


def load_estimators(sampler, path=None):
    """Load the SCAF ensemble saved by experiments/e1_forecast.py as FastSCAF objects."""
    import os
    path = path or os.path.join(C.DATA_PROC, "estimator_ensemble.pkl")
    return [FastSCAF(tr, sampler) for tr in load_models(path)]


def load_crossfit(sampler):
    """Cross-fitted estimators (experiments/e0_crossfit_estimators.py) for training."""
    import os, pickle
    from .forecast import TrainResult, build_scaf
    raw = pickle.load(open(os.path.join(C.DATA_PROC, "crossfit_estimators.pkl"), "rb"))
    folds = np.load(os.path.join(C.DATA_PROC, "crossfit_folds.npy"))
    models = []
    for fold_models in raw:
        lst = []
        for z in fold_models:
            m = build_scaf(z["variant"], d=z["d"], mods=z["mods"])
            m.load_state_dict(z["state"]); m.sm_mu, m.sm_sd = z["sm_mu"], z["sm_sd"]; m.eval()
            lst.append(FastSCAF(TrainResult(m, z["sm_mu"], z["sm_sd"], []), sampler))
        models.append(lst)
    folds = np.where(folds < 0, 0, folds)
    return {"folds": folds, "models": models}
