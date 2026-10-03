"""Fast-timescale 6G RAN simulator for the agricultural network slice of one rural cell.

Channel: 3GPP TR 38.901 rural macro (RMa) path loss (Table 7.4.1-1) with the RMa LOS
probability (Table 7.4.2-1), log-normal shadowing (LOS 4/6 dB, NLOS 8 dB) and Rayleigh
block fading per resource block (RB) and slot.  Link abstraction: attenuated Shannon
mapping of 3GPP TR 36.942 Annex A (DL alpha=0.6, SE_max=4.4 b/s/Hz; UL alpha=0.4,
SE_max=2.0 b/s/Hz; SNR_min=-10 dB).  Noise-limited isolated rural cell.

Traffic of the slice (synthetic, calibrated on real records):
  class 0 - urgent actuation commands (DL), deadline 20 ms
  class 1 - routine irrigation actuation commands (DL), deadline 100 ms
  class 2 - telemetry reports (UL), deadline 1 s, inter-arrivals resampled from the
            real Mendeley IoT log, sizes from the Farm-Flow-calibrated generator
  class 3 - flood traffic from compromised sensors (UL, full buffer, packet sizes of the
            Farm-Flow attack flows), no deadline
An irrigation advisory issued by the slow layer triggers one command per valve at t=0.

Schedulers: RR, PF, EDF, MaxWeight and DRIP (proposed; Lyapunov drift-plus-penalty with
per-class virtual deadline-violation queues and an energy penalty).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

C_LIGHT = 3e8


@dataclass
class RanConfig:
    fc_ghz: float = 3.5
    n_rb: int = 11                 # RBs of the agricultural slice (about 4 MHz at 30 kHz SCS)
    scs_khz: float = 30.0
    slot_ms: float = 0.5
    h_bs: float = 35.0
    h_ut: float = 1.5
    h_bld: float = 5.0
    w_street: float = 20.0
    radius_m: float = 3000.0
    p_gnb_dbm_per_rb: float = 46.0 - 10 * math.log10(51)   # 46 dBm over 51 RBs (20 MHz carrier)
    p_ue_dbm: float = 23.0
    g_bs_dbi: float = 8.0
    nf_gnb_db: float = 5.0
    nf_ue_db: float = 9.0
    sim_ms: float = 2000.0
    n_sensors: int = 150
    n_valves: int = 50
    urgent_frac: float = 0.1
    compromised_frac: float = 0.0
    cmd_bytes: int = 64
    deadlines_ms: tuple = (20.0, 100.0, 1000.0, 1e12)
    ue_circuit_w: float = 0.1      # device circuit power while transmitting (W)
    drip_v_energy: float = 0.05
    drip_eps: tuple = (0.01, 0.01, 0.05)

    @property
    def rb_hz(self):
        return 12 * self.scs_khz * 1e3


def rma_pathloss_db(d2d, fc_ghz, h_bs, h_ut, h, W, rng):
    """3GPP TR 38.901 RMa path loss incl. LOS state and shadow fading (dB)."""
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    dbp = 2 * math.pi * h_bs * h_ut * fc_ghz * 1e9 / C_LIGHT

    def pl1(d):
        return (20 * np.log10(40 * math.pi * d * fc_ghz / 3) + min(0.03 * h ** 1.72, 10) * np.log10(d)
                - min(0.044 * h ** 1.72, 14.77) + 0.002 * math.log10(h) * d)
    los_pl = np.where(d2d <= dbp, pl1(d3d), pl1(dbp) + 40 * np.log10(d3d / dbp))
    nlos_p = (161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h)
              - (24.37 - 3.7 * (h / h_bs) ** 2) * math.log10(h_bs)
              + (43.42 - 3.1 * math.log10(h_bs)) * (np.log10(d3d) - 3)
              + 20 * math.log10(fc_ghz) - (3.2 * (math.log10(11.75 * h_ut)) ** 2 - 4.97))
    nlos_pl = np.maximum(los_pl, nlos_p)
    p_los = np.where(d2d <= 10, 1.0, np.exp(-(d2d - 10) / 1000))
    los = rng.random(len(d2d)) < p_los
    sf = np.where(los, np.where(d2d <= dbp, 4.0, 6.0), 8.0) * rng.standard_normal(len(d2d))
    return np.where(los, los_pl, nlos_pl) + sf, los


def se_from_snr(snr_db, uplink):
    a = np.where(uplink, 0.4, 0.6)
    smax = np.where(uplink, 2.0, 4.4)
    se = a * np.log2(1 + 10 ** (snr_db / 10))
    return np.where(snr_db < -10, 0.0, np.minimum(se, smax))


class Cell:
    def __init__(self, cfg: RanConfig, rng, telemetry_sizes, telemetry_intervals, attack_sizes):
        self.cfg, self.rng = cfg, rng
        n = cfg.n_sensors + cfg.n_valves
        r = np.sqrt(rng.uniform((35 / cfg.radius_m) ** 2, 1, n)) * cfg.radius_m
        self.pl, self.los = rma_pathloss_db(r, cfg.fc_ghz, cfg.h_bs, cfg.h_ut, cfg.h_bld, cfg.w_street, rng)
        self.dist = r
        self.is_valve = np.r_[np.zeros(cfg.n_sensors, bool), np.ones(cfg.n_valves, bool)]
        n_comp = int(round(cfg.compromised_frac * cfg.n_sensors))
        self.compromised = np.zeros(n, bool)
        self.compromised[rng.choice(cfg.n_sensors, n_comp, replace=False)] = True
        noise = lambda nf: -174 + 10 * math.log10(cfg.rb_hz) + nf
        # mean SNR per RB (dB) for DL (valves) and UL (sensors)
        self.snr_dl = cfg.p_gnb_dbm_per_rb + cfg.g_bs_dbi - self.pl - noise(cfg.nf_ue_db)
        self.snr_ul = cfg.p_ue_dbm + cfg.g_bs_dbi - self.pl - noise(cfg.nf_gnb_db)
        self.tel_sizes, self.tel_iv, self.att_sizes = telemetry_sizes, telemetry_intervals, attack_sizes


def simulate(cfg: RanConfig, scheduler: str, seed: int, n_commands: int, telemetry_sizes,
             telemetry_intervals, attack_sizes):
    """Simulate one advisory window; returns per-class metrics."""
    rng = np.random.default_rng(seed)
    cell = Cell(cfg, rng, telemetry_sizes, telemetry_intervals, attack_sizes)
    n = len(cell.dist)
    slots = int(cfg.sim_ms / cfg.slot_ms)
    dl_slots = [int(d / cfg.slot_ms) for d in cfg.deadlines_ms]
    bits_per_rb = lambda se: se * cfg.rb_hz * cfg.slot_ms * 1e-3
    # queues: list per user of [bits_left, arrival_slot, deadline_slot, cls, total_bits]
    Q = [[] for _ in range(n)]
    arrivals = np.zeros(4); delivered = np.zeros(4); missed = np.zeros(4)
    lat = {c: [] for c in range(4)}
    served_bits = np.zeros(4)
    energy_j = 0.0
    # actuation burst at t = 0 (valves selected for irrigation by the advisory)
    valves = np.flatnonzero(cell.is_valve)
    cmd = rng.choice(valves, size=min(n_commands, len(valves)), replace=False) if n_commands > 0 else []
    for u in cmd:
        c = 0 if rng.random() < cfg.urgent_frac else 1
        Q[u].append([cfg.cmd_bytes * 8, 0, dl_slots[c], c, cfg.cmd_bytes * 8]); arrivals[c] += 1
    # telemetry: first report at a uniform phase of the resampled interval
    sensors = np.flatnonzero(~cell.is_valve & ~cell.compromised)
    next_rep = {int(u): int(rng.uniform(0, rng.choice(telemetry_intervals)) * 1000 / cfg.slot_ms) for u in sensors}
    attackers = np.flatnonzero(cell.compromised)
    for u in attackers:          # full-buffer flood
        for _ in range(50):
            b = int(rng.choice(attack_sizes) * 8)
            Q[u].append([b, 0, 10 ** 12, 3, b]); arrivals[3] += 1
    avg_thr = np.full(n, 1.0)
    Z = np.zeros(3)
    rr_ptr = 0
    uplink = ~cell.is_valve
    for t in range(slots):
        arr_t = np.zeros(3)
        if t == 0:
            arr_t[0], arr_t[1] = arrivals[0], arrivals[1]
        for u, ts in list(next_rep.items()):
            if ts == t:
                b = int(rng.choice(telemetry_sizes) * 8)
                Q[u].append([b, t, t + dl_slots[2], 2, b]); arrivals[2] += 1; arr_t[2] += 1
                next_rep[u] = t + max(1, int(rng.choice(telemetry_intervals) * 1000 / cfg.slot_ms))
        # deadline expiry
        miss_t = np.zeros(3)
        for u in range(n):
            q = Q[u]
            while q and q[0][2] < t:
                missed[q[0][3]] += 1; miss_t[q[0][3]] += 1; q.pop(0)
        for u in attackers:      # keep attackers backlogged
            if len(Q[u]) < 10:
                b = int(rng.choice(attack_sizes) * 8)
                Q[u].append([b, t, 10 ** 12, 3, b]); arrivals[3] += 1
        active = np.array([u for u in range(n) if Q[u]], int)
        if len(active) == 0:
            continue
        # fading per (active user, RB)
        h = rng.exponential(1.0, (len(active), cfg.n_rb))
        snr = np.where(uplink[active, None], cell.snr_ul[active, None], cell.snr_dl[active, None]) + 10 * np.log10(h)
        se = se_from_snr(snr, uplink[active, None])
        rate = bits_per_rb(se)                                   # bits per RB this slot
        backlog = np.array([sum(p[0] for p in Q[u]) for u in active], float)
        hol_slack = np.array([max(Q[u][0][2] - t, 0) for u in active], float)
        hol_cls = np.array([Q[u][0][3] for u in active])
        e_rb = np.where(uplink[active], (10 ** ((cfg.p_ue_dbm - 30) / 10) + cfg.ue_circuit_w),
                        10 ** ((cfg.p_gnb_dbm_per_rb - 30) / 10)) * cfg.slot_ms * 1e-3
        used_ul = np.zeros(len(active), bool)
        give = np.zeros(len(active))
        for k in range(cfg.n_rb):
            ok = (backlog - give > 0) & ~(uplink[active] & used_ul) & (rate[:, k] > 0)
            if not ok.any():
                break
            if scheduler == "RR":
                order = (np.arange(len(active)) - rr_ptr) % len(active)
                metric = -order.astype(float)
            elif scheduler == "PF":
                metric = rate[:, k] / avg_thr[active]
            elif scheduler == "EDF":
                metric = -hol_slack + 1e-6 * rate[:, k]
            elif scheduler == "MaxWeight":
                metric = (backlog - give) * rate[:, k]
            elif scheduler == "DRIP":
                urg = np.where(hol_cls < 3, (1.0 + Z[np.minimum(hol_cls, 2)]) / np.maximum(hol_slack, 1.0), 0.0)
                cls_w = np.where(hol_cls < 3, 1.0, 1e-3)
                metric = (urg + 1e-4 * cls_w * np.log1p(np.maximum(backlog - give, 0.0))) * rate[:, k] - cfg.drip_v_energy * e_rb * 1e3
            else:
                raise ValueError(scheduler)
            metric = np.where(ok, metric, -np.inf)
            i = int(np.argmax(metric))
            if not np.isfinite(metric[i]):
                break
            give[i] += rate[i, k]
            energy_j += e_rb[i]
            if uplink[active[i]]:
                used_ul[i] = True
            if scheduler == "RR":
                rr_ptr = (rr_ptr + 1) % len(active)
        # serve FIFO bits
        for i, u in enumerate(active):
            g = give[i]
            avg_thr[u] = 0.99 * avg_thr[u] + 0.01 * g
            while g > 0 and Q[u]:
                p = Q[u][0]
                take = min(g, p[0]); p[0] -= take; g -= take; served_bits[p[3]] += take
                if p[0] <= 0:
                    delivered[p[3]] += 1; lat[p[3]].append((t + 1 - p[1]) * cfg.slot_ms); Q[u].pop(0)
        if scheduler == "DRIP":   # virtual deadline-violation queues Z_c(t+1)=[Z_c+miss_c-eps_c*arr_c]^+
            Z = np.maximum(Z + miss_t - np.array(cfg.drip_eps) * arr_t, 0.0)
    # packets still queued at the end whose deadline lies beyond the window are not counted
    out = {}
    for c, name in enumerate(["urgent", "actuation", "telemetry", "attack"]):
        done = delivered[c] + missed[c]
        out[name] = {"arrivals": float(arrivals[c]), "delivered": float(delivered[c]), "missed": float(missed[c]),
                     "miss_ratio": float(missed[c] / done) if done > 0 else 0.0,
                     "lat_mean_ms": float(np.mean(lat[c])) if lat[c] else float("nan"),
                     "lat_p95_ms": float(np.percentile(lat[c], 95)) if lat[c] else float("nan"),
                     "served_kbit": float(served_bits[c] / 1e3)}
    useful = served_bits[:3].sum()
    out["energy_j"] = float(energy_j)
    out["energy_uj_per_useful_bit"] = float(energy_j / max(useful, 1) * 1e6)
    return out
