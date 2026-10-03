"""E6 - Fast-timescale 6G RAN experiments for the agricultural slice.

F1  scheduler comparison in an irrigation-advisory window (150 sensors, 50 valves)
F2  scalability in the number of devices
F3  robustness to flood traffic from compromised sensors (Farm-Flow-calibrated)
F4  FR1 (3.5 GHz, 30 kHz SCS) versus upper mid-band (7 GHz, 60 kHz SCS, wider slice)
Synthetic traffic is produced by terai.synth from the real Farm-Flow records and the real
Mendeley inter-arrival times.  Output: results/raw/e6_ran.json
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from terai import config as C, ran as R, synth as SY  # noqa: E402

SCHED = ["RR", "PF", "EDF", "MaxWeight", "DRIP"]
N_SEEDS = int(os.environ.get("E6_SEEDS", 20))


def traffic():
    d = SY.load_farmflow()
    g = SY.GaussianCopulaFlows(d, seed=0)
    syn = g.sample({"Normal": 20000, "MQTT_Flood": 5000, "UDP_Flood": 5000, "TCP_Flood": 5000, "HTTP_Flood": 5000})
    tel = syn[syn.traffic == "Normal"].orig_ip_bytes.clip(60, 5000).values
    att = syn[syn.traffic != "Normal"]
    att = (att.orig_ip_bytes / att.orig_pkts.clip(lower=1)).clip(40, 1500).values
    iv = SY.mendeley_intervals()
    return tel, iv, att


_T = {}


def run(job):
    if not _T:
        _T["tel"], _T["iv"], _T["att"] = traffic()
    exp, key, sched, seed, kw = job
    cfg = R.RanConfig(sim_ms=1000.0, **kw)
    t0 = time.time()
    o = R.simulate(cfg, sched, seed, n_commands=cfg.n_valves, telemetry_sizes=_T["tel"],
                   telemetry_intervals=_T["iv"], attack_sizes=_T["att"])
    o["sim_s"] = time.time() - t0
    return exp, key, sched, seed, o


def main():
    t0 = time.time()
    jobs = []
    for s in range(N_SEEDS):
        for sc in SCHED:
            jobs.append(("F1", "base", sc, s, {}))
    for (ns, nv) in [(75, 25), (150, 50), (300, 100), (600, 200), (1200, 400)]:
        for s in range(min(N_SEEDS, 5)):
            for sc in SCHED:
                jobs.append(("F2", "%d" % (ns + nv), sc, s, {"n_sensors": ns, "n_valves": nv}))
    for fr in [0.0, 0.05, 0.1, 0.2]:
        for s in range(min(N_SEEDS, 10)):
            for sc in SCHED:
                jobs.append(("F3", "%.2f" % fr, sc, s, {"compromised_frac": fr}))
    fr3 = {"fc_ghz": 7.0, "n_rb": 27, "scs_khz": 60.0, "slot_ms": 0.25,
           "p_gnb_dbm_per_rb": 46.0 + 10 * np.log10(100 / 20) - 10 * np.log10(135)}
    for s in range(min(N_SEEDS, 10)):
        for sc in SCHED:
            jobs.append(("F4", "FR1 3.5 GHz", sc, s, {}))
            jobs.append(("F4", "FR3 7 GHz", sc, s, fr3))
    out = {"meta": {"seeds": N_SEEDS, "fr3": fr3}, "runs": []}
    tel, iv, att = traffic()
    out["meta"]["telemetry_bytes_median"] = float(np.median(tel))
    out["meta"]["attack_pkt_bytes_median"] = float(np.median(att))
    out["meta"]["interval_s_median"] = float(np.median(iv))
    with Pool(int(os.environ.get("E6_PROCS", 2))) as pool:
        for r in pool.imap_unordered(run, jobs, chunksize=2):
            out["runs"].append(r)
    out["meta"]["runtime_s"] = time.time() - t0
    os.makedirs(C.RES_RAW, exist_ok=True)
    json.dump(out, open(os.path.join(C.RES_RAW, "e6_ran.json"), "w"))
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
