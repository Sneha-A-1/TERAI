"""Generate every table, figure and in-text number of the manuscript from results/raw.

Outputs
  manuscript/results_macros.tex      all numbers quoted in the text (\newcommand)
  manuscript/tables/*.tex            LaTeX tables (single numbers; means and std in separate columns)
  manuscript/figures/*.pdf           figures (also copied to results/figures)
  results/tables/*.csv               the same tables as CSV
Run:  python scripts/make_outputs.py
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from terai import config as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
MAN = os.path.join(ROOT, "manuscript")
TAB = os.path.join(MAN, "tables")
FIG = os.path.join(MAN, "figures")
CSV = os.path.join(ROOT, "results", "tables")
FIGR = os.path.join(ROOT, "results", "figures")
for d in [TAB, FIG, CSV, FIGR]:
    os.makedirs(d, exist_ok=True)

# validated categorical palette (fixed order) + neutral for de-emphasised marks
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, INK2 = "#9b9a96", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "grid.linestyle": "-",
    "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

MACROS = {}


NEG = __import__("re").compile(r"^-(?=[0-9.])")


def tex_minus(x):
    """Typeset a leading minus sign of a number as a math minus."""
    return NEG.sub("$-$", str(x))


def mac(name, value, fmt=None):
    if fmt is not None:
        value = fmt.format(value)
    MACROS[name] = tex_minus(value)


def load(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG, name))
    shutil.copy(os.path.join(FIG, name), os.path.join(FIGR, name))
    plt.close(fig)


def f4(x):
    return "%.4f" % x


def write_table(name, header, rows, caption, label, colspec, star=False, note=None, size=None):
    env = "table*" if star else "table"
    lines = ["\\begin{%s}[t]" % env, "\\caption{%s}" % caption, "\\label{%s}" % label, "\\centering"]
    if size:
        lines.append(size)
    lines += ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{%s}" % colspec, "\\toprule",
              " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(tex_minus(x) for x in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append("\\\\[2pt]\\parbox{\\linewidth}{\\scriptsize %s}" % note)
    lines.append("\\end{%s}" % env)
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")
    pd.DataFrame([r for r in rows if r != "MIDRULE"], columns=[h.replace("\\", "") for h in header]).to_csv(
        os.path.join(CSV, name + ".csv"), index=False)


def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank test and rank-biserial effect size (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[np.abs(d) > 1e-12]
    if len(d) < 3:
        return float("nan"), float("nan")
    p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float(p), float((wp - wm) / (wp + wm))


# =====================================================================================
# E5 data facts and synthetic fidelity
# =====================================================================================
def do_data():
    e5 = load("e5_data_synth.json")
    e1 = load("e1_forecast.json")
    if e5 is None:
        return
    d = e5["data"]
    mac("panelDays", d["panel_days"]); mac("smapDays", d["smap_days_any"])
    mac("sOneRaw", d["s1_raw_rows"]); mac("sOneMissing", d["s1_empty_rows"])
    mac("eraSmapCorr", d["era5_smap_pearson"], "{:.2f}"); mac("eraSmapN", d["era5_smap_n"])
    mac("smapRecommendedPct", 100 * d["smap_recommended_frac"], "{:.1f}")
    mac("smapRetrievedPct", 100 * d["smap_retrieved_frac"], "{:.1f}")
    mac("ffRecords", e5["farmflow_records"])
    mac("ffNormal", e5["farmflow_class_counts"]["Normal"])
    mac("mendeleyIntervals", e5["mendeley"]["n_intervals"]); mac("mendeleyMedian", e5["mendeley"]["median_s"], "{:.0f}")
    ls = e5["landsat"]
    mac("landsatMB", "%.0f--%.0f" % (min(x["sr_bytes"] for x in ls) / 1e6, max(x["sr_bytes"] for x in ls) / 1e6))
    mac("sOneSubsetMB", np.mean(list(e5["s1_subset_mb"].values())), "{:.1f}")
    fd = e5["fidelity"]
    mac("ksMax", max(fd["ks"].values()), "{:.3f}"); mac("corrDist", fd["corr_frobenius"], "{:.3f}")
    mac("cstAUC", fd["c2st_auc"], "{:.2f}")
    mac("telRealMedian", e5["telemetry_bytes"]["real_median"], "{:.0f}")
    mac("telSynMedian", e5["telemetry_bytes"]["synthetic_median"], "{:.0f}")
    if e1:
        mac("testDays", e1["meta"]["n_test_days"]); mac("testTargets", e1["meta"]["n_test_targets"])
        mac("trainDays", e1["meta"]["n_train_days"])
    # dataset table
    rows = [
        ["SMAP SPL3SMP\\_E v6 \\cite{oneill2023smap}", "passive microwave", "regional (722 cells)", "%d of %d days" % (d["smap_days_any"], d["panel_days"]),
         "soil moisture, VWC, $T_s$, retrieval fraction", "target \\& input", "real"],
        ["IMD gridded \\cite{pai2014imd}", "gauge analysis", "5 districts", "daily (%d rows)" % d["imd_rows"], "rain, 3/7-day rain, $T_{\\max}$, $T_{\\min}$", "input", "real"],
        ["NASA POWER \\cite{sparks2018nasapower}", "reanalysis, satellite", "4 districts$^{a}$", "daily (762/district)", "T2M, DTR, RH, rain, irradiance, wind", "input, PV energy", "real"],
        ["Sentinel-1 VV/VH \\cite{torres2012sentinel1}", "C-band SAR", "5 districts", "%d valid of %d rows" % (d["s1_valid_rows"], d["s1_raw_rows"]), "VV, VH, VV$-$VH, $\\Delta$VV", "input", "real"],
        ["ERA5-Land \\cite{munoz2021era5}", "reanalysis", "99 cells, Dehradun", "6-hourly, 10 months 2024", "soil water layer 1", "validation only", "real"],
        ["Mendeley IoT log \\cite{kathole2024mendeley}", "in-field sensors", "one farm", "100 readings, 30 min", "8 sensor channels", "timing calibration", "real"],
        ["Farm-Flow \\cite{ferreira2024farmflowdata}", "network flows", "testbed", "%d unique flows used" % e5["farmflow_records"], "10 flow features, 9 classes", "traffic calibration", "real subset"],
        ["Landsat 8/9 C2 L2 metadata", "optical (metadata)", "5 scenes", "Oct. 2022", "raster dimensions", "payload sizes", "real"],
        ["Synthetic flows and devices", "generated", "rural 6G cells", "seeded", "copula flows, timing", "RAN workloads", "synthetic"],
    ]
    write_table("datasets", ["Dataset", "Modality", "Coverage", "Samples", "Features", "Role", "Type"], rows,
                "Datasets used in this article. Every learning result uses the real data; the synthetic dataset drives network workloads only.",
                "tab:data", "@{}p{2.6cm}p{1.6cm}p{1.6cm}p{2.2cm}p{3.1cm}p{1.9cm}p{1.1cm}@{}", star=True,
                note="$^{a}$Haridwar is not covered; the Almora and Nainital series are identical (same grid cell).")
    rows = [[f.replace("_", "\\_"), "%.3f" % fd["ks"][f], "%.2f" % fd["ks_p"][f]] for f in fd["ks"]]
    rows += ["MIDRULE", ["Correlation distance (Frobenius, relative)", "%.3f" % fd["corr_frobenius"], "--"],
             ["Classifier two-sample AUC", "%.3f" % fd["c2st_auc"], "--"]]
    write_table("fidelity", ["Feature (log scale)", "KS statistic", "KS $p$-value"], rows,
                "Fidelity of the synthetic Farm-Flow-calibrated flows (mean over 5 seeds).", "tab:fidelity",
                "@{}lrr@{}")


# =====================================================================================
# parameter table (from the code configuration)
# =====================================================================================
def do_params():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from terai.env import SlowConfig
    from terai.ran import RanConfig
    from terai.marl import PPOConfig, QMIXConfig
    s, r, p, q, e = SlowConfig(), RanConfig(), PPOConfig(), QMIXConfig(), C.EnergyConfig()
    rows = [
        ["\\multicolumn{3}{@{}l}{\\emph{Slow timescale (daily)}}"],
        ["Districts / hill sites / plains sites", "$N$, $|\\Ncal_H|$, $|\\Ncal_P|$", "5 / 3 / 2"],
        ["Publication latency IMD, POWER, S1, SMAP$^{\\dagger}$", "$L_m$", "%d, %d, %d, %d d" % tuple(C.LATENCY[m] for m in C.MODALITIES)],
        ["Maximum age", "$A_{\\max}$", "%d d" % C.A_MAX],
        ["LEO / terrestrial slots per day$^{\\dagger}$", "$K_H$, $K_P$", "%d, %d" % (s.k_ntn, s.k_terr)],
        ["Slot cost IMD, POWER, S1", "$c_m$", "1, 1, 2"],
        ["Ka-band frequency, elevation", "$f$, $\\theta$", "%.0f GHz, %.0f$^\\circ$" % (s.ntn_freq_ghz, s.ntn_elev_deg)],
        ["Clear-sky / threshold $E_s/N_0$, margin s.d.", "$\\rho_{\\text{clear}},\\rho_{\\text{th}},\\sigma_\\rho$", "%.1f, %.2f, 1.5 dB" % (s.ntn_clear_snr_db, s.ntn_outage_snr_db)],
        ["Rain duration for rate conversion$^{\\dagger}$", "$H_w$", "%.0f h" % s.wet_hours],
        ["Terrestrial slot success", "$p_P$", "0.995"],
        ["Terminal power, standby, slot length$^{\\dagger}$", "$P_T,P_{\\text{sb}},\\tau_{\\text{sl}}$", "%.0f W, %.1f W, %.0f s" % (s.ntn_terminal_w, s.ntn_standby_w, s.slot_s)],
        ["PV rating, performance ratio, battery$^{\\dagger}$", "$P_{\\text{pv}},\\eta,B_{\\max}$", "%.0f W, %.1f, %.0f Wh" % (s.pv_peak_w, s.pv_pr, s.battery_wh)],
        ["SAR / report processing energy$^{\\dagger}$", "$e^{\\text{proc}}_m$", "%.1f, %.1f Wh" % (s.s1_proc_wh, s.report_wh)],
        ["Grid emission factor (FY 2022--23)", "$\\kappa$", "%.3f kgCO$_2$/kWh \\cite{cea2023co2}" % e.grid_kgco2_per_kwh],
        ["Energy weight, normaliser, irrigation threshold", "$w_E,E_{\\text{ref}},\\theta$", "%.2f, %.0f Wh, %.2f" % (s.w_energy, s.e_ref_wh, s.sm_threshold)],
        ["Constraint targets", "$\\epsilon_O,\\epsilon_D$", "%.2f, %.2f" % (p.eps_overflow, p.eps_depletion)],
        ["\\multicolumn{3}{@{}l}{\\emph{Learning}}"],
        ["SCAF width, optimiser, weight decay", "$d$", "32, Adam $3\\times10^{-3}$, $10^{-3}$"],
        ["PPO updates, envs, episode, $\\gamma$, $\\lambda_{\\text{GAE}}$", "", "%d, %d, %d d, %.2f, %.2f" % (p.updates, p.n_envs, s.episode_days, p.gamma, p.gae_lambda)],
        ["PPO clip, lr, epochs, entropy", "", "%.1f, $3\\times10^{-4}$, %d, %.3f$\\to$%.3f" % (p.clip, p.epochs, p.ent_start, p.ent_end)],
        ["CoVIR BC rollouts; warm-up; fine-tuning lr, entropy", "$U_w$", "40$\\times$8 episodes; 10; $10^{-4}$, 0.005$\\to$0.0005"],
        ["Checkpoint selection (all learned methods)", "$U_s$", "every 10 updates, 16 validation episodes"],
        ["QMIX steps, buffer, batch, $\\epsilon$ schedule", "", "%d$\\times$%d, %d, %d, %.1f$\\to$%.2f" % (q.env_steps, q.n_envs, q.buffer, q.batch, q.eps_start, q.eps_end)],
        ["Dual step size; VoI threshold", "$\\eta_\\lambda$; $\\tau_\\nu$", "%.2f; 0.05" % p.lam_lr],
        ["\\multicolumn{3}{@{}l}{\\emph{Fast timescale (RAN slice)}}"],
        ["Carrier, SCS, slot, slice RBs", "", "%.1f GHz, %.0f kHz, %.2f ms, %d" % (r.fc_ghz, r.scs_khz, r.slot_ms, r.n_rb)],
        ["Channel model, cell radius, heights", "", "RMa \\cite{tr38901}, %.0f m, %.0f/%.1f m" % (r.radius_m, r.h_bs, r.h_ut)],
        ["gNB power per RB, antenna gain, NF", "", "%.1f dBm, %.0f dBi, %.0f dB" % (r.p_gnb_dbm_per_rb, r.g_bs_dbi, r.nf_gnb_db)],
        ["Device power, NF", "", "%.0f dBm, %.0f dB" % (r.p_ue_dbm, r.nf_ue_db)],
        ["Sensors, valves, urgent fraction", "", "%d, %d, %.1f" % (r.n_sensors, r.n_valves, r.urgent_frac)],
        ["Deadlines urgent, routine, telemetry", "$D_c$", "%.0f, %.0f, %.0f ms" % r.deadlines_ms[:3]],
        ["Command size; DRIP $V$, $\\epsilon_c$", "", "%d B; %.2f, (%.2f, %.2f, %.2f)" % ((r.cmd_bytes, r.drip_v_energy) + tuple(r.drip_eps))],
        ["Simulated window per run", "", "%.0f ms" % 1000.0],
    ]
    lines = ["\\begin{table}[t]", "\\caption{Simulation and model parameters ($^{\\dagger}$ modelling assumption).}",
             "\\label{tab:params}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
             "\\begin{tabular}{@{}p{3.6cm}p{1.6cm}p{2.9cm}@{}}", "\\toprule", "Parameter & Symbol & Value \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(TAB, "params.tex"), "w").write("\n".join(lines) + "\n")


# =====================================================================================
# E1 forecasting
# =====================================================================================
FORECAST_ORDER = ["Persistence", "Climatology", "Ridge (early fusion)", "Late fusion (ridge)",
                  "Gradient boosting (early fusion)", "MLP (early fusion)", "SCAF without staleness gate",
                  "SCAF (proposed)", "SCAF ensemble (environment estimator)"]


def agg_e1(e1, key, regime, metric):
    v = [x[regime][metric] for x in e1[key]]
    return float(np.mean(v)), float(np.std(v)), len(v)


def do_forecast():
    e1 = load("e1_forecast.json")
    if e1 is None:
        return
    rows = []
    for k in FORECAST_ORDER:
        fr = {m: agg_e1(e1, k, "fresh", m) for m in ["rmse", "mae", "r2", "decision_err"]}
        st = {m: agg_e1(e1, k, "stale", m) for m in ["rmse", "r2", "decision_err"]}
        name = k.replace("SCAF (proposed)", "\\textbf{SCAF (proposed)}")
        rows.append([name, fr["rmse"][2], f4(fr["rmse"][0]), f4(fr["rmse"][1]), f4(fr["mae"][0]), "%.3f" % fr["r2"][0],
                     "%.3f" % fr["decision_err"][0], f4(st["rmse"][0]), f4(st["rmse"][1]), "%.3f" % st["r2"][0],
                     "%.3f" % st["decision_err"][0]])
    write_table("forecast", ["Model", "Runs", "RMSE", "RMSE s.d.", "MAE", "$R^2$", "Dec. err.", "RMSE", "RMSE s.d.", "$R^2$", "Dec. err."],
                rows, "Regional soil-moisture estimation on the unseen 2024 season (%d target days). Left: fresh inputs; right: random staleness (5 fixed draws). RMSE and MAE in m$^3$m$^{-3}$." % e1["meta"]["n_test_targets"],
                "tab:forecast", "@{}lrrrrrr|rrrr@{}", star=True)
    p = agg_e1(e1, "Persistence", "fresh", "rmse")[0]
    s = agg_e1(e1, "SCAF (proposed)", "fresh", "rmse")[0]
    ens = agg_e1(e1, "SCAF ensemble (environment estimator)", "fresh", "rmse")[0]
    best_b = min(agg_e1(e1, k, "fresh", "rmse")[0] for k in FORECAST_ORDER[:6])
    best_bs = min(agg_e1(e1, k, "stale", "rmse")[0] for k in FORECAST_ORDER[:6])
    nm = {"Persistence": "persistence", "Climatology": "climatology", "Ridge (early fusion)": "early-fusion ridge",
          "Late fusion (ridge)": "late fusion", "Gradient boosting (early fusion)": "gradient boosting",
          "MLP (early fusion)": "the early-fusion MLP"}
    mac("bestBaselineName", nm[min(FORECAST_ORDER[:6], key=lambda k: agg_e1(e1, k, "fresh", "rmse")[0])])
    mac("bestBaselineStaleName", nm[min(FORECAST_ORDER[:6], key=lambda k: agg_e1(e1, k, "stale", "rmse")[0])])
    ps = agg_e1(e1, "Persistence", "stale", "rmse")[0]; ss = agg_e1(e1, "SCAF (proposed)", "stale", "rmse")[0]
    mac("staleGainPct", 100 * (ps - ss) / ps, "{:.1f}")
    mac("persistRMSE", f4(p)); mac("scafRMSE", f4(s)); mac("scafEnsRMSE", f4(ens))
    mac("absFusionGainPct", 100 * (p - s) / p, "{:.1f}")
    mac("ensGainPct", 100 * (p - ens) / p, "{:.1f}")
    mac("bestBaselineRMSE", f4(best_b)); mac("bestBaselineStaleRMSE", f4(best_bs))
    mac("scafStaleRMSE", f4(agg_e1(e1, "SCAF (proposed)", "stale", "rmse")[0]))
    mac("persistStaleRMSE", f4(agg_e1(e1, "Persistence", "stale", "rmse")[0]))
    mac("scafRtwo", "%.3f" % agg_e1(e1, "SCAF (proposed)", "fresh", "r2")[0])
    mac("persistRtwo", "%.3f" % agg_e1(e1, "Persistence", "fresh", "r2")[0])
    mac("nogateRMSE", f4(agg_e1(e1, "SCAF without staleness gate", "fresh", "rmse")[0]))
    mac("nogateStaleRMSE", f4(agg_e1(e1, "SCAF without staleness gate", "stale", "rmse")[0]))
    mac("scafDecErr", "%.3f" % agg_e1(e1, "SCAF (proposed)", "fresh", "decision_err")[0])
    mac("lateDecErr", "%.3f" % agg_e1(e1, "Late fusion (ridge)", "fresh", "decision_err")[0])
    mac("persistDecErr", "%.3f" % agg_e1(e1, "Persistence", "fresh", "decision_err")[0])
    mac("lateRMSE", f4(agg_e1(e1, "Late fusion (ridge)", "fresh", "rmse")[0]))
    mac("mlpStaleRMSE", f4(agg_e1(e1, "MLP (early fusion)", "stale", "rmse")[0]))
    # modality ablation table
    rows = []
    for k, lab in [("SCAF (proposed)", "all modalities"), ("SCAF without IMD", "without IMD"), ("SCAF without POWER", "without POWER"),
                   ("SCAF without S1", "without Sentinel-1"), ("SCAF without SMAP", "without SMAP"),
                   ("SCAF without staleness gate", "without staleness gate")]:
        fr = agg_e1(e1, k, "fresh", "rmse"); st = agg_e1(e1, k, "stale", "rmse")
        rows.append([lab, fr[2], f4(fr[0]), f4(fr[1]), f4(st[0]), f4(st[1])])
    write_table("forecast_ablation", ["SCAF variant", "Runs", "Fresh RMSE", "s.d.", "Stale RMSE", "s.d."], rows,
                "SCAF ablations on the 2024 season (RMSE in m$^3$m$^{-3}$).", "tab:fabl", "@{}lrrrrr@{}")
    for k, nm in [("SCAF without IMD", "noIMD"), ("SCAF without SMAP", "noSMAP"), ("SCAF without S1", "noSone"), ("SCAF without POWER", "noPOWER")]:
        mac(nm + "RMSE", f4(agg_e1(e1, k, "fresh", "rmse")[0]))
        mac(nm + "StaleRMSE", f4(agg_e1(e1, k, "stale", "rmse")[0]))
    att = e1["attention_regional"]
    mac("attSMAP", "%.3f" % att["SMAP"])
    mac("attUniform", "%.3f" % (1.0 / len(att)))


# =====================================================================================
# E2 value of freshness
# =====================================================================================
def do_voi():
    e2 = load("e2_voi.json")
    if e2 is None:
        return
    ages = e2["ages"]; te = e2["test"]
    mac("singleDistrictMaxVoI", f4(max(te["imd7_single_district"].values())))
    mac("allDistrictVoISeven", f4(te["curves"]["all_district"][ages.index(7)]))
    mac("imdSevenVoI", f4(te["imd7_all_districts"]))
    mac("smapThirtyVoI", f4(te["curves"]["SMAP"][ages.index(30)]))
    mac("allThirtyVoI", f4(te["curves"]["all_district"][ages.index(30)]))
    mac("allThirtyPct", 100 * te["curves"]["all_district"][ages.index(30)] / te["rmse_fresh"], "{:.0f}")
    mac("smapThirtyPct", 100 * te["curves"]["SMAP"][ages.index(30)] / te["rmse_fresh"], "{:.0f}")
    mac("imdThirtyVoI", f4(te["curves"]["IMD"][ages.index(30)]))
    mac("sOneThirtyVoI", f4(te["curves"]["S1"][ages.index(30)]))
    mac("powerThirtyVoI", f4(te["curves"]["POWER"][ages.index(30)]))
    mac("voiFreshRMSE", f4(te["rmse_fresh"]))
    mac("innovSpearmanAbs", "%.2f" % e2["innovation_proxy"]["spearman_innov_vs_abs_gain"])
    mac("innovSpearman", "%.2f" % e2["innovation_proxy"]["spearman_innov_vs_gain"])
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    series = [("all_district", "all district reports", S1, "o"), ("SMAP", "SMAP", S2, "s"), ("IMD", "IMD", S3, "^"),
              ("S1", "Sentinel-1", S4, "D"), ("POWER", "POWER", S5, "v")]
    for k, lab, col, mk in series:
        y = np.array(te["curves"][k]) * 1000
        ax.plot(ages, y, color=col, marker=mk, ms=3.5, label=lab)
    ax.set_xlabel("age of the held product (days)")
    ax.set_ylabel(r"RMSE increase ($10^{-3}$ m$^3$m$^{-3}$)")
    ax.set_xlim(0, 31)
    ax.legend(loc="upper left", ncol=1, handlelength=1.6)
    save_fig(fig, "fig_voi.pdf")


# =====================================================================================
# E3 slow timescale
# =====================================================================================
NAME = {("covir", "full"): "CoVIR (proposed)", ("ippo", "full"): "IPPO", ("central", "full"): "Centralized PPO",
        ("qmix", "full"): "QMIX \\cite{rashid2018qmix}", ("covir", "no_bc"): "without BC initialization",
        ("covir", "no_select"): "without checkpoint selection", ("covir", "no_lagrangian"): "without Lagrangian (fixed penalties)",
        ("covir", "no_voi"): "without VoI features (and BC)", ("covir", "decentral_critic"): "decentralized critic",
        ("covir", "crossfit_sim"): "cross-fitted training simulator"}
ABLATIONS = ["without BC initialization", "without checkpoint selection", "without Lagrangian (fixed penalties)",
             "without VoI features (and BC)", "decentralized critic", "cross-fitted training simulator"]
# (name, execution: C = centralized schedule within the budget, D = decentralized requests with contention)
SLOW_ORDER = [("Never refresh", "D"), ("Random", "D"), ("IMD+POWER daily", "D"), ("Round-robin", "C"),
              ("Max-age (Greedy)", "C"), ("Whittle index [Kadota]", "C"), ("VoI-greedy", "C"), ("Myopic optimum", "C"),
              ("Centralized PPO", "C"), "MIDRULE", ("VoI-threshold (decentralized)", "D"), ("IPPO", "D"),
              ("QMIX \\cite{rashid2018qmix}", "D"), ("CoVIR (proposed)", "D")]
SLOW_KEYS = [x[0] for x in SLOW_ORDER if x != "MIDRULE"]
DISPLAY = {"Max-age (Greedy)": "Max-age (Greedy) \\cite{kadota2018}", "Whittle index [Kadota]": "Whittle's index \\cite{kadota2018}",
           "VoI-threshold (decentralized)": "VoI threshold"}


def e3_by_method(e3):
    out = {}
    for name, seed, variant, res in e3["runs"]:
        key = NAME.get((name, variant), name)
        out.setdefault(key, {})[seed] = res
    return out


def do_slow():
    e3 = load("e3_slow.json")
    if e3 is None:
        return None
    M = e3_by_method(e3)
    metrics = ["rmse", "decision_err", "energy_wh_day", "carbon_g_day", "ntn_slots_day", "overflow_slots_day",
               "fail_per_day", "imd_deliv_day", "pw_deliv_day", "s1_deliv_day", "reward", "decision_ms"]

    def stat(key, m):
        v = [r[m] for r in M[key].values()]
        return float(np.mean(v)), float(np.std(v)), len(v)

    rows = []
    for item in SLOW_ORDER:
        if item == "MIDRULE":
            rows.append("MIDRULE")
            continue
        k, ex = item
        if k not in M:
            continue
        s = {m: stat(k, m) for m in metrics}
        nm = DISPLAY.get(k, k).replace("CoVIR (proposed)", "\\textbf{CoVIR (proposed)}")
        rows.append([nm, ex, s["rmse"][2], f4(s["rmse"][0]), f4(s["rmse"][1]), "%.3f" % s["decision_err"][0],
                     "%.2f" % s["energy_wh_day"][0], "%.2f" % s["energy_wh_day"][1], "%.2f" % s["ntn_slots_day"][0],
                     "%.2f" % s["overflow_slots_day"][0], "%.2f" % (s["imd_deliv_day"][0] + s["pw_deliv_day"][0]),
                     "%.2f" % s["s1_deliv_day"][0], "%.4f" % s["reward"][0]])
    write_table("slow_main", ["Policy", "Exec.", "Runs", "RMSE", "s.d.", "Dec. err.", "Energy", "s.d.", "LEO slots", "Dropped",
                              "Reports", "SAR", "Reward"], rows,
                "Daily data retrieval on the unseen 2024 season (305 days). Exec.: C = centralized schedule within the slot budget, D = decentralized requests that contend for the slots. RMSE in m$^3$m$^{-3}$; energy in Wh/day of data operations; LEO slots, dropped slots, delivered IMD/POWER reports, and SAR subsets per day; reward of (\\ref{eq:reward}). Means over runs; standard deviations in separate columns.",
                "tab:slow", "@{}llrrrrrrrrrrr@{}", star=True)
    # statistical tests vs CoVIR
    ref = "CoVIR (proposed)"
    rows = []
    if ref in M:
        seeds = sorted(M[ref])
        for k in SLOW_KEYS:
            if k == ref or k not in M or k == "Never refresh":
                continue
            common = [s for s in seeds if s in M[k]]
            line = [DISPLAY.get(k, k)]
            for m in ["rmse", "energy_wh_day", "reward"]:
                a = [M[ref][s][m] for s in common]; b = [M[k][s][m] for s in common]
                p, r = wilcoxon_paired(a, b)
                line += ["%.4f" % (np.mean(a) - np.mean(b)) if m != "energy_wh_day" else "%.2f" % (np.mean(a) - np.mean(b)),
                         ("%.3f" % p) if np.isfinite(p) else "--", ("%.2f" % r) if np.isfinite(r) else "--"]
            rows.append(line)
        write_table("stats", ["Baseline", "$\\Delta$RMSE", "$p$", "$r_{rb}$", "$\\Delta$Energy", "$p$", "$r_{rb}$", "$\\Delta$Reward", "$p$", "$r_{rb}$"],
                    rows, "Paired comparison of CoVIR with each baseline over seeds: difference of means (CoVIR minus baseline), two-sided Wilcoxon signed-rank $p$-value, and rank-biserial effect size $r_{rb}$.",
                    "tab:stats", "@{}lrrrrrrrrr@{}", star=True)
    # ablation (seeds common to all variants)
    abl = [ref] + ABLATIONS
    rows = []
    common = None
    for k in abl:
        if k in M:
            common = set(M[k]) if common is None else common & set(M[k])
    if common:
        common = sorted(common)
        for k in abl:
            if k not in M:
                continue
            v = lambda m: [M[k][s][m] for s in common]
            nm = {ref: "CoVIR (full)"}.get(k, k)
            rows.append([nm, len(common), f4(np.mean(v("rmse"))), f4(np.std(v("rmse"))), "%.3f" % np.mean(v("decision_err")),
                         "%.2f" % np.mean(v("energy_wh_day")), "%.3f" % np.mean(v("overflow_slots_day")), "%.4f" % np.mean(v("reward")),
                         "%.4f" % np.std(v("reward"))])
        write_table("slow_ablation", ["Variant", "Runs", "RMSE", "s.d.", "Dec. err.", "Energy", "Dropped", "Reward", "s.d."], rows,
                    "CoVIR ablations on the 2024 season (seeds common to all variants; energy in Wh/day, dropped LEO slots per day).",
                    "tab:sabl", "@{}lrrrrrrrr@{}", star=True)
        abl_nm = {"without BC initialization": "noBC", "without checkpoint selection": "noSel",
                  "without Lagrangian (fixed penalties)": "noLag", "without VoI features (and BC)": "noVoi",
                  "decentralized critic": "decCritic", "cross-fitted training simulator": "crossfit"}
        for k, nm in abl_nm.items():
            if k in M:
                v = lambda m: [M[k][s][m] for s in common]
                mac(nm + "RMSE", f4(np.mean(v("rmse")))); mac(nm + "Energy", "%.2f" % np.mean(v("energy_wh_day")))
                mac(nm + "Dropped", "%.3f" % np.mean(v("overflow_slots_day"))); mac(nm + "Reward", "%.4f" % np.mean(v("reward")))
                mac(nm + "RewardSD", "%.4f" % np.std(v("reward")))
        v = lambda m: [M[ref][s][m] for s in common]
        mac("covirAblRMSE", f4(np.mean(v("rmse")))); mac("covirAblReward", "%.4f" % np.mean(v("reward")))
        mac("covirAblRewardSD", "%.4f" % np.std(v("reward")))
    # macros
    g = lambda k, m: stat(k, m)[0] if k in M else float("nan")
    mac("covirRMSE", f4(g(ref, "rmse"))); mac("covirRMSEtext", "%.4f" % g(ref, "rmse"))
    mac("covirEnergy", "%.2f" % g(ref, "energy_wh_day")); mac("covirEnergytext", "%.1f" % g(ref, "energy_wh_day"))
    mac("covirDecErr", "%.3f" % g(ref, "decision_err")); mac("covirSlots", "%.2f" % g(ref, "ntn_slots_day"))
    mac("covirReward", "%.4f" % g(ref, "reward")); mac("covirDropped", "%.3f" % g(ref, "overflow_slots_day"))
    W = "Whittle index [Kadota]"
    mac("whittleRMSE", f4(g(W, "rmse"))); mac("whittleRMSEtext", "%.4f" % g(W, "rmse"))
    mac("whittleEnergy", "%.2f" % g(W, "energy_wh_day")); mac("whittleEnergytext", "%.1f" % g(W, "energy_wh_day"))
    mac("whittleDecErr", "%.3f" % g(W, "decision_err")); mac("whittleSlots", "%.2f" % g(W, "ntn_slots_day"))
    mac("whittleSAR", "%.2f" % g(W, "s1_deliv_day"))
    mac("energySavingPct", 100 * (g(W, "energy_wh_day") - g(ref, "energy_wh_day")) / g(W, "energy_wh_day"), "{:.0f}")
    Q = "QMIX \\cite{rashid2018qmix}"
    mac("qmixRMSE", f4(g(Q, "rmse"))); mac("qmixEnergy", "%.2f" % g(Q, "energy_wh_day")); mac("qmixDecErr", "%.3f" % g(Q, "decision_err"))
    mac("qmixReward", "%.4f" % g(Q, "reward")); mac("qmixDropped", "%.3f" % g(Q, "overflow_slots_day"))
    V = "VoI-greedy"
    mac("voigRMSE", f4(g(V, "rmse"))); mac("voigEnergy", "%.2f" % g(V, "energy_wh_day")); mac("voigDecErr", "%.3f" % g(V, "decision_err"))
    mac("voigReward", "%.4f" % g(V, "reward")); mac("voigTau", "%.2f" % e3["meta"]["tau"])
    for k, nm in [("IMD+POWER daily", "daily"), ("Never refresh", "never"), ("Myopic optimum", "myopic"), ("IPPO", "ippo"),
                  ("Centralized PPO", "central"), ("Max-age (Greedy)", "maxage"), ("Round-robin", "rr"), ("Random", "random"),
                  ("VoI-threshold (decentralized)", "voit")]:
        mac(nm + "RMSE", f4(g(k, "rmse"))); mac(nm + "Energy", "%.2f" % g(k, "energy_wh_day"))
        mac(nm + "DecErr", "%.3f" % g(k, "decision_err")); mac(nm + "Reward", "%.4f" % g(k, "reward"))
        mac(nm + "Dropped", "%.3f" % g(k, "overflow_slots_day"))
    mac("maxageSAR", "%.2f" % g("Max-age (Greedy)", "s1_deliv_day"))
    for k, nm in [(ref, "covir"), (W, "whittle"), (V, "voig"), ("IMD+POWER daily", "daily"), (Q, "qmix")]:
        mac(nm + "Carbon", "%.2f" % g(k, "carbon_g_day"))
        mac(nm + "Slots", "%.2f" % g(k, "ntn_slots_day"))
    mac("covirDecisionMs", "%.2f" % g(ref, "decision_ms")); mac("whittleDecisionMs", "%.2f" % g(W, "decision_ms"))
    mac("myopicDecisionMs", "%.1f" % g("Myopic optimum", "decision_ms"))
    mac("numSeeds", len(M[ref]) if ref in M else 0)
    mac("numAblSeeds", len(e3["meta"]["abl_seeds"]))
    mac("interactionBudget", "%d{,}%03d" % divmod(e3["meta"]["updates"] * 8 * 60 * 5, 1000))
    if ref in M:
        lam = [M[ref][s]["final_lambda"][0] for s in M[ref] if "final_lambda" in M[ref][s]]
        mac("finalLambdaO", "%.2f" % np.mean(lam))
        mac("finalLambdaD", "%.2f" % np.mean([M[ref][s]["final_lambda"][1] for s in M[ref] if "final_lambda" in M[ref][s]]))
        bca = [M[ref][s]["bc"]["bc_accuracy"] for s in M[ref] if M[ref][s].get("bc")]
        if bca:
            mac("bcAccuracy", "%.3f" % np.mean(bca))
        su = [M[ref][s]["selected_update"] for s in M[ref] if M[ref][s].get("selected_update") is not None]
        if su:
            mac("selUpdateMedian", "%d" % int(np.median(su)) if len(su) else "--")
            mac("selUpdateMin", "%d" % int(np.min(su))); mac("selUpdateMax", "%d" % int(np.max(su)))
        mac("covirRewardSD", "%.4f" % stat(ref, "reward")[1]); mac("covirRMSESD", f4(stat(ref, "rmse")[1]))
        mac("covirTrainMin", "%.0f" % (np.mean([M[ref][s]["train_time_s"] for s in M[ref]]) / 60))
    if Q in M:
        mac("qmixRewardSD", "%.4f" % stat(Q, "reward")[1])
    for k, nm in [("IPPO", "ippo"), ("Centralized PPO", "central")]:
        if k in M:
            mac(nm + "RewardSD", "%.4f" % stat(k, "reward")[1])
    for k, nm in [("VoI-threshold (decentralized)", "Voit"), ("VoI-greedy", "Voig"), (Q, "Qmix"), ("IPPO", "Ippo")]:
        if ref in M and k in M:
            common = [s for s in M[ref] if s in M[k]]
            mac("covirBetterThan" + nm + "Seeds", sum(M[ref][s]["reward"] > M[k][s]["reward"] for s in common))
            a = [M[ref][s]["reward"] for s in common]; b = [M[k][s]["reward"] for s in common]
            p, r = wilcoxon_paired(a, b)
            mac("pReward" + nm, "%.3f" % p); mac("rbReward" + nm, "%.2f" % r)
            mac("rewardGainPct" + nm, 100 * (np.mean(a) - np.mean(b)) / abs(np.mean(b)), "{:.0f}")
    va = [g(k, "rmse") for k in [ref, "VoI-greedy", "VoI-threshold (decentralized)", "Myopic optimum"] if k in M]
    if va:
        mac("valueAwareRMSESpread", "%.4f" % (max(va) - min(va)))
    if ref in M:
        mac("covirDepleted", "%.2f" % g(ref, "depleted_days"))
        mac("energyRatioWhittle", "%.1f" % (g(W, "energy_wh_day") / g(ref, "energy_wh_day")))
    # rank of CoVIR among decentralized policies / all policies by reward
    allr = {k: g(k, "reward") for k in SLOW_KEYS if k in M}
    if ref in allr:
        order = sorted(allr, key=lambda k: -allr[k])
        mac("covirRewardRank", order.index(ref) + 1); mac("numPolicies", len(order))
    # trade-off figure (energy on a log axis: policies span an order of magnitude)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    LAB = {"Whittle index [Kadota]": "Whittle [Kadota] = Max-age", "Max-age (Greedy)": None, Q: "QMIX",
           "IMD+POWER daily": "IMD+POWER daily", "Round-robin": "round-robin", "Random": "random", "IPPO": "IPPO",
           "Centralized PPO": "central PPO"}
    OFF = {"Whittle index [Kadota]": ((-6, 7), "right"), Q: ((6, -6), "left"), "IMD+POWER daily": ((0, 8), "center"),
           "Round-robin": ((-6, 3), "right"), "Random": ((-6, 3), "right"), "IPPO": ((-6, 3), "right"),
           "Centralized PPO": ((6, 2), "left")}
    LEG = {ref: ("CoVIR (proposed)", S1, "*", 8), "VoI-greedy": ("VoI-greedy (centralized)", S4, "^", 5),
           "VoI-threshold (decentralized)": ("VoI threshold (decentralized)", S5, "v", 5),
           "Myopic optimum": ("myopic optimum", GRAY, "D", 4)}
    for k in SLOW_KEYS:
        if k not in M or k == "Never refresh":
            continue
        e, es = stat(k, "energy_wh_day")[:2]; r_, rs = stat(k, "rmse")[:2]
        if k in LEG:
            lab, col, mk, ms = LEG[k]
            ax.errorbar(e, r_ * 1000, yerr=rs * 1000, fmt=mk, color=col, ms=ms, elinewidth=0.6, capsize=1.5,
                        zorder=6 if k == ref else 4, mec="white", mew=0.6, label=lab)
            continue
        col, z, mk = GRAY, 2, "o"
        if k == W:
            col, z, mk = S2, 4, "s"
        elif k == Q:
            col, z, mk = S3, 4, "D"
        ax.errorbar(e, r_ * 1000, yerr=rs * 1000, fmt=mk, color=col, ms=4.5, elinewidth=0.6,
                    capsize=1.5, zorder=z, mec="white", mew=0.6)
        lab = LAB.get(k, k)
        if lab:
            off, ha = OFF.get(k, ((4, 3), "left"))
            ax.annotate(lab, (e, r_ * 1000), xytext=off, textcoords="offset points", fontsize=6.3, color=INK2, ha=ha,
                        va="center")
    hs, ls = ax.get_legend_handles_labels()
    want = [LEG[k][0] for k in [ref, "VoI-greedy", "VoI-threshold (decentralized)", "Myopic optimum"]]
    idx = [ls.index(w) for w in want if w in ls]
    ax.legend([hs[i] for i in idx], [ls[i] for i in idx], loc="upper left", fontsize=6.1, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.3)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20]); ax.set_xlim(1.3, 30)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: "%g" % v))
    ax.set_xlabel("data-operation energy (Wh/day, log scale)")
    ax.set_ylabel(r"estimation RMSE ($10^{-3}$ m$^3$m$^{-3}$)")
    save_fig(fig, "fig_tradeoff.pdf")
    # convergence figure
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.95))
    curves = [(ref, S1, "CoVIR"), ("without BC initialization", S5, "CoVIR w/o BC"), ("IPPO", S2, "IPPO"),
              ("Centralized PPO", S3, "central PPO")]
    for k, col, lab in curves:
        if k not in M:
            continue
        H = np.array([[h["reward"] for h in M[k][s]["history"]] for s in M[k]])
        x = np.arange(H.shape[1])
        sm = pd.DataFrame(H.T).rolling(10, min_periods=1).mean().values.T
        mu, sd = sm.mean(0), sm.std(0)
        axs[0].plot(x, mu, color=col, label=lab, lw=1.1); axs[0].fill_between(x, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
    axs[0].set_xlabel("PPO update"); axs[0].set_ylabel("training reward (per day)")
    for k, col, lab in curves + [(Q, S4, "QMIX")]:
        if k not in M or not M[k][min(M[k])].get("selection_log"):
            continue
        logs = [M[k][s]["selection_log"] for s in M[k]]
        n = min(len(l) for l in logs)
        Y = np.array([[x["reward"] for x in l[:n]] for l in logs])
        X = np.array([x["update"] for x in logs[0][:n]], float)
        X = X / (e3["meta"]["qmix_steps"] if k == Q else e3["meta"]["updates"] - 1)
        axs[1].plot(X, Y.mean(0), color=col, label=lab, lw=1.1, marker="o", ms=2)
    axs[1].set_xlabel("fraction of interaction budget"); axs[1].set_ylabel("greedy validation reward")
    axs[1].legend(loc="lower right", fontsize=6, ncol=1)
    if ref in M:
        L = np.array([[h["lam_o"] for h in M[ref][s]["history"]] for s in M[ref]])
        O = np.array([[h["overflow"] for h in M[ref][s]["history"]] for s in M[ref]])
        x = np.arange(L.shape[1])
        axs[2].plot(x, L.mean(0), color=S1, label=r"multiplier $\lambda_O$")
        axs[2].plot(x, pd.Series(O.mean(0)).rolling(10, min_periods=1).mean(), color=S2, label="dropped-slot rate")
        axs[2].axhline(0.02, color=GRAY, lw=0.8, ls="--")
        axs[2].set_xlabel("PPO update (CoVIR)"); axs[2].legend(loc="upper right", fontsize=6)
    for a, t in zip(axs, ["(a)", "(b)", "(c)"]):
        a.set_title(t, loc="left")
    fig.tight_layout()
    save_fig(fig, "fig_convergence.pdf")
    return M


# =====================================================================================
# E4 policy analysis
# =====================================================================================
def do_policy():
    e4 = load("e4_policy.json")
    if e4 is None:
        return
    groups = {}
    for r in e4["runs"]:
        groups.setdefault(r["name"], []).append(r)
    rows = []
    for k, lab in [("CoVIR (proposed)", "\\textbf{CoVIR (proposed)}"), ("CoVIR without BC", "CoVIR without BC initialization"),
                   ("QMIX", "QMIX \\cite{rashid2018qmix}"), ("VoI-threshold (decentralized)", "VoI threshold (decentralized)"),
                   ("VoI-greedy", "VoI-greedy (centralized)"), ("Whittle index [Kadota]", "Whittle's index \\cite{kadota2018}"),
                   ("Max-age (Greedy)", "Max-age (Greedy) \\cite{kadota2018}"), ("IMD+POWER daily", "IMD+POWER daily")]:
        if k not in groups:
            continue
        g = groups[k]
        tr = lambda m: np.mean([x["train_period"][m] for x in g]); te = lambda m: np.mean([x["test"][m] for x in g])
        rows.append([lab, len(g), f4(tr("rmse")), "%.2f" % tr("energy_wh_day"), "%.4f" % tr("reward"),
                     f4(te("rmse")), "%.2f" % te("energy_wh_day"), "%.4f" % te("reward")])
    write_table("policy_analysis", ["Policy", "Runs", "RMSE", "Energy", "Reward", "RMSE", "Energy", "Reward"], rows,
                "Generalization: replay of the training period (15 Oct. 2022 to 31 Dec. 2023, left) versus the unseen 2024 season (right), for the same trained policies (seeds 0--%d).  Energy in Wh/day." % (len(e4["runs"]) and max(r["seed"] for r in e4["runs"])),
                "tab:policy", "@{}lrrrr|rrr@{}", star=True)
    for k, nm in [("CoVIR (proposed)", "covir"), ("CoVIR without BC", "noBC"), ("QMIX", "qmix"),
                  ("VoI-threshold (decentralized)", "voit"), ("VoI-greedy", "voig"), ("Whittle index [Kadota]", "whittle")]:
        if k in groups:
            g = groups[k]
            mac(nm + "TrainPeriodReward", "%.4f" % np.mean([x["train_period"]["reward"] for x in g]))
            mac(nm + "TrainPeriodRMSE", f4(np.mean([x["train_period"]["rmse"] for x in g])))
            mac(nm + "EfourTestReward", "%.4f" % np.mean([x["test"]["reward"] for x in g]))
    # budget figure
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.0))
    for k, col, mk, lab in [("CoVIR (proposed)", S1, "o", "CoVIR"), ("QMIX", S3, "D", "QMIX"),
                            ("Whittle index [Kadota]", S2, "s", "Whittle [Kadota]"), ("VoI-greedy", S4, "^", "VoI-greedy (centralized)"),
                            ("VoI-threshold (decentralized)", S5, "v", "VoI threshold (decentralized)"),
                            ("IMD+POWER daily", GRAY, "x", "IMD+POWER daily")]:
        if k not in groups:
            continue
        Ks = sorted(int(x) for x in groups[k][0]["budget"])
        r_ = [np.mean([x["budget"][str(K)]["rmse"] for x in groups[k]]) * 1000 for K in Ks]
        e_ = [np.mean([x["budget"][str(K)]["energy_wh_day"] for x in groups[k]]) for K in Ks]
        axs[0].plot(Ks, r_, color=col, marker=mk, label=lab, lw=1.2); axs[1].plot(Ks, e_, color=col, marker=mk, label=lab, lw=1.2)
        if k in ("CoVIR (proposed)", "Whittle index [Kadota]", "VoI-threshold (decentralized)", "QMIX"):
            nm = {"CoVIR (proposed)": "covir", "Whittle index [Kadota]": "whittle", "VoI-threshold (decentralized)": "voit", "QMIX": "qmix"}[k]
            mac(nm + "RMSEKtwo", f4(r_[0] / 1000)); mac(nm + "RMSEKeight", f4(r_[-1] / 1000))
            mac(nm + "EnergyKtwo", "%.2f" % e_[0]); mac(nm + "EnergyKeight", "%.2f" % e_[-1])
            mac(nm + "DroppedKtwo", "%.3f" % np.mean([x["budget"]["2"]["overflow_slots_day"] for x in groups[k]]))
    axs[0].set_xlabel("LEO slots per day $K_H$"); axs[0].set_ylabel(r"RMSE ($10^{-3}$ m$^3$m$^{-3}$)")
    axs[1].set_xlabel("LEO slots per day $K_H$"); axs[1].set_ylabel("energy (Wh/day)")
    axs[1].set_yscale("log")
    axs[1].set_yticks([2, 3, 5, 10, 20, 30])
    axs[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: "%g" % v))
    axs[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.3)
    for a, t in zip(axs, ["(a)", "(b)"]):
        a.set_title(t, loc="left")
        a.set_xticks([2, 3, 4, 6, 8])
    fig.tight_layout()
    save_fig(fig, "fig_budget.pdf")
    # rain-fade margin sensitivity (clear-sky Es/N0), no retraining
    if "link" in e4["runs"][0]:
        snrs = sorted(e4["runs"][0]["link"], key=lambda x: -float(x))
        rows = []
        for k, lab in [("CoVIR (proposed)", "\\textbf{CoVIR (proposed)}"), ("QMIX", "QMIX \\cite{rashid2018qmix}"),
                       ("VoI-threshold (decentralized)", "VoI threshold"), ("VoI-greedy", "VoI-greedy"),
                       ("Whittle index [Kadota]", "Whittle's index \\cite{kadota2018}"), ("Max-age (Greedy)", "Max-age (Greedy)")]:
            if k not in groups:
                continue
            g = groups[k]
            row = [lab]
            for sn in snrs:
                row.append("%.4f" % np.mean([x["link"][sn]["reward"] for x in g]))
            row.append(f4(np.mean([x["link"][snrs[-1]]["rmse"] for x in g])))
            row.append("%.2f" % np.mean([x["link"][snrs[-1]]["fail_per_day"] for x in g]))
            rows.append(row)
            nm = {"CoVIR (proposed)": "covir", "Whittle index [Kadota]": "whittle", "Max-age (Greedy)": "maxage",
                  "VoI-threshold (decentralized)": "voit", "VoI-greedy": "voig", "QMIX": "qmix"}[k]
            mac(nm + "LinkLowReward", "%.4f" % np.mean([x["link"][snrs[-1]]["reward"] for x in g]))
            mac(nm + "LinkLowRMSE", f4(np.mean([x["link"][snrs[-1]]["rmse"] for x in g])))
            mac(nm + "LinkLowFail", "%.2f" % np.mean([x["link"][snrs[-1]]["fail_per_day"] for x in g]))
        hdr = ["Policy"] + [tex_minus("%g" % float(sn)) + " dB" for sn in snrs] + ["RMSE", "Failed"]
        write_table("link_margin", hdr, rows,
                    "Sensitivity to the rain-fade margin on the 2024 season without retraining: mean daily reward for clear-sky $E_s/N_0$ of %s~dB (threshold $%.2f$~dB), and RMSE and failed slot transmissions per day at the smallest margin." % (", ".join(tex_minus("%g" % float(x)) for x in snrs), -2.35),
                    "tab:link", "@{}l" + "r" * (len(snrs) + 2) + "@{}", star=False, size="\\scriptsize")
        mac("linkLowSnr", "%g" % float(snrs[-1]))
    t = e4["timing"]
    mac("scafInferMs", "%.1f" % t["scaf_day_ms"])
    mac("actorMsFive", "%.3f" % t["actor_ms_vs_agents"]["5"]); mac("actorMsBig", "%.3f" % t["actor_ms_vs_agents"]["160"])
    mac("whittleMsBig", "%.2f" % t["whittle_ms_vs_sources"]["240"]); mac("whittleMsSmall", "%.3f" % t["whittle_ms_vs_sources"]["15"])


# =====================================================================================
# E6 RAN
# =====================================================================================
SCHED = ["RR", "PF", "EDF", "MaxWeight", "DRIP"]
SCOL = {"RR": S5, "PF": S4, "EDF": S3, "MaxWeight": S2, "DRIP": S1}
SMK = {"RR": "v", "PF": "^", "EDF": "D", "MaxWeight": "s", "DRIP": "o"}
SLAB = {"RR": "RR", "PF": "PF", "EDF": "EDF", "MaxWeight": "Max-Weight", "DRIP": "DRIP (proposed)"}


def do_ran():
    e6 = load("e6_ran.json")
    if e6 is None:
        return
    df = []
    for exp, key, sched, seed, o in e6["runs"]:
        row = {"exp": exp, "key": key, "sched": sched, "seed": seed, "energy": o["energy_uj_per_useful_bit"], "sim_s": o["sim_s"]}
        for c in ["urgent", "actuation", "telemetry", "attack"]:
            for m in ["miss_ratio", "lat_mean_ms", "lat_p95_ms", "served_kbit", "arrivals"]:
                row["%s_%s" % (c, m)] = o[c][m]
        df.append(row)
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(CSV, "ran_all_runs.csv"), index=False)
    b = df[df.exp == "F1"]
    rows = []
    for s in SCHED:
        x = b[b.sched == s]
        rows.append([SLAB[s].replace("DRIP (proposed)", "\\textbf{DRIP (proposed)}"), len(x), "%.4f" % x.urgent_miss_ratio.mean(),
                     "%.4f" % x.actuation_miss_ratio.mean(),
                     "%.2f" % x.urgent_lat_mean_ms.mean(), "%.2f" % x.actuation_lat_p95_ms.mean(),
                     "%.1f" % x.telemetry_lat_mean_ms.mean(), "%.3f" % x.energy.mean()])
    f3 = df[(df.exp == "F3") & (df.key == "0.10")]
    rows.append("MIDRULE")
    for s in SCHED:
        x = f3[f3.sched == s]
        rows.append([SLAB[s].replace("DRIP (proposed)", "\\textbf{DRIP (proposed)}") + ", 10\\% flood", len(x), "%.4f" % x.urgent_miss_ratio.mean(),
                     "%.4f" % x.actuation_miss_ratio.mean(),
                     "%.2f" % x.urgent_lat_mean_ms.mean(), "%.2f" % x.actuation_lat_p95_ms.mean(),
                     "%.1f" % x.telemetry_lat_mean_ms.mean(), "%.3f" % x.energy.mean()])
    write_table("ran_main", ["Scheduler", "Runs", "Urgent miss", "Routine miss", "Urgent lat.", "Routine p95", "Telem. lat.", "Energy"],
                rows, "Agricultural RAN slice (11 RBs, 150 sensors, 50 valves, 3.5 GHz RMa) in an irrigation-advisory window: deadline-miss ratios, mean urgent latency and 95th-percentile routine latency (ms), mean telemetry latency (ms), and transmit energy ($\\mu$J per useful bit). Upper block: no attack; lower block: 10\\% of the sensors compromised.",
                "tab:ran", "@{}lrrrrrrr@{}", star=True)
    m = lambda d, s, c: d[d.sched == s][c].mean()
    for frac, tag in [("0.10", "Flood"), ("0.20", "FloodTwenty")]:
        f = df[(df.exp == "F3") & (df.key == frac)]
        mac("dripUrgentMiss" + tag, "%.3f" % m(f, "DRIP", "urgent_miss_ratio"))
        mac("mwUrgentMiss" + tag, "%.3f" % m(f, "MaxWeight", "urgent_miss_ratio"))
        mac("dripRoutineMiss" + tag, "%.3f" % m(f, "DRIP", "actuation_miss_ratio"))
        mac("mwRoutineMiss" + tag, "%.3f" % m(f, "MaxWeight", "actuation_miss_ratio"))
        mac("rrUrgentMiss" + tag, "%.3f" % m(f, "RR", "urgent_miss_ratio"))
        mac("pfUrgentMiss" + tag, "%.3f" % m(f, "PF", "urgent_miss_ratio"))
        mac("edfUrgentMiss" + tag, "%.3f" % m(f, "EDF", "urgent_miss_ratio"))
        mac("dripTelLat" + tag, "%.1f" % m(f, "DRIP", "telemetry_lat_mean_ms"))
        mac("edfTelLat" + tag, "%.1f" % m(f, "EDF", "telemetry_lat_mean_ms"))
    for s in SCHED:
        mac(s.lower().replace("maxweight", "mw") + "BaseUrgentLat", "%.2f" % m(b, s, "urgent_lat_mean_ms"))
        mac(s.lower().replace("maxweight", "mw") + "BaseTelLat", "%.1f" % m(b, s, "telemetry_lat_mean_ms"))
        mac(s.lower().replace("maxweight", "mw") + "BaseRoutineMiss", "%.4f" % m(b, s, "actuation_miss_ratio"))
        mac(s.lower().replace("maxweight", "mw") + "BaseEnergy", "%.3f" % m(b, s, "energy"))
    slots = 1000.0 / C.EnvConfig().__class__.__dict__.get("slot_ms", 0.5) if False else 2000.0
    mac("dripSlotMs", "%.3f" % (b[b.sched == "DRIP"].sim_s.mean() / slots * 1000))
    # flood figure
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.0))
    f3 = df[df.exp == "F3"]
    fr = sorted(f3.key.unique(), key=float)
    for s in SCHED:
        ya = [f3[(f3.key == k) & (f3.sched == s)][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean() for k in fr]
        yt = [f3[(f3.key == k) & (f3.sched == s)].telemetry_lat_mean_ms.mean() for k in fr]
        x = [100 * float(k) for k in fr]
        axs[0].plot(x, ya, color=SCOL[s], marker=SMK[s], label=SLAB[s])
        axs[1].plot(x, yt, color=SCOL[s], marker=SMK[s], label=SLAB[s])
    axs[0].set_xlabel("compromised sensors (%)"); axs[0].set_ylabel("command deadline-miss ratio")
    axs[1].set_xlabel("compromised sensors (%)"); axs[1].set_ylabel("telemetry latency (ms)")
    axs[0].legend(loc="upper left", fontsize=6.3)
    for a, t in zip(axs, ["(a)", "(b)"]):
        a.set_title(t, loc="left")
    fig.tight_layout()
    save_fig(fig, "fig_ran_flood.pdf")
    # scaling figure
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.0))
    f2 = df[df.exp == "F2"]
    ks = sorted(f2.key.unique(), key=int)
    for s in SCHED:
        ya = [f2[(f2.key == k) & (f2.sched == s)].actuation_lat_p95_ms.mean() for k in ks]
        yt = [f2[(f2.key == k) & (f2.sched == s)][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean() for k in ks]
        x = [int(k) for k in ks]
        axs[0].plot(x, ya, color=SCOL[s], marker=SMK[s], label=SLAB[s])
        axs[1].plot(x, yt, color=SCOL[s], marker=SMK[s], label=SLAB[s])
    for a in axs:
        a.set_xscale("log"); a.set_xlabel("devices per cell")
    axs[0].set_ylabel("routine command p95 latency (ms)"); axs[1].set_ylabel("command deadline-miss ratio")
    axs[0].legend(loc="upper left", fontsize=6.3)
    for a, t in zip(axs, ["(a)", "(b)"]):
        a.set_title(t, loc="left")
    fig.tight_layout()
    save_fig(fig, "fig_ran_scale.pdf")
    big = f2[f2.key == ks[-1]]
    mac("scaleDevicesMax", ks[-1])
    mac("dripScaleMiss", "%.3f" % big[big.sched == "DRIP"][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean())
    mac("edfScaleMiss", "%.3f" % big[big.sched == "EDF"][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean())
    mac("pfScaleMiss", "%.3f" % big[big.sched == "PF"][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean())
    mac("rrScaleMiss", "%.3f" % big[big.sched == "RR"][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean())
    mac("mwScaleMiss", "%.3f" % big[big.sched == "MaxWeight"][["urgent_miss_ratio", "actuation_miss_ratio"]].mean().mean())
    # FR1 vs FR3
    f4_ = df[df.exp == "F4"]
    rows = []
    for key in ["FR1 3.5 GHz", "FR3 7 GHz"]:
        for s in SCHED:
            x = f4_[(f4_.key == key) & (f4_.sched == s)]
            rows.append([key.replace("FR1 3.5 GHz", "3.5 GHz").replace("FR3 7 GHz", "7 GHz"), SLAB[s], len(x),
                         "%.4f" % x.urgent_miss_ratio.mean(), "%.4f" % x.actuation_miss_ratio.mean(),
                         "%.2f" % x.actuation_lat_p95_ms.mean(), "%.1f" % x.telemetry_lat_mean_ms.mean()])
        if key.startswith("FR1"):
            rows.append("MIDRULE")
    write_table("ran_band", ["Carrier", "Scheduler", "Runs", "Urg. miss", "Rout. miss", "Rout. p95", "Tel. lat."], rows,
                "FR1 (3.5 GHz, 30 kHz SCS, 11 RBs) versus upper mid-band (7 GHz, 60 kHz SCS, 27 RBs) slices under the RMa model: deadline-miss ratios of urgent and routine commands, 95th-percentile routine latency and mean telemetry latency (ms).",
                "tab:band", "@{}llrrrrr@{}", size="\\scriptsize")
    for key, tag in [("FR1 3.5 GHz", "One"), ("FR3 7 GHz", "Three")]:
        x = f4_[(f4_.key == key) & (f4_.sched == "DRIP")]
        mac("bandTelLat" + tag, "%.1f" % x.telemetry_lat_mean_ms.mean())
        mac("bandUrgentMiss" + tag, "%.4f" % x.urgent_miss_ratio.mean())
        mac("bandRoutineMiss" + tag, "%.4f" % x.actuation_miss_ratio.mean())
        mac("bandRoutineP" + tag, "%.2f" % x.actuation_lat_p95_ms.mean())
    mac("ranRunsBase", int(b.groupby("sched").size().min()))


# =====================================================================================
# data timeline figure and test trace
# =====================================================================================
def do_timeline(M=None):
    from terai import data as D
    P = D.build_panel()
    days = P.days
    fig, axs = plt.subplots(3, 1, figsize=(7.1, 3.3), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.1, 1.3]})
    rain = np.nanmean(P.rain_mm, 1)
    axs[0].bar(days, rain, width=1.0, color=S1, lw=0)
    axs[0].set_ylabel("rain (mm/day)\n5-district mean")
    axs[1].plot(days, P.sm_target, ".", ms=1.8, color=S2)
    axs[1].set_ylabel("SMAP soil moisture\n(m$^3$m$^{-3}$)")
    for j, d in enumerate(C.DISTRICTS):
        a = np.flatnonzero(P.avail["S1"][:, j])
        axs[2].plot(days[a], np.full(len(a), j), "|", ms=5, color=S3, mew=0.8)
    a = np.flatnonzero(np.isfinite(P.sm_target))
    axs[2].plot(days[a], np.full(len(a), 5), "|", ms=5, color=S2, mew=0.6)
    axs[2].set_yticks(range(6)); axs[2].set_yticklabels(["S1 " + d for d in C.DISTRICTS] + ["SMAP"], fontsize=6.3)
    axs[2].grid(False)
    for a in axs:
        a.axvspan(pd.Timestamp(C.TEST_START), pd.Timestamp(C.TEST_END), color="#f0efec", zorder=0, lw=0)
    axs[0].text(pd.Timestamp("2023-03-01"), axs[0].get_ylim()[1] * 0.85, "training period", fontsize=7, color=INK2)
    axs[0].text(pd.Timestamp("2024-04-15"), axs[0].get_ylim()[1] * 0.85, "test period (2024)", fontsize=7, color=INK2)
    fig.tight_layout()
    save_fig(fig, "fig_data.pdf")
    if M is not None and "CoVIR (proposed)" in M and "Whittle index [Kadota]" in M:
        t0 = P.day_index(C.TEST_START)
        tgt = P.sm_target[t0:t0 + 305]
        cv = np.array(M["CoVIR (proposed)"][0]["est_trace_copy0"]); wh = np.array(M["Whittle index [Kadota]"][0]["est_trace_copy0"])
        nv = np.array(M["Never refresh"][0]["est_trace_copy0"]) if "Never refresh" in M else None
        x = days[t0:t0 + len(cv)]
        fig, ax = plt.subplots(figsize=(7.1, 1.9))
        ax.plot(x, tgt[:len(cv)], ".", color=INK2, ms=2.5, label="SMAP target")
        ax.plot(x, cv, color=S1, lw=1.1, label="CoVIR")
        ax.plot(x, wh, color=S2, lw=1.0, label="Whittle [Kadota]")
        if nv is not None:
            ax.plot(x, nv, color=GRAY, lw=0.9, label="never refresh")
        ax.set_ylabel("soil moisture (m$^3$m$^{-3}$)")
        ax.legend(loc="upper left", ncol=4)
        save_fig(fig, "fig_trace.pdf")


def do_link_stats():
    """Fraction of hill-site days on which the rain fade lowers the LEO slot success below 0.99."""
    from terai import data as D, forecast as F, env as E
    P = D.build_panel(); S = F.Sampler(P)
    out = {}
    for snr in [12.0, -1.0]:
        cfg = E.SlowConfig(); cfg.ntn_clear_snr_db = snr
        env = E.SlowEnv(P, S, [lambda t, h: np.zeros(len(t))], cfg, B=8, mode="test", seed=1000)
        env.use_voi = False
        env.reset(); ps = []
        while not env.done.all():
            ps.append(env.p_succ[:, 2:].copy()); env.step(np.zeros((8, 5), int))
        ps = np.array(ps)
        out[snr] = ps
    mac("leoFadeDaysPct", 100 * np.mean(out[12.0] < 0.99), "{:.2f}")
    mac("leoMeanP", "%.4f" % np.mean(out[12.0]))
    mac("leoMeanPLow", "%.2f" % np.mean(out[-1.0])); mac("leoMinPLow", "%.2f" % np.min(out[-1.0]))


def do_coverage():
    """Fraction of devices whose mean SNR is below the -10 dB decoding floor (RMa, 3 km)."""
    from terai import ran as R
    out = {}
    for tag, kw in [("One", {}), ("Three", {"fc_ghz": 7.0, "n_rb": 27, "scs_khz": 60.0, "slot_ms": 0.25,
                                            "p_gnb_dbm_per_rb": 46.0 + 10 * np.log10(100 / 20) - 10 * np.log10(135)})]:
        dl, ul = [], []
        for s in range(20):
            cfg = R.RanConfig(**kw)
            cell = R.Cell(cfg, np.random.default_rng(s), None, None, None)
            dl.append(np.mean(cell.snr_dl[cell.is_valve] < -10)); ul.append(np.mean(cell.snr_ul[~cell.is_valve] < -10))
        mac("valveOutagePct" + tag, 100 * np.mean(dl), "{:.1f}")
        mac("sensorOutagePct" + tag, 100 * np.mean(ul), "{:.1f}")


def write_macros():
    import torch, sklearn
    mac("torchVersion", torch.__version__.split("+")[0]); mac("sklearnVersion", sklearn.__version__)
    tot = 0.0
    for f in ["e1_forecast.json", "e3_slow.json", "e4_policy.json", "e6_ran.json"]:
        x = load(f)
        if x:
            tot += x.get("meta", {}).get("runtime_s", 0) + x.get("runtime_s", 0)
    e0 = 0
    mac("totalRuntimeHours", "%.1f" % (tot / 3600 + 0.25))
    lines = ["%% generated by scripts/make_outputs.py -- do not edit"]
    for k, v in sorted(MACROS.items()):
        lines.append("\\newcommand{\\%s}{%s}" % (k, v))
    open(os.path.join(MAN, "results_macros.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(MACROS, open(os.path.join(ROOT, "results", "macros.json"), "w"), indent=1)


def main():
    do_data(); do_params(); do_forecast(); do_voi()
    M = do_slow(); do_policy(); do_ran(); do_coverage(); do_link_stats(); do_timeline(M)
    write_macros()
    print("macros:", len(MACROS))


if __name__ == "__main__":
    main()
