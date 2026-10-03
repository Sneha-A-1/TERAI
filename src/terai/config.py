"""Global configuration for the TERAI experiments.

Every numerical constant used by the code lives here (or in configs/*.yaml) so that
the manuscript tables can be generated from a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
RES_RAW = os.path.join(RESULTS, "raw")
RES_TAB = os.path.join(RESULTS, "tables")
RES_FIG = os.path.join(RESULTS, "figures")

# ----------------------------------------------------------------------------------
# Study region and periods
# ----------------------------------------------------------------------------------
DISTRICTS = ["Dehradun", "Haridwar", "Nainital", "Almora", "Pithoragarh"]
# Backhaul type of each district edge site (modelling assumption, see Sec. III):
# plains/valley sites reach the regional cloud over terrestrial (fibre/microwave)
# backhaul, hill sites over a shared LEO satellite (NTN) beam.
BACKHAUL = {"Dehradun": "terrestrial", "Haridwar": "terrestrial",
            "Nainital": "ntn", "Almora": "ntn", "Pithoragarh": "ntn"}
# District reference points (used for the satellite elevation/rain-height model).
DISTRICT_LATLON = {"Dehradun": (30.3165, 78.0322), "Haridwar": (29.9457, 78.1642),
                   "Nainital": (29.3803, 79.4636), "Almora": (29.5971, 79.6591),
                   "Pithoragarh": (29.5829, 80.2182)}

PANEL_START = "2022-10-01"
PANEL_END = "2024-10-31"
TRAIN_START = "2022-10-15"   # first decision day (allows a 14-day warm-up)
TRAIN_END = "2023-12-31"     # last decision day of the training period
TEST_START = "2024-01-01"
TEST_END = "2024-10-31"      # last decision day of the test period
HORIZON = 0                  # target = regional soil-moisture state on the decision day

# ----------------------------------------------------------------------------------
# Modalities
# ----------------------------------------------------------------------------------
MODALITIES = ["IMD", "POWER", "S1", "SMAP"]
# Publication latency (days) between acquisition and availability at the data hub.
# These are modelling assumptions (documented in Table "simulation parameters").
LATENCY = {"IMD": 1, "POWER": 2, "S1": 1, "SMAP": 2}
A_MAX = 30                     # ages are capped at this many days
RAIN_EVENT_MM = 10.0           # next-day rain-event threshold (mm/day)

# Payload (bytes) of one district-level refresh of each product (see synth.payloads)
# are computed in terai.synth from product geometry and are stored in results.

SEEDS = list(range(10))


@dataclass
class EnvConfig:
    """Slow-timescale (daily) environment parameters."""
    window_s: float = 3600.0          # processing window before the advisory deadline (s)
    terr_capacity_mbps: float = 100.0 # terrestrial pool shared by plains sites (Mbit/s)
    ntn_capacity_mbps: float = 50.0   # LEO beam capacity shared by hill sites (Mbit/s, clear sky)
    ntn_clear_snr_db: float = 12.0    # clear-sky Es/N0 of the NTN user terminal (dB)
    ntn_outage_snr_db: float = -2.35  # lowest decodable Es/N0 (QPSK 1/4 DVB-S2 class)
    ntn_freq_ghz: float = 20.0        # Ka-band downlink
    ntn_elev_deg: float = 40.0        # mean LEO elevation angle
    wet_hours: float = 4.0            # effective rain duration used to convert mm/day to mm/h
    ntn_daily_cap_mb: float = 400.0   # metered NTN volume budget per day (MB, all hill sites)
    edge_cpu_speed: float = 1.0       # relative speed of an edge server (cloud = 4x)
    cloud_cpu_speed: float = 4.0
    cloud_slots: int = 3              # parallel cloud workers available to the region
    deadline_violation_target: float = 0.02
    episode_days: int = 60
    w_regional: float = 1.0           # weight of the regional soil-moisture loss
    w_energy: float = 0.05            # weight of normalised energy in the reward
    lambda_init: float = 1.0
    lambda_lr: float = 0.05


@dataclass
class EnergyConfig:
    """Energy and carbon parameters (documented assumptions)."""
    terr_j_per_bit: float = 20e-9     # terrestrial backhaul transport energy (J/bit)
    ntn_terminal_w: float = 60.0      # satellite user-terminal power while transmitting (W)
    edge_server_w: float = 150.0      # edge server power when busy (W)
    cloud_server_w: float = 350.0     # cloud server power when busy (W)
    cloud_pue: float = 1.5            # data-centre power usage effectiveness
    grid_kgco2_per_kwh: float = 0.716 # Indian grid emission factor (CEA CO2 baseline database)
