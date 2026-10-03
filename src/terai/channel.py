"""Backhaul channel models.

* Terrestrial backhaul (plains sites): reliable link with a small random failure rate.
* LEO satellite (NTN) backhaul (hill sites): Ka-band link whose Es/N0 is degraded by
  rain attenuation computed with ITU-R P.838 (specific attenuation), P.839 (rain
  height) and the P.618 horizontal path-reduction factor, driven by the *real* IMD daily
  rainfall of the district.  Daily rain (mm/day) is converted to a rain rate by
  assuming it falls during ``wet_hours``; the decision window is rainy with
  probability wet_hours/24 (independently per day and seed).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import config as C


@lru_cache(maxsize=None)
def _site_geometry(district: str):
    from itur.models import itu839, itu1511
    lat, lon = C.DISTRICT_LATLON[district]
    hr = float(itu839.rain_height(lat, lon).value)          # km
    hs = float(itu1511.topographic_altitude(lat, lon).value)  # km
    return hr, hs


@lru_cache(maxsize=4096)
def _gamma(R_mmh: float, f_ghz: float, el_deg: float) -> float:
    from itur.models import itu838
    if R_mmh <= 0:
        return 0.0
    return float(itu838.rain_specific_attenuation(R_mmh, f_ghz, el_deg, 45.0).value)


def rain_attenuation_db(district: str, R_mmh: float, f_ghz: float, el_deg: float) -> float:
    """Slant-path rain attenuation (dB) for an instantaneous rain rate R (mm/h)."""
    if R_mmh <= 0:
        return 0.0
    hr, hs = _site_geometry(district)
    el = math.radians(el_deg)
    Ls = max(hr - hs, 0.0) / math.sin(el)                   # slant path below rain height (km)
    LG = Ls * math.cos(el)                                   # horizontal projection (km)
    g = _gamma(round(R_mmh, 2), f_ghz, el_deg)
    r = 1.0 / (1.0 + 0.78 * math.sqrt(LG * g / f_ghz) - 0.38 * (1.0 - math.exp(-2.0 * LG)))
    return g * Ls * max(r, 0.0)


class BackhaulDay:
    """Per-day backhaul state of every district (sampled once per day and seed)."""

    def __init__(self, cfg: C.EnvConfig, rain_mm_today: np.ndarray, rng: np.random.Generator):
        n = len(C.DISTRICTS)
        self.rainy = np.zeros(n, bool)
        self.rate_mmh = np.zeros(n)
        self.atten_db = np.zeros(n)
        self.snr_db = np.full(n, np.inf)
        self.p_success = np.zeros(n)
        self.rate_factor = np.ones(n)            # fraction of clear-sky spectral efficiency
        for j, d in enumerate(C.DISTRICTS):
            if C.BACKHAUL[d] == "terrestrial":
                self.p_success[j] = 0.995
                continue
            if rain_mm_today[j] > 0 and rng.random() < cfg.wet_hours / 24.0:
                self.rainy[j] = True
                self.rate_mmh[j] = rain_mm_today[j] / cfg.wet_hours
            self.atten_db[j] = rain_attenuation_db(d, self.rate_mmh[j], cfg.ntn_freq_ghz, cfg.ntn_elev_deg)
            self.snr_db[j] = cfg.ntn_clear_snr_db - self.atten_db[j]
            # per-slot scintillation/fading margin: N(0, 1.5 dB) around the mean Es/N0
            z = (self.snr_db[j] - cfg.ntn_outage_snr_db) / 1.5
            self.p_success[j] = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            se_clear = math.log2(1 + 10 ** (cfg.ntn_clear_snr_db / 10))
            se = math.log2(1 + 10 ** (max(self.snr_db[j], -10) / 10))
            self.rate_factor[j] = se / se_clear


def expected_p_success(cfg: C.EnvConfig, district: str, rain_mm: float) -> float:
    """Success probability an agent expects given today's local rain (used by baselines
    that need p_i, e.g. the Whittle index, and as an observation feature)."""
    if C.BACKHAUL[district] == "terrestrial":
        return 0.995
    q = cfg.wet_hours / 24.0 if rain_mm > 0 else 0.0
    a = rain_attenuation_db(district, rain_mm / cfg.wet_hours if rain_mm > 0 else 0.0,
                            cfg.ntn_freq_ghz, cfg.ntn_elev_deg)
    pr = 0.5 * (1 + math.erf(((cfg.ntn_clear_snr_db - a) - cfg.ntn_outage_snr_db) / 1.5 / math.sqrt(2)))
    pc = 0.5 * (1 + math.erf((cfg.ntn_clear_snr_db - cfg.ntn_outage_snr_db) / 1.5 / math.sqrt(2)))
    return q * pr + (1 - q) * pc
