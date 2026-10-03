"""Staleness-Conditioned Attention Fusion (SCAF) forecaster and baselines.

Two variants share one architecture:
  * SCAF-cloud ("full"): attends over the tokens of *all* districts plus the regional
    SMAP token; predicts every district's next-day rain-event probability and the
    regional next-day SMAP soil moisture.
  * SCAF-edge ("lite"): attends only over the district's own tokens; predicts the
    district's rain-event probability.

A token for modality m of district j is

    tok = g(a) * E_m(x) + (1 - g(a)) * mu_m + P_m(a) + e_j + u_m,
    g(a) = exp(-softplus(beta_m) * a / 7)          (per-dimension staleness gate)

where x is the (standardised) feature vector of the product held by the agent, a its
age in days, E_m a two-layer encoder, mu_m a learned prior, P_m an age embedding,
e_j a district embedding and u_m a modality embedding.  Missing products are masked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import Panel, latest_available

MOD_LOCAL = ["IMD", "POWER", "S1"]       # district-level modalities
N_DIST = len(C.DISTRICTS)


# --------------------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------------------
class Sampler:
    """Builds model inputs for decision days given the acquisition each agent holds."""

    def __init__(self, panel: Panel):
        self.P = panel
        self.latest = {m: latest_available(panel, m) for m in C.MODALITIES}
        # list of acquisition days per (m, j) for random staleness sampling
        self.acq = {m: [np.flatnonzero(panel.avail[m][:, j]) for j in range(N_DIST)]
                    for m in C.MODALITIES}
        doy = panel.days.dayofyear.values
        self.cal = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], -1)
        # monthly rain-event climatology per district, estimated on training targets only
        tr = self.day_range(C.TRAIN_START, C.TRAIN_END)
        months = panel.days[tr + C.HORIZON].month.values
        y = panel.rain_event[tr + C.HORIZON]
        clim = np.zeros((13, N_DIST))
        for mo in range(1, 13):
            sel = months == mo
            # Laplace-smoothed frequency (keeps logits finite)
            clim[mo] = (y[sel].sum(0) + 0.5) / (sel.sum() + 1.0) if sel.any() else y.mean(0)
        clim[0] = y.mean(0)
        self.clim = clim
        self.prior_logit_table = np.log(clim / (1 - clim)).astype(np.float32)
        sm_tr = panel.sm_target[tr + C.HORIZON]
        self.sm_mu = float(np.nanmean(sm_tr))
        self.sm_sd = float(np.nanstd(sm_tr))

    def day_range(self, start, end):
        return np.arange(self.P.day_index(start), self.P.day_index(end) + 1)

    def fresh_held(self, t_idx):
        """Held acquisition = newest available (what an always-refreshing agent holds)."""
        return {m: self.latest[m][t_idx].copy() for m in C.MODALITIES}

    def random_held(self, t_idx, rng, p_fresh=0.4, p_missing=0.05):
        """Random staleness patterns covering the ages a scheduling policy can produce."""
        held = {}
        for m in C.MODALITIES:
            L = self.latest[m][t_idx]
            h = L.copy()
            for b, t in enumerate(t_idx):
                for j in range(N_DIST):
                    if L[b, j] < 0:
                        continue
                    u = rng.random()
                    if u < p_fresh:
                        continue
                    if u < p_fresh + p_missing:
                        h[b, j] = -1
                        continue
                    acq = self.acq[m][j]
                    lo = t - C.A_MAX
                    cand = acq[(acq >= lo) & (acq <= L[b, j])]
                    h[b, j] = int(rng.choice(cand)) if len(cand) else -1
            if m == "SMAP":   # one regional product: same holding for every district
                h[:] = h[:, :1]
            held[m] = h
        return held

    def inputs(self, t_idx, held):
        """Return dict of arrays: x[m] (B,N,f), age[m] (B,N), mask[m] (B,N), cal (B,2)."""
        B = len(t_idx)
        tgt = np.minimum(t_idx + C.HORIZON, self.P.n_days - 1)
        out = {"x": {}, "age": {}, "mask": {}, "cal": self.cal[tgt],
               "prior_logit": self.prior_logit_table[self.P.days[tgt].month.values]}
        for m in C.MODALITIES:
            h = held[m]
            mu, sd = self.P.norm[m]
            f = self.P.feat[m]
            x = np.zeros((B, N_DIST, f.shape[-1]), np.float32)
            age = np.full((B, N_DIST), float(C.A_MAX), np.float32)
            mask = np.zeros((B, N_DIST), bool)
            valid = h >= 0
            bi, ji = np.nonzero(valid)
            hv = h[bi, ji]
            x[bi, ji] = (f[hv, ji] - mu) / sd
            a = (t_idx[bi] - hv).astype(np.float32)
            ok = a <= C.A_MAX
            age[bi[ok], ji[ok]] = a[ok]
            mask[bi[ok], ji[ok]] = True
            x[~mask] = 0.0
            out["x"][m], out["age"][m], out["mask"][m] = x, age, mask
        # persistence base for regional soil moisture: SM of the held SMAP product
        hs = held["SMAP"][:, 0]
        base = np.full(B, self.sm_mu, np.float32)
        okb = out["mask"]["SMAP"][:, 0]
        base[okb] = self.P.feat["SMAP"][hs[okb], 0, 0]
        out["sm_base"] = base
        return out

    def targets(self, t_idx):
        y_rain = self.P.rain_event[t_idx + C.HORIZON]             # (B, N)
        sm = self.P.sm_target[t_idx + C.HORIZON]                  # (B,)
        return y_rain, sm


# --------------------------------------------------------------------------------------
# SCAF model (PyTorch)
# --------------------------------------------------------------------------------------
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_scaf(variant: str, d: int = 32, seed: int = 0, mods=None):
    torch, nn = _torch()
    torch.manual_seed(seed)
    fdim = {m: len(v) for m, v in __import__("terai.data", fromlist=["F_NAMES"]).F_NAMES.items()}
    if mods is None:
        mods = C.MODALITIES if variant == "cloud" else MOD_LOCAL

    class SCAF(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant, self.mods, self.d = variant, mods, d
            self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(fdim[m], d), nn.GELU(), nn.Linear(d, d))
                                      for m in mods})
            self.beta = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in mods})
            self.prior = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in mods})
            self.age_emb = nn.ModuleDict({m: nn.Linear(2, d) for m in mods})
            self.mod_emb = nn.ParameterDict({m: nn.Parameter(0.02 * torch.randn(d)) for m in mods})
            self.dist_emb = nn.Parameter(0.02 * torch.randn(N_DIST, d))
            self.cal = nn.Linear(2, d)
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.o = nn.Linear(d, d)
            self.ln = nn.LayerNorm(d)
            self.head_rain = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            if variant == "cloud":
                self.q_reg = nn.Parameter(0.02 * torch.randn(d))
                self.head_sm = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.use_gate = True
            self.sm_mu, self.sm_sd = 0.0, 1.0
            # residual heads start at the climatology / persistence priors
            nn.init.zeros_(self.head_rain[2].weight); nn.init.zeros_(self.head_rain[2].bias)
            if variant == "cloud":
                nn.init.zeros_(self.head_sm[2].weight); nn.init.zeros_(self.head_sm[2].bias)
            self.drop = nn.Dropout(0.1)

        def tokens(self, batch):
            """Return tokens (B, T, d), mask (B, T) bool, district of token (T,)"""
            toks, masks, owner = [], [], []
            for m in self.mods:
                x = batch["x"][m]                       # (B,N,f)
                a = batch["age"][m]                     # (B,N)
                e = self.enc[m](x)
                aa = torch.stack([torch.log1p(a) / math.log(1 + C.A_MAX), a / C.A_MAX], -1)
                if self.use_gate:
                    g = torch.exp(-torch.nn.functional.softplus(self.beta[m]) * a.unsqueeze(-1) / 7.0)
                    e = g * e + (1 - g) * self.prior[m]
                t = e + self.age_emb[m](aa) + self.mod_emb[m] + self.dist_emb.unsqueeze(0)
                if m == "SMAP":                          # regional token: keep one copy
                    toks.append(t[:, :1]); masks.append(batch["mask"][m][:, :1]); owner.append([-1])
                else:
                    toks.append(t); masks.append(batch["mask"][m]); owner.append(list(range(N_DIST)))
            T = torch.cat(toks, 1)
            M = torch.cat(masks, 1)
            own = [o for ol in owner for o in ol]
            return T, M, torch.tensor(own)

        def attend(self, qv, T, M):
            # qv (B,Q,d); T (B,K,d); M (B,Q,K) bool
            q = self.q(qv); k = self.k(T); v = self.v(T)
            s = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.d)
            s = s.masked_fill(~M, -1e9)
            w = torch.softmax(s, -1)
            return self.ln(qv + self.o(torch.einsum("bqk,bkd->bqd", w, v))), w

        def forward(self, batch):
            T, M, own = self.tokens(batch)
            B = T.shape[0]
            c = self.cal(batch["cal"])                                # (B,d)
            ctx = c.unsqueeze(1)                                      # context token
            T = torch.cat([T, ctx], 1)
            M = torch.cat([M, torch.ones(B, 1, dtype=torch.bool)], 1)
            own = torch.cat([own, torch.tensor([-2])])
            qd = self.dist_emb.unsqueeze(0) + c.unsqueeze(1)          # (B,N,d)
            if self.variant == "cloud":
                Mq = M.unsqueeze(1).expand(B, N_DIST, -1)
            else:
                allow = (own.unsqueeze(0) == torch.arange(N_DIST).unsqueeze(1)) | (own == -2).unsqueeze(0)
                Mq = M.unsqueeze(1) & allow.unsqueeze(0)
            z, w = self.attend(qd, T, Mq)
            out = {"rain_logit": batch["prior_logit"] + self.head_rain(z).squeeze(-1), "attn": w}
            if self.variant == "cloud":
                qr = (self.q_reg + c).unsqueeze(1)
                zr, wr = self.attend(qr, T, M.unsqueeze(1))
                # residual w.r.t. persistence of the held SMAP product (standardised units)
                out["sm"] = (batch["sm_base"] - self.sm_mu) / self.sm_sd + self.head_sm(zr).squeeze(-1).squeeze(-1)
                out["attn_reg"] = wr
            return out

    return SCAF()


def to_torch(inp):
    torch, _ = _torch()
    return {"x": {m: torch.tensor(v) for m, v in inp["x"].items()},
            "age": {m: torch.tensor(v) for m, v in inp["age"].items()},
            "mask": {m: torch.tensor(v) for m, v in inp["mask"].items()},
            "cal": torch.tensor(inp["cal"], dtype=torch.float32),
            "prior_logit": torch.tensor(inp["prior_logit"], dtype=torch.float32),
            "sm_base": torch.tensor(inp["sm_base"], dtype=torch.float32)}


def _slice(inp, idx):
    return {"x": {m: v[idx] for m, v in inp["x"].items()},
            "age": {m: v[idx] for m, v in inp["age"].items()},
            "mask": {m: v[idx] for m, v in inp["mask"].items()},
            "cal": inp["cal"][idx], "prior_logit": inp["prior_logit"][idx], "sm_base": inp["sm_base"][idx]}


@dataclass
class TrainResult:
    model: object
    sm_mu: float
    sm_sd: float
    history: list


def split_train_val(sampler: "Sampler", seed: int = 2024, frac_val: float = 0.2):
    """Random day-level split of the training period (validation used for early stopping
    and hyper-parameter selection only; the 2024 test period is never used)."""
    t_idx = sampler.day_range(C.TRAIN_START, C.TRAIN_END)
    rng = np.random.default_rng(seed)
    # split by 7-day blocks to limit autocorrelation leakage
    blocks = (t_idx - t_idx[0]) // 7
    ub = np.unique(blocks)
    vb = rng.choice(ub, size=int(round(frac_val * len(ub))), replace=False)
    is_val = np.isin(blocks, vb)
    return t_idx[~is_val], t_idx[is_val]


def _loss_np(sampler, tr_res, inp, yr, sm, w_rain=0.0):
    pr = predict_scaf(tr_res, inp)
    p = np.clip(pr["p_rain"], 1e-6, 1 - 1e-6)
    bce = float(-(yr * np.log(p) + (1 - yr) * np.log(1 - p)).mean())
    out = w_rain * bce
    if "sm" in pr:
        ok = np.isfinite(sm)
        out += float((((pr["sm"][ok] - sm[ok]) / sampler.sm_sd) ** 2).mean())
    return out


def train_scaf(sampler: Sampler, variant: str, seed: int, epochs: int = 250, n_aug: int = 8,
               lr: float = 3e-3, wd: float = 1e-3, d: int = 32, use_gate: bool = True,
               train_days=None, val_days=None, patience: int = 50, w_rain: float = 0.0,
               mods=None, verbose=False) -> TrainResult:
    import copy
    torch, nn = _torch()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    if train_days is None:
        train_days, val_days = split_train_val(sampler)
    t_idx = train_days
    model = build_scaf(variant, d=d, seed=seed, mods=mods)
    model.use_gate = use_gate
    sm_mu, sm_sd = sampler.sm_mu, sampler.sm_sd
    model.sm_mu, model.sm_sd = sm_mu, sm_sd
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    hist = []

    def draw(days, k, r):
        tt = np.concatenate([days] * k)
        held = sampler.random_held(tt, r)
        fresh = sampler.fresh_held(tt)
        sel = r.random(len(tt)) < 0.3
        for m in held:
            held[m][sel] = fresh[m][sel]
        inp = sampler.inputs(tt, held)
        yr, sm = sampler.targets(tt)
        return inp, yr, sm

    inp, yr, sm = draw(t_idx, n_aug, rng)
    vinp = vy = vsm = None
    if val_days is not None and len(val_days):
        vinp, vy, vsm = draw(val_days, 4, np.random.default_rng(10_000 + seed))
    best, best_state, bad = np.inf, None, 0
    tmp = TrainResult(model, sm_mu, sm_sd, hist)
    for ep in range(epochs):
        model.train()
        if ep > 0 and ep % 20 == 0:
            inp, yr, sm = draw(t_idx, n_aug, rng)
        perm = rng.permutation(len(yr))
        tot = 0.0
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            batch = to_torch(_slice(inp, b))
            out = model(batch)
            y = torch.tensor(yr[b])
            loss = w_rain * nn.functional.binary_cross_entropy_with_logits(out["rain_logit"], y)
            if variant == "cloud":
                st = torch.tensor(sm[b])
                ok = torch.isfinite(st)
                if ok.any():
                    loss = loss + nn.functional.mse_loss(out["sm"][ok], (st[ok] - sm_mu) / sm_sd)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        vl = _loss_np(sampler, tmp, vinp, vy, vsm, w_rain) if vinp is not None else tot / len(perm)
        hist.append((tot / len(perm), vl))
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(variant, ep, hist[-1])
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    res = TrainResult(model, sm_mu, sm_sd, hist)
    res.best_val = best
    res.epochs_run = len(hist)
    return res


def predict_scaf(tr: TrainResult, inp):
    torch, _ = _torch()
    with torch.no_grad():
        out = tr.model(to_torch(inp))
    p = torch.sigmoid(out["rain_logit"]).numpy()
    res = {"p_rain": p}
    if "sm" in out:
        res["sm"] = out["sm"].numpy() * tr.sm_sd + tr.sm_mu
    return res


# --------------------------------------------------------------------------------------
# NumPy inference (used inside the RL environment; verified against PyTorch)
# --------------------------------------------------------------------------------------
class NumpySCAF:
    def __init__(self, tr: TrainResult):
        m = tr.model
        self.variant, self.mods, self.d = m.variant, m.mods, m.d
        self.use_gate = m.use_gate
        g = lambda t: t.detach().numpy().astype(np.float64)
        self.enc = {k: (g(s[0].weight), g(s[0].bias), g(s[2].weight), g(s[2].bias)) for k, s in m.enc.items()}
        self.beta = {k: np.log1p(np.exp(g(v))) for k, v in m.beta.items()}
        self.prior = {k: g(v) for k, v in m.prior.items()}
        self.age_emb = {k: (g(v.weight), g(v.bias)) for k, v in m.age_emb.items()}
        self.mod_emb = {k: g(v) for k, v in m.mod_emb.items()}
        self.dist = g(m.dist_emb)
        self.cal = (g(m.cal.weight), g(m.cal.bias))
        self.W = {n: (g(getattr(m, n).weight), g(getattr(m, n).bias)) for n in ["q", "k", "v", "o"]}
        self.ln = (g(m.ln.weight), g(m.ln.bias))
        self.hr = (g(m.head_rain[0].weight), g(m.head_rain[0].bias), g(m.head_rain[2].weight), g(m.head_rain[2].bias))
        if self.variant == "cloud":
            self.q_reg = g(m.q_reg)
            self.hs = (g(m.head_sm[0].weight), g(m.head_sm[0].bias), g(m.head_sm[2].weight), g(m.head_sm[2].bias))
        self.sm_mu, self.sm_sd = tr.sm_mu, tr.sm_sd

    @staticmethod
    def _gelu(x):
        from scipy.special import erf
        return 0.5 * x * (1 + erf(x / math.sqrt(2)))

    @staticmethod
    def _lin(x, wb):
        return x @ wb[0].T + wb[1]

    def _ln(self, x):
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * self.ln[0] + self.ln[1]

    def __call__(self, inp):
        toks, masks, own = [], [], []
        for m in self.mods:
            x, a = inp["x"][m].astype(np.float64), inp["age"][m].astype(np.float64)
            w1, b1, w2, b2 = self.enc[m]
            e = self._gelu(x @ w1.T + b1) @ w2.T + b2
            if self.use_gate:
                gt = np.exp(-self.beta[m] * a[..., None] / 7.0)
                e = gt * e + (1 - gt) * self.prior[m]
            aa = np.stack([np.log1p(a) / math.log(1 + C.A_MAX), a / C.A_MAX], -1)
            t = e + self._lin(aa, self.age_emb[m]) + self.mod_emb[m] + self.dist[None]
            if m == "SMAP":
                toks.append(t[:, :1]); masks.append(inp["mask"][m][:, :1]); own += [-1]
            else:
                toks.append(t); masks.append(inp["mask"][m]); own += list(range(N_DIST))
        T = np.concatenate(toks, 1); M = np.concatenate(masks, 1)
        B = T.shape[0]
        c = self._lin(inp["cal"].astype(np.float64), self.cal)
        T = np.concatenate([T, c[:, None]], 1)
        M = np.concatenate([M, np.ones((B, 1), bool)], 1)
        own = np.array(own + [-2])
        qd = self.dist[None] + c[:, None]
        if self.variant == "cloud":
            Mq = np.broadcast_to(M[:, None, :], (B, N_DIST, M.shape[1]))
        else:
            allow = (own[None] == np.arange(N_DIST)[:, None]) | (own == -2)[None]
            Mq = M[:, None, :] & allow[None]
        z = self._attend(qd, T, Mq)
        h = self._gelu(self._lin(z, self.hr[:2])) @ self.hr[2].T + self.hr[3]
        logit = inp["prior_logit"].astype(np.float64) + h[..., 0]
        out = {"p_rain": 1 / (1 + np.exp(-logit))}
        if self.variant == "cloud":
            zr = self._attend((self.q_reg + c)[:, None], T, M[:, None])
            hs = self._gelu(self._lin(zr, self.hs[:2])) @ self.hs[2].T + self.hs[3]
            out["sm"] = inp["sm_base"].astype(np.float64) + hs[:, 0, 0] * self.sm_sd
        return out

    def _attend(self, qv, T, M):
        q = self._lin(qv, self.W["q"]); k = self._lin(T, self.W["k"]); v = self._lin(T, self.W["v"])
        s = np.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.d)
        s = np.where(M, s, -1e9)
        s = s - s.max(-1, keepdims=True)
        w = np.exp(s); w = w / w.sum(-1, keepdims=True)
        return self._ln(qv + self._lin(np.einsum("bqk,bkd->bqd", w, v), self.W["o"]))


# --------------------------------------------------------------------------------------
# Baselines (scikit-learn) operating on the same inputs
# --------------------------------------------------------------------------------------
def flat_features(inp, district_onehot=True, local_only_for=None):
    """Early-fusion design matrix, one row per (sample, district)."""
    B = inp["cal"].shape[0]
    rows = []
    for j in range(N_DIST):
        parts = []
        mods = C.MODALITIES if local_only_for is None else MOD_LOCAL
        for m in mods:
            if local_only_for is None and m != "SMAP":
                # cloud baseline: own-district features + mean of other districts
                own = inp["x"][m][:, j]
                parts += [own, inp["age"][m][:, j:j + 1] / C.A_MAX, inp["mask"][m][:, j:j + 1]]
                others = [k for k in range(N_DIST) if k != j]
                mk = inp["mask"][m][:, others].astype(np.float32)
                avg = (inp["x"][m][:, others] * mk[..., None]).sum(1) / np.maximum(mk.sum(1, keepdims=True), 1)
                parts.append(avg)
            else:
                jj = 0 if m == "SMAP" else j
                parts += [inp["x"][m][:, jj], inp["age"][m][:, jj:jj + 1] / C.A_MAX, inp["mask"][m][:, jj:jj + 1]]
        parts.append(inp["cal"])
        if district_onehot:
            oh = np.zeros((B, N_DIST), np.float32); oh[:, j] = 1
            parts.append(oh)
        rows.append(np.concatenate(parts, 1))
    return np.stack(rows, 1)   # (B, N, F)


def regional_features(inp):
    parts = []
    for m in C.MODALITIES:
        if m == "SMAP":
            parts += [inp["x"][m][:, 0], inp["age"][m][:, :1] / C.A_MAX, inp["mask"][m][:, :1]]
        else:
            mk = inp["mask"][m].astype(np.float32)
            avg = (inp["x"][m] * mk[..., None]).sum(1) / np.maximum(mk.sum(1, keepdims=True), 1)
            parts += [avg, inp["age"][m].mean(1, keepdims=True) / C.A_MAX]
    parts.append(inp["cal"])
    return np.concatenate(parts, 1)


def save_models(results, path):
    import pickle
    pickle.dump([{"state": r.model.state_dict(), "sm_mu": r.sm_mu, "sm_sd": r.sm_sd,
                  "variant": r.model.variant, "mods": list(r.model.mods), "d": r.model.d,
                  "use_gate": r.model.use_gate} for r in results], open(path, "wb"))


def load_models(path):
    import pickle
    out = []
    for z in pickle.load(open(path, "rb")):
        m = build_scaf(z["variant"], d=z["d"], mods=z["mods"])
        m.load_state_dict(z["state"]); m.sm_mu, m.sm_sd = z["sm_mu"], z["sm_sd"]
        m.use_gate = z["use_gate"]; m.eval()
        out.append(TrainResult(m, z["sm_mu"], z["sm_sd"], []))
    return out


class FastSCAF:
    """Table-driven inference of the regional head of SCAF for the RL environment.

    Encoder outputs E_m(x) depend only on the held acquisition, and the staleness gate and
    age embedding depend only on the age, so both are pre-computed; inference then reduces
    to table look-ups plus one attention step.  Verified to match NumpySCAF/PyTorch."""

    def __init__(self, tr: "TrainResult", sampler: "Sampler"):
        npm = NumpySCAF(tr)
        assert npm.variant == "cloud"
        self.npm = npm
        P = sampler.P
        self.mods = npm.mods
        ages = np.arange(C.A_MAX + 1, dtype=np.float64)
        aa = np.stack([np.log1p(ages) / math.log(1 + C.A_MAX), ages / C.A_MAX], -1)
        self.enc_tab, self.gate_tab, self.age_tab = {}, {}, {}
        for m in self.mods:
            mu, sd = P.norm[m]
            x = np.nan_to_num((P.feat[m] - mu) / sd).astype(np.float64)       # (D,N,f)
            w1, b1, w2, b2 = npm.enc[m]
            self.enc_tab[m] = npm._gelu(x @ w1.T + b1) @ w2.T + b2              # (D,N,d)
            self.gate_tab[m] = (np.exp(-npm.beta[m][None] * ages[:, None] / 7.0) if npm.use_gate
                                else np.ones((len(ages), npm.d)))                # (A,d)
            self.age_tab[m] = npm._lin(aa, npm.age_emb[m])                       # (A,d)
        self.cal_tab = npm._lin(sampler.cal.astype(np.float64), npm.cal)         # (D,d)
        self.sm_feat = P.feat["SMAP"][:, 0, 0].astype(np.float64)
        self.sm_mu, self.sm_sd = tr.sm_mu, tr.sm_sd
        W = npm.W
        self.Wk, self.bk, self.Wv, self.bv = W["k"][0], W["k"][1], W["v"][0], W["v"][1]

    def __call__(self, t_idx, held):
        npm = self.npm
        B = len(t_idx)
        toks, masks = [], []
        for m in self.mods:
            h = held[m] if m != "SMAP" else held[m][:, :1]
            valid = h >= 0
            hh = np.where(valid, h, 0)
            age = t_idx[:, None] - hh
            ok = valid & (age <= C.A_MAX) & (age >= 0)
            ac = np.clip(age, 0, C.A_MAX)
            jj = np.arange(h.shape[1])[None].repeat(B, 0)
            e = self.enc_tab[m][hh, jj]                                           # (B,n,d)
            g = self.gate_tab[m][ac]
            e = g * e + (1 - g) * npm.prior[m]
            t = e + self.age_tab[m][ac] + npm.mod_emb[m] + npm.dist[None, :h.shape[1]]
            toks.append(t); masks.append(ok)
        c = self.cal_tab[np.minimum(t_idx + C.HORIZON, len(self.cal_tab) - 1)]
        T = np.concatenate(toks + [c[:, None]], 1)
        M = np.concatenate(masks + [np.ones((B, 1), bool)], 1)
        qv = (npm.q_reg + c)[:, None]
        z = npm._attend(qv, T, M[:, None])
        hs = npm._gelu(npm._lin(z, npm.hs[:2])) @ npm.hs[2].T + npm.hs[3]
        hsm = held["SMAP"][:, 0]
        okb = (hsm >= 0) & ((t_idx - hsm) <= C.A_MAX)
        base = np.where(okb, self.sm_feat[np.maximum(hsm, 0)], self.sm_mu)
        return base + hs[:, 0, 0] * self.sm_sd
