"""Multi-agent reinforcement learning for the slow-timescale environment.

Algorithms
----------
* CoVIR (proposed): cooperative value-of-information retrieval.  Parameter-shared
  decentralised actors pi(a_i | o_i) with action masking, a centralised critic V(s)
  (CTDE, MAPPO-style), innovation/VoI observation features and primal-dual Lagrange
  multipliers for the slot-overflow and terminal-depletion constraints.
* IPPO: same actor, decentralised critics V(o_i), fixed penalties.
* Central-PPO: one controller observing the global state and emitting all agents'
  actions (centralised execution), fixed penalties.
* QMIX (Rashid et al., ICML 2018): per-agent Q networks with a monotonic hyper-network
  mixer conditioned on the global state, fixed penalties.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import IS_NTN, N, N_ACT, SlowEnv

torch.set_num_threads(1)


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


@dataclass
class PPOConfig:
    updates: int = 200
    n_envs: int = 8
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    ent_start: float = 0.02
    ent_end: float = 0.001
    vf_coef: float = 0.5
    max_grad: float = 0.5
    lam_lr: float = 0.05
    eps_overflow: float = 0.02      # target dropped-slot rate (fraction of K_N per day)
    eps_depletion: float = 0.01     # target fraction of terminal-days with an empty battery


def shaped_reward(r, info, env, lam_o, lam_d):
    ovf = info["overflow"] / max(env.cfg.k_ntn, 1)
    dep = info["depleted"] / max(IS_NTN.sum(), 1)
    return r - lam_o * ovf - lam_d * dep, ovf, dep


class PPOAgent:
    """kind in {'covir', 'ippo', 'central'}"""

    def __init__(self, kind, obs_dim, state_dim, cfg: PPOConfig, seed=0, lagrangian=None):
        torch.manual_seed(seed)
        self.kind, self.cfg = kind, cfg
        self.lagrangian = (kind == "covir") if lagrangian is None else lagrangian
        if kind == "central":
            self.actor = mlp(state_dim, N * N_ACT, 128)
            self.critic = mlp(state_dim, 1, 128)
        else:
            self.actor = mlp(obs_dim, N_ACT)
            self.critic = mlp(state_dim if kind == "covir" else obs_dim, 1)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr)
        self.lam_o, self.lam_d = 0.2, 1.0          # initial / fixed penalty weights
        self.ret_mu, self.ret_sd = 0.0, 1.0

    def logits(self, obs, state, mask):
        if self.kind == "central":
            lg = self.actor(state).view(-1, N, N_ACT)
        else:
            lg = self.actor(obs)
        return lg.masked_fill(~mask, -1e9)

    def value(self, obs, state):
        if self.kind == "ippo":
            return self.critic(obs).squeeze(-1)                     # (B,N)
        v = self.critic(state).squeeze(-1)                          # (B,)
        return v.unsqueeze(-1).expand(-1, N)

    def act(self, env, obs, greedy=False):
        o = torch.tensor(obs); s = torch.tensor(env.state(obs)); m = torch.tensor(env.action_mask())
        with torch.no_grad():
            lg = self.logits(o, s, m)
            if greedy:
                a = lg.argmax(-1)
            else:
                a = torch.distributions.Categorical(logits=lg).sample()
        return a.numpy(), False

    # ------------------------------------------------------------------ checkpoint selection
    def snapshot(self):
        return {"actor": copy.deepcopy(self.actor.state_dict()), "critic": copy.deepcopy(self.critic.state_dict()),
                "lam": (self.lam_o, self.lam_d), "ret": (self.ret_mu, self.ret_sd)}

    def restore(self, snap):
        self.actor.load_state_dict(snap["actor"]); self.critic.load_state_dict(snap["critic"])
        self.lam_o, self.lam_d = snap["lam"]; self.ret_mu, self.ret_sd = snap["ret"]

    # ------------------------------------------------------------------ training
    def train(self, make_env, log_every=20, verbose=False, critic_warmup=0, select=None):
        """critic_warmup: number of initial updates in which only the critic is trained
        (used after behaviour-cloning initialisation so that a random critic does not
        destroy the cloned policy).
        select: optional dict(make_env=callable() -> fresh validation env on the TRAINING
        period with a fixed seed, every=int).  The greedy policy is evaluated every
        `every` updates and the best feasible checkpoint (constraints (c_ovf), (c_dep)
        satisfied, highest reward) is restored at the end; see `select_score`."""
        cfg = self.cfg
        env = make_env(cfg.n_envs)
        hist = []
        sel_log, best = [], None
        t0 = time.time()
        for u in range(cfg.updates):
            frac = u / max(cfg.updates - 1, 1)
            ent = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * frac
            buf = {k: [] for k in ["obs", "state", "mask", "act", "logp", "rew", "val", "done"]}
            obs = env.reset()
            ep_r, ep_ovf, ep_dep = [], [], []
            while not env.done.all():
                o = torch.tensor(obs); s = torch.tensor(env.state(obs)); mk = torch.tensor(env.action_mask())
                with torch.no_grad():
                    lg = self.logits(o, s, mk)
                    dist = torch.distributions.Categorical(logits=lg)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                    v = self.value(o, s)
                obs2, r, d, info = env.step(a.numpy())
                rs, ovf, dep = shaped_reward(r, info, env, self.lam_o, self.lam_d)
                for k, x in zip(["obs", "state", "mask", "act", "logp", "val"], [o, s, mk, a, lp, v]):
                    buf[k].append(x)
                buf["rew"].append(torch.tensor(rs, dtype=torch.float32))
                buf["done"].append(torch.tensor(d, dtype=torch.float32))
                ep_r.append(r); ep_ovf.append(ovf); ep_dep.append(dep)
                obs = obs2
            T = len(buf["rew"])
            rew = torch.stack(buf["rew"])                  # (T,B)
            val = torch.stack(buf["val"])                  # (T,B,N)
            done = torch.zeros_like(torch.stack(buf["done"]))   # episodes end by time limit only
            with torch.no_grad():
                v_last = self.value(torch.tensor(obs), torch.tensor(env.state(obs)))
            # GAE with team reward broadcast to agents
            adv = torch.zeros_like(val)
            last = torch.zeros_like(val[0])
            vn = val * self.ret_sd + self.ret_mu
            vl_n = v_last * self.ret_sd + self.ret_mu
            for t in reversed(range(T)):
                nv = vn[t + 1] if t + 1 < T else vl_n
                nonterm = 1.0 - done[t].unsqueeze(-1)
                delta = rew[t].unsqueeze(-1) + cfg.gamma * nv * nonterm - vn[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                adv[t] = last
            ret = adv + vn
            # running return normalisation for the critic
            self.ret_mu = 0.9 * self.ret_mu + 0.1 * float(ret.mean())
            self.ret_sd = 0.9 * self.ret_sd + 0.1 * float(ret.std() + 1e-6)
            ret_n = (ret - self.ret_mu) / self.ret_sd
            O = torch.stack(buf["obs"]).flatten(0, 1); S = torch.stack(buf["state"]).flatten(0, 1)
            M = torch.stack(buf["mask"]).flatten(0, 1); A = torch.stack(buf["act"]).flatten(0, 1)
            LP = torch.stack(buf["logp"]).flatten(0, 1)
            ADV = adv.flatten(0, 1); RET = ret_n.flatten(0, 1)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            n = O.shape[0]
            for _ in range(cfg.epochs):
                perm = torch.randperm(n)
                for mb in perm.chunk(cfg.minibatches):
                    lg = self.logits(O[mb], S[mb], M[mb])
                    dist = torch.distributions.Categorical(logits=lg)
                    lp = dist.log_prob(A[mb])
                    ratio = torch.exp(lp - LP[mb])
                    if self.kind == "central":       # joint policy: product over agents
                        ratio = torch.exp((lp - LP[mb]).sum(-1, keepdim=True)).expand(-1, N)
                    s1 = ratio * ADV[mb]; s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADV[mb]
                    pl = -torch.min(s1, s2).mean()
                    vl = ((self.value(O[mb], S[mb]) - RET[mb]) ** 2).mean()
                    el = dist.entropy().mean()
                    if u < critic_warmup:            # critic-only warm-up
                        loss = cfg.vf_coef * vl
                    else:
                        loss = pl + cfg.vf_coef * vl - ent * el
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad)
                    self.opt.step()
            m_ovf, m_dep = float(np.mean(ep_ovf)), float(np.mean(ep_dep))
            if self.lagrangian:   # dual ascent on the constraint multipliers
                self.lam_o = max(0.0, self.lam_o + cfg.lam_lr * (m_ovf - cfg.eps_overflow) * 10)
                self.lam_d = max(0.0, self.lam_d + cfg.lam_lr * (m_dep - cfg.eps_depletion) * 10)
            hist.append({"update": u, "reward": float(np.mean(ep_r)), "overflow": m_ovf,
                         "depletion": m_dep, "lam_o": self.lam_o, "lam_d": self.lam_d,
                         "time": time.time() - t0})
            if select is not None and u >= critic_warmup and ((u + 1) % select["every"] == 0 or u == cfg.updates - 1):
                sc = select_score(self, select["make_env"](), cfg.eps_overflow, cfg.eps_depletion)
                sc["update"] = u
                sel_log.append(sc)
                if best is None or sc["key"] < best[0]:
                    best = (sc["key"], self.snapshot(), u)
            if verbose and u % log_every == 0:
                print(self.kind, u, hist[-1], flush=True)
        if best is not None:
            self.restore(best[1])
            self.selected_update = best[2]
            self.selection_log = [{k: v for k, v in s.items() if k != "key"} for s in sel_log]
        return hist


def select_score(agent, venv, eps_o, eps_d):
    """Greedy evaluation of `agent` on a validation copy of the training simulator.
    Returns the mean daily reward, constraint values and a lexicographic key:
    feasible checkpoints first (ordered by reward), then by total violation."""
    obs = venv.reset()
    R, O, D = [], [], []
    while not venv.done.all():
        a, _ = agent.act(venv, obs, greedy=True)
        obs, r, d, info = venv.step(a)
        R.append(r); O.append(info["overflow"] / max(venv.cfg.k_ntn, 1)); D.append(info["depleted"] / max(IS_NTN.sum(), 1))
    rew, ovf, dep = float(np.mean(R)), float(np.mean(O)), float(np.mean(D))
    viol = max(0.0, ovf - eps_o) + max(0.0, dep - eps_d)
    return {"reward": rew, "overflow": ovf, "depletion": dep, "key": (viol > 0, viol if viol > 0 else -rew)}


class GreedyWrap:
    """Evaluation wrapper: deterministic (arg-max) execution of a trained agent."""

    def __init__(self, agent, name):
        self.agent, self.name = agent, name

    def act(self, env, obs):
        return self.agent.act(env, obs, greedy=True)


# ======================================================================================
# QMIX
# ======================================================================================
class Mixer(nn.Module):
    def __init__(self, state_dim, n_agents, embed=32):
        super().__init__()
        self.n, self.e = n_agents, embed
        self.hw1 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, n_agents * embed))
        self.hb1 = nn.Linear(state_dim, embed)
        self.hw2 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, embed))
        self.hb2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q, s):            # q (B,N), s (B,S)
        w1 = torch.abs(self.hw1(s)).view(-1, self.n, self.e)
        h = torch.nn.functional.elu(torch.bmm(q.unsqueeze(1), w1).squeeze(1) + self.hb1(s))
        w2 = torch.abs(self.hw2(s)).unsqueeze(-1)
        return torch.bmm(h.unsqueeze(1), w2).squeeze(-1).squeeze(-1) + self.hb2(s).squeeze(-1)


@dataclass
class QMIXConfig:
    env_steps: int = 6000          # joint environment steps (x n_envs transitions)
    n_envs: int = 16
    gamma: float = 0.95
    lr: float = 5e-4
    batch: int = 256
    buffer: int = 60000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.7
    target_every: int = 200
    grad_clip: float = 10.0


class QMIXAgent:
    def __init__(self, obs_dim, state_dim, cfg: QMIXConfig, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.q = mlp(obs_dim, N_ACT)
        self.mix = Mixer(state_dim, N)
        self.tq, self.tmix = copy.deepcopy(self.q), copy.deepcopy(self.mix)
        self.opt = torch.optim.Adam(list(self.q.parameters()) + list(self.mix.parameters()), lr=cfg.lr)
        self.lam_o, self.lam_d = 0.2, 1.0

    def act(self, env, obs, greedy=True, eps=0.0):
        m = env.action_mask()
        with torch.no_grad():
            qv = self.q(torch.tensor(obs)).masked_fill(~torch.tensor(m), -1e9).numpy()
        a = qv.argmax(-1)
        if not greedy and eps > 0:
            rnd = self.rng.random(a.shape) < eps
            for b, j in zip(*np.nonzero(rnd)):
                a[b, j] = self.rng.choice(np.flatnonzero(m[b, j]))
        return a, False

    def snapshot(self):
        return {"q": copy.deepcopy(self.q.state_dict()), "mix": copy.deepcopy(self.mix.state_dict())}

    def restore(self, snap):
        self.q.load_state_dict(snap["q"]); self.mix.load_state_dict(snap["mix"])

    def train(self, make_env, verbose=False, select=None, eps_o=0.02, eps_d=0.01):
        """select: as in PPOAgent.train, with `every` counted in joint environment steps."""
        cfg = self.cfg
        env = make_env(cfg.n_envs)
        sel_log, best = [], None
        cap = cfg.buffer
        Sd, Od = env.state_dim, env.obs_dim
        buf = {"s": np.zeros((cap, Sd), np.float32), "o": np.zeros((cap, N, Od), np.float32),
               "a": np.zeros((cap, N), np.int64), "r": np.zeros(cap, np.float32),
               "s2": np.zeros((cap, Sd), np.float32), "o2": np.zeros((cap, N, Od), np.float32),
               "m2": np.zeros((cap, N, N_ACT), bool), "d": np.zeros(cap, np.float32)}
        ptr, size, step, grad_steps = 0, 0, 0, 0
        hist = []
        t0 = time.time()
        obs = env.reset()
        ep_r, ep_o = [], []
        while step < cfg.env_steps:
            eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / (cfg.eps_frac * cfg.env_steps))
            s = env.state(obs)
            a, _ = self.act(env, obs, greedy=False, eps=eps)
            obs2, r, d, info = env.step(a)
            rs, ovf, dep = shaped_reward(r, info, env, self.lam_o, self.lam_d)
            ep_r.append(r.mean()); ep_o.append(ovf.mean())
            s2 = env.state(obs2); m2 = env.action_mask()
            for b in range(env.B):
                buf["s"][ptr] = s[b]; buf["o"][ptr] = obs[b]; buf["a"][ptr] = a[b]; buf["r"][ptr] = rs[b]
                buf["s2"][ptr] = s2[b]; buf["o2"][ptr] = obs2[b]; buf["m2"][ptr] = m2[b]; buf["d"][ptr] = 0.0
                ptr = (ptr + 1) % cap; size = min(size + 1, cap)
            obs = obs2
            step += 1
            if env.done.all():
                obs = env.reset()
                hist.append({"step": step, "reward": float(np.mean(ep_r)), "overflow": float(np.mean(ep_o)),
                             "eps": eps, "time": time.time() - t0})
                ep_r, ep_o = [], []
                if verbose and len(hist) % 20 == 0:
                    print("qmix", hist[-1], flush=True)
            if size >= cfg.batch:
                for _ in range(2):
                    idx = self.rng.integers(0, size, cfg.batch)
                    S = torch.tensor(buf["s"][idx]); O = torch.tensor(buf["o"][idx]); A = torch.tensor(buf["a"][idx])
                    R = torch.tensor(buf["r"][idx]); S2 = torch.tensor(buf["s2"][idx]); O2 = torch.tensor(buf["o2"][idx])
                    M2 = torch.tensor(buf["m2"][idx]); Dn = torch.tensor(buf["d"][idx])
                    qa = self.q(O).gather(-1, A.unsqueeze(-1)).squeeze(-1)
                    qt = self.mix(qa, S)
                    with torch.no_grad():
                        a2 = self.q(O2).masked_fill(~M2, -1e9).argmax(-1, keepdim=True)    # double Q
                        q2 = self.tq(O2).gather(-1, a2).squeeze(-1)
                        y = R + cfg.gamma * (1 - Dn) * self.tmix(q2, S2)
                    loss = ((qt - y) ** 2).mean()
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(self.q.parameters()) + list(self.mix.parameters()), cfg.grad_clip)
                    self.opt.step(); grad_steps += 1
                    if grad_steps % cfg.target_every == 0:
                        self.tq.load_state_dict(self.q.state_dict()); self.tmix.load_state_dict(self.mix.state_dict())
            if select is not None and size >= cfg.batch and (step % select["every"] == 0 or step == cfg.env_steps):
                sc = select_score(self, select["make_env"](), eps_o, eps_d)
                sc["update"] = step
                sel_log.append(sc)
                if best is None or sc["key"] < best[0]:
                    best = (sc["key"], self.snapshot(), step)
        if best is not None:
            self.restore(best[1])
            self.selected_update = best[2]
            self.selection_log = [{k: v for k, v in s.items() if k != "key"} for s in sel_log]
        return hist


def bc_pretrain(agent: PPOAgent, make_env, teacher, rollouts: int = 60, epochs: int = 30, lr: float = 1e-3,
                seed: int = 0):
    """Behaviour-cloning initialisation of a decentralised actor from a (centralised)
    teacher policy, using the masked cross-entropy of the teacher's per-agent actions."""
    torch.manual_seed(seed)
    env = make_env(agent.cfg.n_envs)
    O, Mk, A = [], [], []
    for _ in range(rollouts):
        obs = env.reset()
        while not env.done.all():
            a, central = teacher.act(env, obs)
            O.append(torch.tensor(obs)); Mk.append(torch.tensor(env.action_mask()))
            A.append(torch.tensor(a))
            obs, r, d, info = env.step(a, centralized_schedule=False)
    O = torch.cat(O).flatten(0, 0); Mk = torch.cat(Mk); A = torch.cat(A)
    O = O.reshape(-1, N, O.shape[-1]); Mk = Mk.reshape(-1, N, N_ACT); A = A.reshape(-1, N)
    # teacher actions are always feasible except when masked by battery; clip to mask
    ok = Mk.gather(-1, A.unsqueeze(-1)).squeeze(-1)
    A = torch.where(ok, A, torch.zeros_like(A))
    opt = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    n = O.shape[0]
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for mb in perm.chunk(max(1, n // 512)):
            lg = agent.logits(O[mb], O[mb].reshape(len(mb), -1), Mk[mb])
            loss = nn.functional.cross_entropy(lg.reshape(-1, N_ACT), A[mb].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(mb)
        hist.append(tot / n)
    with torch.no_grad():
        lg = agent.logits(O, O.reshape(n, -1), Mk)
        acc = float((lg.argmax(-1) == A).float().mean())
    return {"bc_loss": hist, "bc_accuracy": acc, "bc_samples": int(n)}
