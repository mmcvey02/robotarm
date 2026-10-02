"""PPO training for the handover policy with vectorised MuJoCo environments."""

import argparse
import json
import os
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn

from .env import HandoverEnv, OBS_DIM, ACT_DIM
from .give_env import GiveEnv

TASKS = {"receive": HandoverEnv, "give": GiveEnv}
from .scripted import scripted_action

EVAL_SEED0 = 10_000_000


def eval_is_distractor(i):
    return i % 7 == 6  # ~14% distractor episodes in the evaluation set


# --------------------------------------------------------------------- workers
def worker(remote, wid, n_envs, seed, task="receive"):
    os.environ["OMP_NUM_THREADS"] = "1"
    envs = [TASKS[task](seed=seed + 1000 * wid + i) for i in range(n_envs)]
    obs = np.stack([e.reset() for e in envs])
    eval_queue = None  # per-env list of (seed, distractor)
    idle = np.zeros(n_envs, bool)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            rews = np.zeros(n_envs, np.float32)
            dones = np.zeros(n_envs, bool)
            truncs = np.zeros(n_envs, bool)
            infos = [None] * n_envs
            for i, e in enumerate(envs):
                if idle[i]:
                    continue
                o, r, d, info = e.step(data[i])
                rews[i], dones[i] = r, d
                if d:
                    truncs[i] = info["timeout"]
                    infos[i] = info
                    if eval_queue is None:
                        o = e.reset()
                    elif eval_queue[i]:
                        s, dis = eval_queue[i].pop(0)
                        o = e.reset(seed=s, distractor=dis)
                    else:
                        idle[i] = True
                obs[i] = o
            teacher = np.stack([scripted_action(e) for e in envs]).astype(np.float32)
            remote.send((obs.copy(), rews, dones, truncs, infos, idle.copy(), teacher))
        elif cmd == "teacher":
            remote.send(np.stack([scripted_action(e) for e in envs]).astype(np.float32))
        elif cmd == "eval":
            eval_queue = data
            idle[:] = False
            for i, e in enumerate(envs):
                if eval_queue[i]:
                    s, dis = eval_queue[i].pop(0)
                    obs[i] = e.reset(seed=s, distractor=dis)
                else:
                    idle[i] = True
            remote.send((obs.copy(), idle.copy()))
        elif cmd == "train":
            eval_queue = None
            idle[:] = False
            obs = np.stack([e.reset() for e in envs])
            remote.send(obs.copy())
        elif cmd == "close":
            remote.close()
            break


class VecEnv:
    def __init__(self, n_workers, envs_per_worker, seed, task="receive"):
        ctx = mp.get_context("fork")
        self.remotes, self.procs = [], []
        self.n_workers, self.epw = n_workers, envs_per_worker
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, w, envs_per_worker, seed, task), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.n = n_workers * envs_per_worker

    def train_mode(self):
        for r in self.remotes:
            r.send(("train", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step(self, actions):
        acts = actions.reshape(self.n_workers, self.epw, -1)
        for r, a in zip(self.remotes, acts):
            r.send(("step", a))
        res = [r.recv() for r in self.remotes]
        obs = np.concatenate([x[0] for x in res])
        rew = np.concatenate([x[1] for x in res])
        done = np.concatenate([x[2] for x in res])
        trunc = np.concatenate([x[3] for x in res])
        infos = sum([x[4] for x in res], [])
        idle = np.concatenate([x[5] for x in res])
        self.teacher = np.concatenate([x[6] for x in res])
        return obs, rew, done, trunc, infos, idle

    def teacher_actions(self):
        for r in self.remotes:
            r.send(("teacher", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def start_eval(self, episodes):
        """episodes: list of (seed, distractor). Distributed round-robin over envs."""
        queues = [[] for _ in range(self.n)]
        for j, ep in enumerate(episodes):
            queues[j % self.n].append(ep)
        for w, r in enumerate(self.remotes):
            r.send(("eval", queues[w * self.epw:(w + 1) * self.epw]))
        res = [r.recv() for r in self.remotes]
        return np.concatenate([x[0] for x in res]), np.concatenate([x[1] for x in res])

    def close(self):
        for r in self.remotes:
            r.send(("close", None))


# --------------------------------------------------------------------- model
class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)

    def state(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


def mlp(i, o, h=(256, 256)):
    layers, d = [], i
    for k in h:
        layers += [nn.Linear(d, k), nn.ELU()]
        d = k
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = mlp(OBS_DIM, ACT_DIM)
        self.v = mlp(OBS_DIM, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()

    def dist(self, o):
        mu = self.pi(o)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, o):
        return self.v(o).squeeze(-1)


# --------------------------------------------------------------------- eval
def evaluate(venv, ac, norm, n_episodes, seed0=EVAL_SEED0):
    eps = [(seed0 + i, eval_is_distractor(i)) for i in range(n_episodes)]
    obs, idle = venv.start_eval(eps)
    infos = []
    while not idle.all():
        with torch.no_grad():
            a = ac.pi(torch.as_tensor(norm(obs))).numpy()
        obs, _, _, _, inf, idle = venv.step(a)
        infos += [x for x in inf if x is not None]
    return infos


def summarize(infos):
    h = [i for i in infos if not i["distractor"]]
    dd = [i for i in infos if i["distractor"]]
    s = {}
    if h:
        s["success"] = float(np.mean([i["success"] for i in h]))
        s["grasp_rate"] = float(np.mean([i["grasped"] for i in h]))
        s["drop_rate"] = float(np.mean([i["fail"] == "drop" for i in h]))
        s["timeout_rate"] = float(np.mean([i["fail"] == "timeout" for i in h]))
        for f in sorted({i["fail"] for i in h if i["fail"]}):
            s[f"fail_{f}"] = float(np.mean([i["fail"] == f for i in h]))
        rd = [i["release_delay"] for i in h if np.isfinite(i.get("release_delay", np.nan))]
        if rd:
            s["release_delay"] = float(np.mean(rd))
        s["unsafe_rate"] = float(np.mean([i["fail"] == "unsafe_contact" for i in h]))
        s["pstop_rate"] = float(np.mean([i["fail"] == "protective_stop" for i in h]))
        s["hand_contact_rate"] = float(np.mean([i["hand_contact"] for i in h]))
        tg = [i["time_to_grasp"] for i in h if i["grasped"] and np.isfinite(i["time_to_grasp"])]
        s["time_to_grasp"] = float(np.mean(tg)) if tg else float("nan")
        s["peak_hand_force"] = float(np.max([i["max_hand_force"] for i in h]))
        s["max_tcp_speed"] = float(np.max([i["max_tcp_speed"] for i in h]))
        s["max_qd_ratio"] = float(np.max([i["max_qd_ratio"] for i in h]))
        s["torque_sat_frac"] = float(np.mean([i["torque_sat_frac"] for i in h]))
        s["jl_frac"] = float(np.mean([i["jl_frac"] for i in h]))
        s["path"] = float(np.mean([i["path"] for i in h]))
        for k, n in enumerate(["cyl", "box", "cap"]):
            sub = [i["success"] for i in h if i["obj_type"] == k]
            s[f"success_{n}"] = float(np.mean(sub)) if sub else float("nan")
        heavy = [i["success"] for i in h if i["mass"] > 0.5]
        s["success_heavy"] = float(np.mean(heavy)) if heavy else float("nan")
        s["n_handover"] = len(h)
    if dd:
        s["distractor_success"] = float(np.mean([i["success"] for i in dd]))
        s["false_reach_rate"] = float(np.mean([i["false_reach"] for i in dd]))
        s["n_distractor"] = len(dd)
    return s


# --------------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    venv = VecEnv(args.workers, args.envs_per_worker, args.seed, args.task)
    N, T = venv.n, args.horizon
    ac = ActorCritic()
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm(OBS_DIM)
    it0, steps = 0, 0
    best = -1.0
    ck = os.path.join(args.out, "ckpt_latest.pt")
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        ac.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        norm.load(s["norm"])
        it0, steps, best = s["it"], s["steps"], s.get("best", -1.0)
        print(f"resumed from it {it0}, {steps} steps", flush=True)
    elif args.init_from:
        s = torch.load(args.init_from, weights_only=False)
        ac.load_state_dict(s["model"])
        norm.load(s["norm"])
        with torch.no_grad():
            ac.log_std.fill_(args.init_log_std)
        opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)
        print(f"initialised actor from {args.init_from}", flush=True)
    obs = venv.train_mode()
    ep_ret = np.zeros(N)
    ep_len = np.zeros(N)
    recent, recent_ret = [], []
    t_start = time.time()
    passes = 0
    for it in range(it0, args.iters):
        frac = min(1.0, steps / args.total_steps)
        lr = args.lr * (1 - 0.7 * frac)
        for g in opt.param_groups:
            g["lr"] = lr
        buf_o = np.zeros((T, N, OBS_DIM), np.float32)
        buf_a = np.zeros((T, N, ACT_DIM), np.float32)
        buf_lp = np.zeros((T, N), np.float32)
        buf_r = np.zeros((T, N), np.float32)
        buf_d = np.zeros((T, N), np.float32)
        buf_v = np.zeros((T + 1, N), np.float32)
        t0 = time.time()
        torch.set_num_threads(1)
        for t in range(T):
            norm.update(obs)
            on = norm(obs)
            with torch.no_grad():
                ot = torch.as_tensor(on)
                dist = ac.dist(ot)
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
                v = ac.value(ot)
            obs, r, d, tr, infos, _ = venv.step(a.numpy())
            r = r * args.rew_scale
            # timeouts are task failures (the human is left waiting) -> treated as terminal
            buf_o[t], buf_a[t], buf_lp[t], buf_r[t], buf_d[t], buf_v[t] = on, a.numpy(), lp.numpy(), r, d, v.numpy()
            ep_ret += r / args.rew_scale
            ep_len += 1
            for i in np.nonzero(d)[0]:
                recent.append(infos[i])
                recent_ret.append(ep_ret[i])
                ep_ret[i] = 0
                ep_len[i] = 0
        with torch.no_grad():
            buf_v[T] = ac.value(torch.as_tensor(norm(obs))).numpy()
        steps += N * T
        t_roll = time.time() - t0
        torch.set_num_threads(4)
        # GAE
        adv = np.zeros((T, N), np.float32)
        last = 0
        for t in reversed(range(T)):
            nd = 1.0 - buf_d[t]
            delta = buf_r[t] + args.gamma * buf_v[t + 1] * nd - buf_v[t]
            last = delta + args.gamma * args.lam * nd * last
            adv[t] = last
        ret = adv + buf_v[:T]
        bo = torch.as_tensor(buf_o.reshape(-1, OBS_DIM))
        ba = torch.as_tensor(buf_a.reshape(-1, ACT_DIM))
        blp = torch.as_tensor(buf_lp.reshape(-1))
        badv = torch.as_tensor(adv.reshape(-1))
        bret = torch.as_tensor(ret.reshape(-1))
        bv_old = torch.as_tensor(buf_v[:T].reshape(-1))
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        M = bo.shape[0]
        kls, clipf, vls = [], [], []
        stop = False
        for ep in range(args.epochs):
            perm = torch.randperm(M)
            for s0 in range(0, M, args.minibatch):
                idx = perm[s0:s0 + args.minibatch]
                dist = ac.dist(bo[idx])
                lp = dist.log_prob(ba[idx]).sum(-1)
                ratio = (lp - blp[idx]).exp()
                pg = -torch.min(ratio * badv[idx],
                                ratio.clamp(1 - args.clip, 1 + args.clip) * badv[idx]).mean()
                v = ac.value(bo[idx])
                v_cl = bv_old[idx] + (v - bv_old[idx]).clamp(-args.clip * 10, args.clip * 10)
                vl = torch.max((v - bret[idx]) ** 2, (v_cl - bret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                if it < args.critic_warmup:
                    loss = 0.5 * vl  # fit the critic to the imitation policy before moving the actor
                else:
                    loss = pg + 0.5 * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (lp - blp[idx])).mean().item()
                kls.append(kl)
                clipf.append(((ratio - 1).abs() > args.clip).float().mean().item())
                vls.append(vl.item())
                if kl > 1.5 * args.target_kl:
                    stop = True
                    break
            if stop:
                break
        with torch.no_grad():
            ac.log_std.clamp_(-2.5, 0.5)
        recent = recent[-400:]
        recent_ret = recent_ret[-400:]
        tr_s = summarize(recent) if recent else {}
        log = dict(it=it + 1, steps=steps, time=time.time(), fps=N * T / (time.time() - t0),
                   roll_time=t_roll, lr=lr, kl=float(np.mean(kls)), clipfrac=float(np.mean(clipf)),
                   vloss=float(np.mean(vls)), std=float(ac.log_std.exp().mean().item()),
                   ep_return=float(np.mean(recent_ret)) if recent_ret else 0.0,
                   **{"train_" + k: v for k, v in tr_s.items()})
        with open(os.path.join(args.out, "train.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        if (it + 1) % 5 == 0:
            print(f"it {it+1} steps {steps/1e6:.2f}M fps {log['fps']:.0f} ret {log['ep_return']:.1f} "
                  f"succ {tr_s.get('success', 0):.3f} grasp {tr_s.get('grasp_rate', 0):.3f} "
                  f"dis {tr_s.get('distractor_success', 0):.2f} std {log['std']:.3f} kl {log['kl']:.4f}",
                  flush=True)

        state = dict(model=ac.state_dict(), opt=opt.state_dict(), norm=norm.state(), it=it + 1,
                     steps=steps, best=best)
        if (it + 1) % args.eval_every == 0:
            infos = evaluate(venv, ac, norm, args.eval_episodes)
            es = summarize(infos)
            es.update(it=it + 1, steps=steps, time=time.time())
            with open(os.path.join(args.out, "eval.jsonl"), "a") as f:
                f.write(json.dumps(es) + "\n")
            print(f"[EVAL] it {it+1} steps {steps/1e6:.2f}M success {es['success']:.3f} "
                  f"grasp {es['grasp_rate']:.3f} drop {es['drop_rate']:.3f} unsafe {es['unsafe_rate']:.3f} "
                  f"distractor {es.get('distractor_success', float('nan')):.3f}", flush=True)
            score = es["success"] * 0.9 + 0.1 * es.get("distractor_success", 0)
            if score > best:
                best = score
                state["best"] = best
                torch.save(state, os.path.join(args.out, "ckpt_best.pt"))
            obs = venv.train_mode()
            ep_ret[:] = 0
            passes = passes + 1 if (es["success"] >= args.target and es.get("distractor_success", 1) >= 0.85) else 0
        torch.save(state, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        if passes >= 2 or steps >= args.total_steps:
            print("target reached" if passes >= 2 else "step budget exhausted", flush=True)
            torch.save(state, os.path.join(args.out, "ckpt_final.pt"))
            break
    venv.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--envs_per_worker", type=int, default=16)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--total_steps", type=float, default=150e6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--ent", type=float, default=0.0)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--rew_scale", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=15)
    p.add_argument("--eval_episodes", type=int, default=210)
    p.add_argument("--target", type=float, default=0.88)
    p.add_argument("--init_from", default=None)
    p.add_argument("--task", default="receive", choices=["receive", "give"])
    p.add_argument("--init_log_std", type=float, default=-1.2)
    p.add_argument("--critic_warmup", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    train(args)


if __name__ == "__main__":
    main()
