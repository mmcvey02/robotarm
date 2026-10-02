"""Teacher-student imitation (DAgger).

The teacher is the privileged scripted controller (it reads the true object pose).
The student is the deployable policy network: it sees only the noisy, delayed
sensor observations. The student's own rollouts are relabelled with teacher
actions, so it learns to recover from its own mistakes. The result initialises
PPO fine-tuning.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .env import OBS_DIM, ACT_DIM
from .ppo import VecEnv, ActorCritic, RunningNorm, evaluate, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/dagger")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--envs_per_worker", type=int, default=16)
    p.add_argument("--max_data", type=int, default=1_500_000)
    p.add_argument("--grad_steps", type=int, default=150)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=210)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--init", default=None, help="warm-start student weights + obs normaliser")
    p.add_argument("--beta_iters", type=float, default=8, help="iterations over which teacher mixing decays")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    venv = VecEnv(a.workers, a.envs_per_worker, a.seed)
    ac = ActorCritic()
    opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
    norm = RunningNorm(OBS_DIM)
    if a.init:
        s0 = torch.load(a.init, weights_only=False)
        ac.load_state_dict(s0["model"])
        norm.load(s0["norm"])
    N = venv.n
    obs = venv.train_mode()
    teacher = venv.teacher_actions()
    data_o, data_a = [], []
    steps = 0
    use_teacher = np.ones(N, bool) if a.beta_iters > 0 else np.zeros(N, bool)
    for it in range(a.iters):
        beta = max(0.0, 1.0 - it / a.beta_iters) if a.beta_iters > 0 else 0.0  # probability an episode is driven by the teacher
        t0 = time.time()
        for t in range(a.horizon):
            data_o.append(obs.copy())
            data_a.append(teacher.copy())
            norm.update(obs)
            with torch.no_grad():
                sa = ac.pi(torch.as_tensor(norm(obs))).numpy()
            sa = sa + rng.normal(scale=0.1, size=sa.shape)
            act = np.where(use_teacher[:, None], teacher, sa)
            obs, r, d, tr, infos, _ = venv.step(act)
            teacher = venv.teacher
            for i in np.nonzero(d)[0]:
                use_teacher[i] = rng.random() < beta
            steps += N
        O = np.concatenate(data_o)[-a.max_data:]
        A = np.concatenate(data_a)[-a.max_data:]
        data_o, data_a = [O], [A]
        On = torch.as_tensor(norm(O))
        At = torch.as_tensor(A)
        # gripper dimension is a binary command -> weight it more (timing matters)
        w = torch.tensor([1, 1, 1, 0.5, 0.5, 0.5, 2.0])
        losses = []
        for _ in range(a.grad_steps):
            idx = torch.randint(0, len(On), (4096,))
            pred = ac.pi(On[idx])
            loss = ((pred - At[idx]) ** 2 * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log = dict(it=it + 1, steps=steps, data=len(O), beta=beta, loss=float(np.mean(losses[-50:])),
                   time=time.time(), dt=time.time() - t0)
        print(json.dumps(log), flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            es = summarize(evaluate(venv, ac, norm, a.eval_episodes))
            es.update(it=it + 1, steps=steps, time=time.time())
            print("[EVAL]", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in es.items()}),
                  flush=True)
            with open(os.path.join(a.out, "eval.jsonl"), "a") as f:
                f.write(json.dumps(es) + "\n")
            torch.save(dict(model=ac.state_dict(), norm=norm.state(), it=0, steps=0, eval=es),
                       os.path.join(a.out, f"bc_it{it+1}.pt"))
            obs = venv.train_mode()
            teacher = venv.teacher_actions()
            use_teacher[:] = rng.random(N) < beta
    venv.close()


if __name__ == "__main__":
    main()
