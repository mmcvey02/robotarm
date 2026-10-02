"""Large held-out evaluation of a checkpoint (seeds disjoint from training and periodic evals)."""

import argparse
import json

import numpy as np
import torch

from .ppo import VecEnv, ActorCritic, RunningNorm, evaluate, summarize

HELDOUT_SEED0 = 50_000_000


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=1400)  # ~1200 offers + ~200 non-offers
    p.add_argument("--out", default=None)
    a = p.parse_args()
    s = torch.load(a.ckpt, weights_only=False)
    ac = ActorCritic()
    ac.load_state_dict(s["model"])
    norm = RunningNorm(len(s["norm"]["mean"]))
    norm.load(s["norm"])
    venv = VecEnv(4, 16, seed=123)
    infos = evaluate(venv, ac, norm, a.episodes, seed0=HELDOUT_SEED0)
    venv.close()
    res = summarize(infos)
    h = [i for i in infos if not i["distractor"]]
    k = sum(i["success"] for i in h)
    lo, hi = wilson(k, len(h))
    res["success_ci95"] = [lo, hi]
    fails = {}
    for i in h:
        if not i["success"]:
            fails[i["fail"]] = fails.get(i["fail"], 0) + 1
    res["failure_modes"] = fails
    ok = [i for i in h if i["success"]]
    # physical-constraint usage within successful handovers (the deployable behaviour)
    res["success_only"] = dict(
        max_qd_ratio=float(max(i["max_qd_ratio"] for i in ok)),
        max_tcp_speed=float(max(i["max_tcp_speed"] for i in ok)),
        max_hand_force=float(max(i["max_hand_force"] for i in ok)),
        torque_sat_frac=float(np.mean([i["torque_sat_frac"] for i in ok])),
        jl_frac=float(np.mean([i["jl_frac"] for i in ok])),
        mean_time_to_grasp=float(np.nanmean([i["time_to_grasp"] for i in ok])),
        mean_episode_time=float(np.mean([i["episode_time"] for i in ok])),
    )
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
