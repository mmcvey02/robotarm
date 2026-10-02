"""Progress dashboard: learning curves on multiple metrics + rendered rollout filmstrip."""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import torch

from .env import HandoverEnv
from .ppo import ActorCritic, RunningNorm, EVAL_SEED0

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TARGET = "#8a8984"


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def smooth(y, k=9):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    out = np.convolve(np.nan_to_num(y), np.ones(k) / k, mode="valid")
    return np.r_[np.full(k - 1, np.nan), out]


def style(ax, title, ylabel=None, pct=False):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("environment steps (millions)", fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    if pct:
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def line(ax, x, y, c, label, lw=2, ls="-", marker=None):
    if marker is None and len(x) < 40:
        marker = "o"
    ax.plot(x, y, color=c, lw=lw, ls=ls, label=label, marker=marker, ms=4)


def legend(ax):
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="best")


def rollout_frames(ckpt, seed, n_frames=6, size=(320, 400)):
    s = torch.load(ckpt, weights_only=False)
    ac = ActorCritic()
    ac.load_state_dict(s["model"])
    norm = RunningNorm(len(s["norm"]["mean"]))
    norm.load(s["norm"])
    env = HandoverEnv(seed=0)
    obs = env.reset(seed=seed, distractor=False)
    r = mujoco.Renderer(env.m, *size)
    frames, done, info = [], False, {}
    while not done:
        with torch.no_grad():
            a = ac.pi(torch.as_tensor(norm(obs[None]))).numpy()[0]
        obs, _, done, info = env.step(a)
        r.update_scene(env.d, camera="overview")
        frames.append((env.t, r.render().copy()))
    r.update_scene(env.d, camera="wrist_cam")
    wrist = r.render().copy()
    idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
    return [frames[i] for i in idx], wrist, info


def load_stages(stages):
    """Concatenate logs of consecutive training stages on a common step axis."""
    tr, ev, bounds, off = [], [], [], 0
    for run, label in stages:
        t = load_jsonl(os.path.join(run, "train.jsonl"))
        e = load_jsonl(os.path.join(run, "eval.jsonl"))
        if not t and not e:
            continue
        for x in t:
            x["steps"] += off
        for x in e:
            x["steps"] += off
        tr += t
        ev += e
        bounds.append((off, label))
        off = max([x["steps"] for x in t + e])
    return tr, ev, bounds


def make_report(stages, out, ckpt=None, seed=None, baseline=None):
    tr, ev, bounds = load_stages(stages)
    base = load_jsonl(os.path.join(baseline, "eval.jsonl")) if baseline else []
    fig = plt.figure(figsize=(16, 15), facecolor="white")
    gs = fig.add_gridspec(5, 4, height_ratios=[0.22, 1, 1, 1, 1.15], hspace=0.5, wspace=0.28)

    xs = np.array([t["steps"] for t in tr]) / 1e6 if tr else np.array([])
    xe = np.array([e["steps"] for e in ev]) / 1e6 if ev else np.array([])
    g = lambda L, k: np.array([x.get(k, np.nan) for x in L], float)

    # ---- header / KPI tiles
    axh = fig.add_subplot(gs[0, :])
    axh.axis("off")
    times = [x["time"] for x in tr + ev + base]
    hours = (max(times) - min(times)) / 3600 if times else 0
    last = ev[-1] if ev else {}
    best = max(ev, key=lambda e: e["success"]) if ev else {}
    tiles = [
        ("Eval success (latest)", f"{last.get('success', float('nan')):.1%}"),
        ("Best eval success", f"{best.get('success', float('nan')):.1%}"),
        ("Target", "> 85%"),
        ("Env steps", f"{max(np.r_[xs, xe, 0]):.1f} M"),
        ("Simulated time", f"{max(np.r_[xs, xe, 0])*1e6*0.05/3600:.0f} h"),
        ("Wall-clock", f"{hours:.2f} h"),
    ]
    for i, (k, v) in enumerate(tiles):
        x0 = i / len(tiles)
        axh.text(x0, 0.95, k, fontsize=10, color=INK2, transform=axh.transAxes, va="top")
        axh.text(x0, 0.45, v, fontsize=22, color=INK, fontweight="bold", transform=axh.transAxes, va="top")
    fig.suptitle("Robot-arm handover policy — training progress (MuJoCo; teacher–student DAgger)", x=0.06, ha="left",
                 y=0.93, fontsize=16, fontweight="bold", color=INK)

    # ---- 1 success
    ax = fig.add_subplot(gs[1, 0:2])
    if len(xs):
        line(ax, xs, smooth(g(tr, "train_success")), C2, "training rollouts (stochastic, smoothed)", lw=1.5)
    if base:
        line(ax, np.array([e["steps"] for e in base]) / 1e6, g(base, "success"), C3,
             "baseline: PPO from scratch (eval)", lw=1.5)
    if len(xe):
        line(ax, xe, g(ev, "success"), C1, "evaluation (deterministic, fixed 180 offers)", marker="o")
    ax.axhline(0.85, color=TARGET, ls="--", lw=1.2)
    for b0, lab in bounds:
        ax.axvline(b0 / 1e6, color=GRID, lw=1.5)
        ax.text(b0 / 1e6, 0.95, " " + lab, color=INK2, fontsize=8)
    ax.text(ax.get_xlim()[0], 0.865, " 85% target", color=INK2, fontsize=8)
    style(ax, "Handover success rate (grasped + held 1 s after human lets go)", pct=True)
    legend(ax)

    # ---- 2 outcome breakdown
    ax = fig.add_subplot(gs[1, 2:4])
    if len(xe):
        line(ax, xe, g(ev, "grasp_rate"), C1, "grasp established")
        line(ax, xe, g(ev, "drop_rate"), C2, "dropped after release")
        line(ax, xe, g(ev, "timeout_rate"), C3, "timed out (no grasp in 9 s)")
    style(ax, "Outcome breakdown (evaluation)", pct=True)
    legend(ax)

    # ---- 3 safety
    ax = fig.add_subplot(gs[2, 0])
    if len(xe):
        line(ax, xe, g(ev, "hand_contact_rate"), C1, "any robot-hand contact")
        line(ax, xe, g(ev, "unsafe_rate"), C2, "contact > 140 N (ISO/TS 15066)")
        line(ax, xe, g(ev, "pstop_rate"), C3, "protective stop (speed limit)")
    style(ax, "Human safety", pct=True)
    legend(ax)

    # ---- 4 intent detection
    ax = fig.add_subplot(gs[2, 1])
    if len(xe):
        line(ax, xe, g(ev, "distractor_success"), C1, "correctly ignored non-offers")
        line(ax, xe, g(ev, "false_reach_rate"), C2, "false reach")
    style(ax, "Intent: human not offering", pct=True)
    legend(ax)

    # ---- 5 time to grasp
    ax = fig.add_subplot(gs[2, 2])
    if len(xe):
        line(ax, xe, g(ev, "time_to_grasp"), C1, "")
    style(ax, "Offer → secure grasp time", "seconds")

    # ---- 6 return
    ax = fig.add_subplot(gs[2, 3])
    fe = "reports/final_eval.json"
    if os.path.exists(fe):
        with open(fe) as f:
            r = json.load(f)
        so = r["success_only"]
        lo, hi = r["success_ci95"]
        txt = (f"Held-out test ({r['n_handover']} offers, {r['n_distractor']} non-offers)\n\n"
               f"Success: {r['success']:.1%}  (95% CI {lo:.1%}–{hi:.1%})\n"
               f"Non-offers ignored: {r['distractor_success']:.0%}\n"
               f"Failures: drop {r['drop_rate']:.1%}, timeout {r['timeout_rate']:.1%},\n"
               f"  >140 N contact {r['unsafe_rate']:.1%}, prot. stop {r['pstop_rate']:.1%}\n\n"
               f"In successful handovers:\n"
               f"  peak joint speed {so['max_qd_ratio']:.2f}× limit\n"
               f"  peak TCP speed {so['max_tcp_speed']:.2f} m/s\n"
               f"  torque-limited cycles {so['torque_sat_frac']:.2%}\n"
               f"  offer→grasp {so['mean_time_to_grasp']:.2f} s")
        ax.axis("off")
        ax.set_title("Final held-out evaluation", loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.text(0, 1, txt, va="top", fontsize=9, color=INK, transform=ax.transAxes, family="monospace")
    else:
        if len(xs):
            line(ax, xs, smooth(g(tr, "ep_return")), C1, "")
        style(ax, "Episode return (training)", "return")

    # ---- 7 physical constraints
    ax = fig.add_subplot(gs[3, 0])
    if len(xe):
        line(ax, xe, g(ev, "max_qd_ratio"), C1, "peak joint speed / 180°/s limit")
    ax.axhline(1.0, color=TARGET, ls="--", lw=1.2)
    style(ax, "Peak joint speed (all eps)", "ratio")
    legend(ax)

    ax = fig.add_subplot(gs[3, 1])
    if len(xe):
        line(ax, xe, g(ev, "torque_sat_frac"), C1, "servo cycles at torque limit")
        line(ax, xe, g(ev, "jl_frac"), C2, "steps at a joint position limit")
    style(ax, "Torque / position limit usage", pct=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    legend(ax)

    ax = fig.add_subplot(gs[3, 2])
    if len(xe):
        line(ax, xe, g(ev, "max_tcp_speed"), C1, "peak TCP speed")
    ax.axhline(0.6, color=TARGET, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0], 0.62, " 0.6 m/s cap (0.25 m/s near hand)", color=INK2, fontsize=8)
    style(ax, "Peak tool speed (all eps)", "m/s")

    # ---- 8 success by object type (latest eval)
    ax = fig.add_subplot(gs[3, 3])
    if last:
        names = ["cylinder", "box", "capsule", "heavy"]
        vals = [last.get("success_cyl", np.nan), last.get("success_box", np.nan),
                last.get("success_cap", np.nan), last.get("success_heavy", np.nan)]
        bars = ax.bar(names, vals, color=C1, width=0.6, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, (v if np.isfinite(v) else 0) + 0.02, f"{v:.0%}",
                    ha="center", fontsize=9, color=INK)
        ax.axhline(0.85, color=TARGET, ls="--", lw=1.2)
    style(ax, "Success by object (latest eval)", pct=True)
    ax.set_xlabel("")

    # ---- filmstrip
    if ckpt and os.path.exists(ckpt):
        frames, wrist, info = rollout_frames(ckpt, seed if seed is not None else EVAL_SEED0 + 3)
        sub = gs[4, :].subgridspec(1, len(frames) + 1, wspace=0.04)
        for i, (t, im) in enumerate(frames):
            ax = fig.add_subplot(sub[0, i])
            ax.imshow(im)
            ax.axis("off")
            ax.set_title(f"t = {t:.2f} s", fontsize=9, color=INK2)
            if i == 0:
                first = ax
        ax = fig.add_subplot(sub[0, len(frames)])
        ax.imshow(wrist)
        ax.axis("off")
        ax.set_title("wrist camera (final)", fontsize=9, color=INK2)
        outcome = "SUCCESS" if info.get("success") else f"FAIL ({info.get('fail')})"
        first.text(0, 1.2, f"Sample evaluation rollout with current best policy — outcome: {outcome}",
                   transform=first.transAxes, fontsize=11, fontweight="bold", color=INK)
    fig.savefig(out, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="runs/dagger:1 DAgger,runs/dagger2:1b DAgger (protective stop),runs/ppo_ft:2 PPO fine-tune")
    p.add_argument("--baseline", default="runs/ppo_b")
    p.add_argument("--out", default="reports/progress.png")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    stages = [tuple(x.split(":", 1)) for x in a.stages.split(",")]
    print(make_report(stages, a.out, a.ckpt, a.seed, a.baseline))
