"""Privileged scripted controller (uses true object state). Only used to sanity-check
that the task is physically feasible; the learned policy never sees privileged state."""
import numpy as np
from .env import HandoverEnv, V_MAX, W_MAX


def orient_error(Rt, R, k):
    """Rotation vector that aligns the gripper for a side grasp of the object."""
    axis = R[:, 2]
    # desired closing axis: perpendicular to object axis (box: nearest face normal)
    if k == 1:
        cands = [R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]]
        y_des = max(cands, key=lambda c: c @ Rt[:, 1])
    else:
        y = Rt[:, 1] - (Rt[:, 1] @ axis) * axis
        y_des = y / (np.linalg.norm(y) + 1e-9)
    z = Rt[:, 2] - (Rt[:, 2] @ axis) * axis - (Rt[:, 2] @ y_des) * y_des
    z_des = z / (np.linalg.norm(z) + 1e-9)
    x_des = np.cross(y_des, z_des)
    Rd = np.c_[x_des, y_des, z_des]
    return 0.5 * sum(np.cross(Rt[:, i], Rd[:, i]) for i in range(3))


def scripted_give(env):
    """Privileged teacher for the robot-to-human handover."""
    d = env.d
    tcp = d.site_xpos[env.tcp]
    Rt = d.site_xmat[env.tcp].reshape(3, 3)
    a = np.zeros(7)
    a[6] = 1  # keep holding by default
    # keep the tool (and thus the part) in its home orientation
    a[3:6] = np.clip(0.5 * sum(np.cross(Rt[:, i], env.Rt_home[:, i]) for i in range(3)) * 3 / W_MAX, -1, 1)
    if env.release_t is not None or (env.pull_start_t is not None and env.t - env.pull_start_t >= 0.1):
        a[6] = -1  # the person has the part and is pulling: let go, then back away
        if env.release_t is not None:
            a[:3] = np.clip((env.tcp_home - tcp) * 3 / V_MAX, -1, 1)
        return a
    if env.distractor or env.t < env.t_present - 0.2 or env.human_grasp_t is not None:
        tgt = env.tcp_home if (env.distractor or env.t < env.t_present - 0.2) else tcp
    else:
        tgt, _ = env._target()
    a[:3] = np.clip((tgt - tcp) * 5 / V_MAX, -1, 1)
    return a


def scripted_action(env):
    if hasattr(env, "human_grasp_t"):
        return scripted_give(env)
    d = env.d
    tcp = d.site_xpos[env.tcp]
    Rt = d.site_xmat[env.tcp].reshape(3, 3)
    g, R, c = env._true_object()
    a = np.zeros(7)
    a[6] = -1
    if env.distractor or env.t < env.t_present - 0.2:
        a[:3] = np.clip((env.tcp_home - tcp) * 4 / V_MAX, -1, 1)
        return a
    e = g - tcp
    # approach from behind the grasp point along the gripper axis first
    pre = g - 0.06 * Rt[:, 2]
    lateral = e - (e @ Rt[:, 2]) * Rt[:, 2]
    if env.release_t is None and np.linalg.norm(lateral) > 0.012 and e @ Rt[:, 2] > -0.01:
        tgt = pre
    else:
        tgt = g
    a[:3] = np.clip((tgt - tcp) * 5 / V_MAX, -1, 1)
    a[3:6] = np.clip(orient_error(Rt, R, env.k) * 4 / W_MAX, -1, 1)
    if np.linalg.norm(e) < 0.012 or env.release_t is not None or env.mt["contact_both_t"] is not None:
        a[6] = 1
    if env.release_t is not None:
        a[:6] = 0
    return a
