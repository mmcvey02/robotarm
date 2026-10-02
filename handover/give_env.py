"""Robot-to-human handover ("give") environment.

The robot starts at home holding a part. A person may reach out an open hand to
receive it. The robot must:
  * recognise a real reach-to-receive (and ignore other hand motion),
  * bring the part's free end into the person's hand without striking it,
  * keep holding while the person closes their grip,
  * release when it feels the person pull (wrist F/T sensor), neither before
    (the part would fall) nor too late (tug-of-war).

Same arm, gripper, sensors, control stack and safety monitor as the receive task.
"""

import numpy as np
import mujoco

from .env import (
    HandoverEnv, CTRL_DT, FINGER_OPEN, Q_HOME, min_jerk, quat_from_mat, rot_axis_angle,
)

PULL_PATIENCE = 1.2  # s a person keeps pulling before giving up (tug-of-war = failure)
PULL_DIST = 0.25  # m the person draws the part back towards themselves
HAND_GRASP_RADIUS = 0.035  # m: part must be this close to the palm centre to be grasped


class GiveEnv(HandoverEnv):
    def __init__(self, seed=0, distractor_prob=0.15, max_time=10.0):
        super().__init__(seed=seed, distractor_prob=distractor_prob, max_time=max_time)
        self.Rt_home = None

    # ------------------------------------------------------------ planning
    def _plan_task(self):
        rng = self.rng
        h = self.half[2]
        # orientation of the part in the gripper: upright, tilted about the finger-closing axis
        tilt = rng.uniform(-np.deg2rad(10), np.deg2rad(10))
        a = rot_axis_angle(np.array([0, 1.0, 0]), tilt) @ np.array([0, 0, 1.0])
        x = np.array([0, 1.0, 0])  # box narrow face normal along the closing direction
        y = np.cross(a, x)
        self.R_obj = np.c_[x, y, a]
        self.grasp_off = (h - 0.025) * a  # gripper holds the top end
        self.low_off = -(h - 0.022) * a  # the person takes the lower end
        start = np.array([rng.uniform(1.0, 1.1), rng.uniform(-0.25, 0.25), rng.uniform(0.08, 0.15)])
        self.t0 = rng.uniform(0.3, 2.0)
        self.waypoints = []
        if not self.distractor:
            recv = np.array([rng.uniform(0.5, 0.72), rng.uniform(-0.3, 0.3), rng.uniform(0.12, 0.45)])
            T = rng.uniform(0.8, 1.5)
            self.waypoints.append((self.t0, T, start, recv))
            self.t_present = self.t0 + T
            if rng.random() < 0.2:  # the person shifts their open hand while waiting
                t1 = self.t_present + rng.uniform(0.8, 2.0)
                delta = rng.normal(size=3)
                delta = delta / np.linalg.norm(delta) * rng.uniform(0.04, 0.10)
                self.waypoints.append((t1, 0.6, recv, recv + delta))
        else:
            self.t_present = np.inf
            if rng.random() < 0.5:  # waves / passes the hand through the workspace
                side = rng.choice([-1, 1])
                p0 = np.array([rng.uniform(0.95, 1.05), side * rng.uniform(0.35, 0.5), rng.uniform(0.1, 0.4)])
                p1 = np.array([rng.uniform(0.55, 0.8), rng.uniform(-0.1, 0.1), rng.uniform(0.15, 0.45)])
                p2 = np.array([rng.uniform(0.95, 1.05), -side * rng.uniform(0.35, 0.5), rng.uniform(0.1, 0.4)])
                T = rng.uniform(0.8, 1.4)
                self.waypoints = [(self.t0, 0.01, start, p0), (self.t0 + 0.01, T, p0, p1),
                                  (self.t0 + 0.01 + T, T, p1, p2)]
            else:  # gestures near their own body
                p, t = start.copy(), self.t0
                for _ in range(4):
                    q = np.array([rng.uniform(0.85, 1.0), rng.uniform(-0.3, 0.3), rng.uniform(0.08, 0.45)])
                    T = rng.uniform(0.5, 1.2)
                    self.waypoints.append((t, T, p, q))
                    t += T + rng.uniform(0.0, 0.6)
                    p = q
        self.hand_start = start
        self.drift_amp = rng.uniform(0.0, 0.012, size=3)
        self.drift_f = rng.uniform(0.15, 0.6, size=3)
        self.drift_ph = rng.uniform(0, 2 * np.pi, size=3)
        self.tremor = rng.uniform(0.0005, 0.002)
        self.reaction = rng.uniform(0.2, 0.4)  # time for the person to close their grip
        self.pull_delay = rng.uniform(0.1, 0.3)
        self.pull_T = rng.uniform(0.8, 1.3)
        self.pull_dir = np.array([1.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.4, 0.1)])
        self.pull_dir /= np.linalg.norm(self.pull_dir)
        self.f_pull_max = rng.uniform(15.0, 35.0)  # N a person exerts before waiting for the robot to let go
        self.obj_in_hand = -self.low_off  # used by _clear_table

    def _hand_pos(self, t):
        if self.human_grasp_t is not None:
            return self.grasp_hand + min_jerk(self.pull_s) * PULL_DIST * self.pull_dir
        return super()._hand_pos(t)

    def _init_task(self):
        m, d = self.m, self.d
        k = self.k
        self.human_grasp_t = None
        self.grasp_hand = None
        self.pull_start_t = None
        self.near_since = None
        self.pull_s = 0.0
        self.yanked = False
        d.mocap_pos[self.hand_mocap] = self._hand_pos(0.0)
        d.mocap_quat[self.hand_mocap] = [1, 0, 0, 0]
        for j in range(3):
            d.eq_active[self.eq_store[j]] = j != k
            d.eq_active[self.eq_hold[j]] = False
        m.eq_solref[self.eq_hold[k]] = [0.03, 1.0]  # compliant human grip
        mujoco.mj_forward(m, d)
        tcp = d.site_xpos[self.tcp].copy()
        self.Rt_home = d.site_xmat[self.tcp].reshape(3, 3).copy()
        oa = self.obj_qadr[k]
        d.qpos[oa:oa + 3] = tcp - self.grasp_off
        d.qpos[oa + 3:oa + 7] = quat_from_mat(self.R_obj)
        half_w = self.half[0]  # cylinder/capsule radius or box narrow half-width
        d.qpos[self.fq] = min(FINGER_OPEN, half_w + 0.001)
        self.finger_tgt = 0.0
        self.grip_cmd = 1.0
        d.ctrl[6:8] = 0.0
        # let the grip settle under gravity (robot already holding the part at t = 0)
        for _ in range(100):
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
        d.time = 0.0
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------- sensing
    def _low_point(self):
        b = self.obj_body[self.k]
        R = self.d.xmat[b].reshape(3, 3)
        c = self.d.xpos[b]
        return c + R @ (self.R_obj.T @ self.low_off), R, c

    def _target(self):
        """TCP position that would put the part's free end in the person's palm."""
        low, R, _ = self._low_point()
        tcp = self.d.site_xpos[self.tcp]
        hand = self.d.mocap_pos[self.hand_mocap]
        return tcp + (hand - low), R

    # ------------------------------------------------------------- dynamics
    def _task_step(self, a, tcp):
        m, d = self.m, self.d
        pad_obj, hand_f, obj_table, obj_robot = self._contacts()
        holding = pad_obj[0] or pad_obj[1]
        low, R, c = self._low_point()
        hand = d.mocap_pos[self.hand_mocap].copy()
        k = self.k

        # --- person closes their hand on the part once it rests in the palm
        if self.human_grasp_t is None and not self.distractor and self.t >= self.t_present - 0.2:
            b = self.obj_body[k]
            near = np.linalg.norm(low - hand) < HAND_GRASP_RADIUS
            if near:
                self.near_since = self.near_since if self.near_since is not None else self.t
                if self.t - self.near_since >= self.reaction:
                    self.human_grasp_t = self.t
                    self.grasp_hand = hand.copy()
                    eq = self.eq_hold[k]
                    m.eq_data[eq, 0:3] = 0.0
                    m.eq_data[eq, 3:6] = c - hand  # weld at the current relative pose
                    m.eq_data[eq, 6:10] = d.xquat[b]
                    m.eq_data[eq, 10] = 1.0
                    d.eq_active[eq] = 1
            else:
                self.near_since = None
        if self.human_grasp_t is not None and self.pull_start_t is None and \
                self.t >= self.human_grasp_t + self.pull_delay:
            self.pull_start_t = self.t
        if self.pull_start_t is not None:
            # force-limited pull: the person stops drawing back while the robot resists too hard
            tension = np.linalg.norm(self._ft_raw()[:3] - self.ft_bias[:3])
            if not (holding and tension > self.f_pull_max):
                self.pull_s = min(1.0, self.pull_s + CTRL_DT / self.pull_T)
        d.mocap_pos[self.hand_mocap] = self._hand_pos(self.t)
        self._perceive()

        tcp_new, dist_home = self._common_metrics(tcp, hand_f, obj_robot)
        mt = self.mt
        success = False
        rew, done, fail = self._safety_terms(a, hand_f)
        presented = self.t >= self.t_present - 0.3
        grasped = self.human_grasp_t is not None
        if self.release_t is None and grasped and not holding:
            self.release_t = self.t
            self.yanked = self.grip_cmd > 0.5  # torn out of a closed gripper, not handed over
        if self.distractor:
            rew -= 2.0 * max(0.0, dist_home - 0.08)
            if not holding:
                rew -= 1.0
            if self.steps >= int(5.0 / CTRL_DT) and not done:
                done = True
                success = holding and mt["max_dist_home"] < 0.15 and not mt["hand_contact"]
                rew += 5.0 if success else 0.0
        else:
            if not grasped:
                if not presented:
                    rew -= 0.5 * max(0.0, dist_home - 0.10)
                else:
                    rew += 1.0 - np.tanh(4 * np.linalg.norm(low - hand))
                if self.grip_cmd < 0.5:
                    rew -= 1.0  # never open before the person has hold of the part
            elif self.pull_start_t is not None:
                rew += -0.5 if holding else 0.5
            if not grasped and (c[2] < 0.03 or np.linalg.norm(c - tcp_new) > 0.2):
                done, fail = True, "drop"
                rew -= 15.0
            if grasped and holding and self.pull_start_t is not None and \
                    self.t - self.pull_start_t > PULL_PATIENCE and not done:
                done, fail = True, "no_release"
                rew -= 10.0
            if self.yanked and not done:
                done, fail = True, "yanked"
                rew -= 10.0
            if grasped and not holding and not done and \
                    np.linalg.norm(d.mocap_pos[self.hand_mocap] - self.grasp_hand) > 0.08:
                done, success = True, True  # the person walked away with the part
                rew += 30.0
        timeout = False
        if not done and self.steps >= self.max_steps:
            done, timeout = True, True
            fail = fail or "timeout"
        self.prev_action = a.copy()
        info = {}
        if done:
            info = self._common_info(success, fail, timeout)
            info.update(
                grasped=grasped,  # the person took hold of the part
                released=self.release_t is not None,
                time_to_grasp=(self.release_t - self.t_present) if (
                    self.release_t is not None and np.isfinite(self.t_present)) else np.nan,
                release_delay=(self.release_t - self.pull_start_t) if (
                    self.release_t is not None and self.pull_start_t is not None) else np.nan,
            )
        return self._obs(), float(rew), done, info
