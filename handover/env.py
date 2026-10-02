"""Human-to-robot handover environment.

Control architecture (as on a real cobot):
  policy (20 Hz)  ->  Cartesian TCP twist + binary gripper command
  safety layer    ->  TCP speed cap (ISO 10218 / TS 15066 speed & separation
                      monitoring), Cartesian accel limit
  kinematics      ->  damped-least-squares differential IK, joint velocity
                      limits, joint position limits
  servo (500 Hz)  ->  joint PD + velocity feed-forward + gravity compensation,
                      torque-saturated at the motor limits (MuJoCo actuators)
The learned policy only ever sees noisy, delayed sensor data (external depth
camera tracking + wrist camera, joint encoders, F/T sensor, finger tactile).
"""

import numpy as np
import mujoco

from .model import (
    build_xml, ARM_JOINTS, JOINT_LIMITS, TORQUE_LIMITS, VEL_LIMITS,
    OBJECT_NAMES, OBJ, ROBOT, TABLE,
)

Q_HOME = np.array([-0.84, -2.264, 2.238, 0.026, 0.731, -1.571])
CTRL_DT = 0.05
N_SUB = 25  # 500 Hz servo
V_MAX = 0.6  # m/s TCP
V_COLLAB = 0.25  # m/s when within SSM distance of the human hand
SSM_DIST = 0.25
W_MAX = 2.0  # rad/s
A_MAX = 4.0  # m/s^2
ALPHA_MAX = 12.0  # rad/s^2
FINGER_OPEN = 0.0425
HAND_FORCE_LIMIT = 140.0  # N, ISO/TS 15066 quasi-static limit for hand/fingers
OBS_DIM = 69
ACT_DIM = 7


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def rot_axis_angle(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class HandoverEnv:
    def __init__(self, seed=0, distractor_prob=0.15, max_time=9.0):
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.distractor_prob = distractor_prob
        self.max_steps = int(max_time / CTRL_DT)
        m = self.m
        self.tcp = m.site("tcp").id
        self.jid = [m.joint(n).id for n in ARM_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.fq = [m.jnt_qposadr[m.joint("finger_l").id], m.jnt_qposadr[m.joint("finger_r").id]]
        self.pad = [m.geom("pad_l").id, m.geom("pad_r").id]
        self.hand_geoms = {m.geom("hand_fist").id, m.geom("hand_forearm").id}
        self.table_geom = m.geom("table").id
        self.hand_mocap = m.body_mocapid[m.body("hand").id]
        self.obj_body = [m.body(n).id for n in OBJECT_NAMES]
        self.obj_geom = [m.geom(n).id for n in OBJECT_NAMES]
        self.obj_qadr = [m.jnt_qposadr[m.joint(n).id] for n in OBJECT_NAMES]
        self.obj_vadr = [m.jnt_dofadr[m.joint(n).id] for n in OBJECT_NAMES]
        self.eq_hold = [m.equality("hold_" + n[4:]).id for n in OBJECT_NAMES]
        self.eq_store = [m.equality("store_" + n[4:]).id for n in OBJECT_NAMES]
        self.robot_bodies = set(range(m.body("base").id, m.body("finger_r").id + 1))
        self.kp = m.actuator_gainprm[:6, 0].copy()
        self.kv = -m.actuator_biasprm[:6, 2].copy()
        self.ft_force_adr = m.sensor_adr[m.sensor("ft_force").id]
        self.ft_torque_adr = m.sensor_adr[m.sensor("ft_torque").id]
        self.touch_adr = [m.sensor_adr[m.sensor("touch_l").id], m.sensor_adr[m.sensor("touch_r").id]]
        self.nominal_grip = m.actuator_forcerange[6, 1]
        # TCP home pose
        self.d.qpos[self.qadr] = Q_HOME
        mujoco.mj_forward(m, self.d)
        self.tcp_home = self.d.site_xpos[self.tcp].copy()
        self._jac = np.zeros((6, m.nv))
        self.force_episode = None  # (is_distractor) override for evaluation

    # ------------------------------------------------------------------ setup
    def _setup_object(self):
        m, rng = self.m, self.rng
        k = rng.choice(3, p=[0.4, 0.35, 0.25])
        self.k = k
        g = self.obj_geom[k]
        if k == 0:  # cylinder (bottle, tool handle, shaft)
            r = rng.uniform(0.012, 0.034)
            hz = rng.uniform(0.065, 0.11)
            size = np.array([r, hz, 0])
            half = np.array([r, r, hz])
        elif k == 1:  # box (electronic component, part, carton)
            hx = rng.uniform(0.012, 0.034)
            hy = rng.uniform(hx, 0.045)
            hz = rng.uniform(0.065, 0.11)
            size = np.array([hx, hy, hz])
            half = size.copy()
        else:  # capsule (rounded handle)
            r = rng.uniform(0.012, 0.03)
            hl = rng.uniform(0.045, 0.08)
            size = np.array([r, hl, 0])
            half = np.array([r, r, hl + r])
        m.geom_size[g] = size
        # geom_rbound / geom_aabb / bvh_aabb keep their compile-time (max-size) values,
        # which conservatively enclose every randomised size.
        mass = rng.uniform(0.05, 0.8)
        b = self.obj_body[k]
        m.body_mass[b] = mass
        # solid-body inertia (box approx for all; adequate for contact dynamics)
        hx, hy, hz = half
        if k == 1:
            m.body_inertia[b] = mass / 3.0 * np.array([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
        else:  # solid cylinder approximation (capsule incl. caps)
            r = half[0]
            m.body_inertia[b] = mass * np.array([r**2 / 4 + hz**2 / 3, r**2 / 4 + hz**2 / 3, r**2 / 2])
        mu = rng.uniform(0.5, 1.0)
        m.geom_friction[g] = [mu, 0.01, 0.001]
        for j in range(3):
            act = j == k
            m.geom_contype[self.obj_geom[j]] = OBJ if act else 0
            m.geom_conaffinity[self.obj_geom[j]] = (ROBOT | TABLE) if act else 0
        self.half = half
        self.mass = mass
        # grip force of the gripper (configurable on 2F-85; randomised)
        f = rng.uniform(30.0, 60.0)
        m.actuator_forcerange[6:8] = [[-f, f], [-f, f]]
        # recompute constraint scaling (invweight0 etc.) for the new inertial properties
        mujoco.mj_setConst(m, self.d)

    def _object_frame(self):
        """Object orientation in the human's hand: mostly upright, random tilt and yaw."""
        rng = self.rng
        tilt = rng.uniform(0, np.deg2rad(30))
        tdir = rng.uniform(0, 2 * np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        R = rot_axis_angle(np.array([np.cos(tdir), np.sin(tdir), 0.0]), tilt) @ rot_axis_angle(
            np.array([0, 0, 1.0]), yaw)
        return R

    def _plan_human(self):
        rng = self.rng
        h = self.half[2]
        R = self.R_obj
        a = R[:, 2]
        # hand (fist) holds the lower end of the object; robot should grasp the upper end
        self.obj_in_hand = (h - 0.022) * a  # object centre relative to fist centre (world-aligned)
        self.grasp_off = (h - 0.025) * a  # grasp point relative to object centre
        start = np.array([rng.uniform(1.0, 1.1), rng.uniform(-0.25, 0.25), rng.uniform(0.02, 0.15)])
        self.t0 = rng.uniform(0.2, 1.5)
        self.T_app = rng.uniform(0.8, 1.6)
        self.waypoints = []  # list of (t_start, duration, from, to)
        if not self.distractor:
            g = np.array([rng.uniform(0.42, 0.68), rng.uniform(-0.3, 0.3), rng.uniform(0.18, 0.55)])
            hand_tgt = g - self.grasp_off - self.obj_in_hand
            self.waypoints.append((self.t0, self.T_app, start, hand_tgt))
            self.t_present = self.t0 + self.T_app
            if rng.random() < 0.2:  # human re-adjusts the offer mid-way
                t1 = self.t_present + rng.uniform(1.0, 2.5)
                delta = rng.normal(size=3)
                delta = delta / np.linalg.norm(delta) * rng.uniform(0.04, 0.10)
                self.waypoints.append((t1, 0.6, hand_tgt, hand_tgt + delta))
        else:
            self.t_present = np.inf
            if rng.random() < 0.5:  # walks the object across the workspace without stopping
                side = rng.choice([-1, 1])
                p0 = np.array([rng.uniform(0.95, 1.05), side * rng.uniform(0.35, 0.5), rng.uniform(0.1, 0.4)])
                p1 = np.array([rng.uniform(0.55, 0.8), rng.uniform(-0.1, 0.1), rng.uniform(0.15, 0.5)])
                p2 = np.array([rng.uniform(0.95, 1.05), -side * rng.uniform(0.35, 0.5), rng.uniform(0.1, 0.4)])
                T = rng.uniform(0.8, 1.4)
                self.waypoints = [(self.t0, 0.01, start, p0), (self.t0 + 0.01, T, p0, p1),
                                  (self.t0 + 0.01 + T, T, p1, p2)]
            else:  # inspects / manipulates the object close to their own body
                p = start.copy()
                t = self.t0
                for _ in range(4):
                    q = np.array([rng.uniform(0.85, 1.0), rng.uniform(-0.3, 0.3), rng.uniform(0.05, 0.45)])
                    T = rng.uniform(0.5, 1.2)
                    self.waypoints.append((t, T, p, q))
                    t += T + rng.uniform(0.0, 0.6)
                    p = q
        self.hand_start = start
        # hold behaviour: slow drift + physiological tremor
        self.drift_amp = rng.uniform(0.0, 0.015, size=3)
        self.drift_f = rng.uniform(0.15, 0.6, size=3)
        self.drift_ph = rng.uniform(0, 2 * np.pi, size=3)
        self.tremor = rng.uniform(0.0005, 0.002)
        self.reaction = rng.uniform(0.15, 0.35)
        self.retreat_T = rng.uniform(0.6, 1.0)

    def _hand_pos(self, t):
        if self.release_t is not None:
            s = min_jerk((t - self.release_t) / self.retreat_T)
            back = self.release_hand + np.array([0.25, 0.0, -0.1])
            return self.release_hand + s * (back - self.release_hand)
        p = self.hand_start.copy()
        for (ts, T, a, b) in self.waypoints:
            if t >= ts:
                p = a + min_jerk((t - ts) / T) * (b - a)
        if t > self.t0:
            p = p + self.drift_amp * np.sin(2 * np.pi * self.drift_f * t + self.drift_ph) \
                - self.drift_amp * np.sin(2 * np.pi * self.drift_f * self.t0 + self.drift_ph)
            p = p + self.tremor * self.rng.normal(size=3)
        return p

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, distractor=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d, rng = self.m, self.d, self.rng
        mujoco.mj_resetData(m, d)
        self.distractor = (rng.random() < self.distractor_prob) if distractor is None else distractor
        self._setup_object()
        self.R_obj = self._object_frame()
        self._plan_human()
        self.release_t = None
        self.release_hand = None
        self.t = 0.0
        self.steps = 0

        q0 = Q_HOME + rng.normal(scale=0.03, size=6)
        d.qpos[self.qadr] = q0
        d.qpos[self.fq] = FINGER_OPEN
        self.q_tgt = q0.copy()
        d.ctrl[:6] = q0
        d.ctrl[6:8] = FINGER_OPEN
        self.finger_tgt = FINGER_OPEN
        self.grip_cmd = 0.0

        # object & hand
        k = self.k
        hp = self._hand_pos(0.0)
        d.mocap_pos[self.hand_mocap] = hp
        d.mocap_quat[self.hand_mocap] = [1, 0, 0, 0]
        for j in range(3):
            d.eq_active[self.eq_store[j]] = j != k
            d.eq_active[self.eq_hold[j]] = j == k
        rel = self.obj_in_hand
        qrel = quat_from_mat(self.R_obj)
        # weld data: anchor(3), relpos(3), relquat(4), torquescale
        self.m.eq_data[self.eq_hold[k], 0:3] = 0.0
        self.m.eq_data[self.eq_hold[k], 3:6] = rel  # object pose in (world-aligned) hand frame
        self.m.eq_data[self.eq_hold[k], 6:10] = qrel
        self.m.eq_data[self.eq_hold[k], 10] = 1.0
        oa = self.obj_qadr[k]
        d.qpos[oa:oa + 3] = hp + rel
        d.qpos[oa + 3:oa + 7] = qrel
        mujoco.mj_forward(m, d)

        # sensors
        self.lat = rng.integers(1, 3)  # camera pipeline latency (control steps)
        self.cam_bias = rng.normal(scale=0.004, size=3)
        self.percept_hist = []
        self.ft_bias = None
        self.prev_action = np.zeros(ACT_DIM)
        self.v_prev = np.zeros(3)
        self.w_prev = np.zeros(3)
        self.vel_hist = []
        self.last_seen = None
        # metrics
        self.mt = dict(grasp_t=None, contact_both_t=None, hold_steps=0, hand_contact=False,
                       max_hand_force=0.0, max_tcp_speed=0.0, max_qd_ratio=0.0, torque_sat=0,
                       jl_hits=0, max_dist_home=0.0, obj_touched=False, path=0.0, sub=0)
        self.ft_bias = self._ft_raw()
        self._perceive(init=True)
        return self._obs()

    # ------------------------------------------------------------- sensing
    def _ft_raw(self):
        d = self.d
        return np.r_[d.sensordata[self.ft_force_adr:self.ft_force_adr + 3],
                     d.sensordata[self.ft_torque_adr:self.ft_torque_adr + 3]]

    def _true_object(self):
        d = self.d
        b = self.obj_body[self.k]
        R = d.xmat[b].reshape(3, 3)
        c = d.xpos[b]
        return c + R @ (self.R_obj.T @ self.grasp_off), R, c

    def _perceive(self, init=False):
        """External depth camera (+ wrist camera when close): noisy, delayed, may drop out."""
        g, R, c = self._true_object()
        hand = self.d.mocap_pos[self.hand_mocap].copy()
        tcp = self.d.site_xpos[self.tcp]
        near = np.linalg.norm(g - tcp) < 0.15
        sig = 0.0015 if near else 0.003
        visible = init or self.rng.random() > 0.03
        noise = self.rng.normal(scale=sig, size=3) + (0 if near else self.cam_bias)
        Rn = R @ rot_axis_angle(self.rng.normal(size=3) + 1e-9, np.deg2rad(self.rng.normal(scale=2.0)))
        rec = dict(g=g + noise, R=Rn, hand=hand + self.rng.normal(scale=0.005, size=3),
                   half=self.half * (1 + self.rng.normal(scale=0.05, size=3)), vis=visible)
        if not visible and self.percept_hist:
            prev = self.percept_hist[-1].copy()
            prev["vis"] = False
            rec = prev
        self.percept_hist.append(rec)
        if len(self.percept_hist) > 12:
            self.percept_hist.pop(0)

    def _percept(self, lag=0):
        i = max(0, len(self.percept_hist) - 1 - self.lat - lag)
        return self.percept_hist[i]

    def _obs(self):
        d = self.d
        q = d.qpos[self.qadr]
        qd = d.qvel[self.vadr]
        tcp = d.site_xpos[self.tcp]
        Rt = d.site_xmat[self.tcp].reshape(3, 3)
        p = self._percept()
        p1 = self._percept(5)
        p2 = self._percept(10)
        rel = p["g"] - tcp
        v5 = (p["g"] - p1["g"]) / (5 * CTRL_DT)
        v10 = (p["g"] - p2["g"]) / (10 * CTRL_DT)
        v1 = (p["g"] - self._percept(1)["g"]) / CTRL_DT
        touch = np.array([d.sensordata[a] for a in self.touch_adr])
        ft = self._ft_raw() - self.ft_bias
        ft = ft + self.rng.normal(scale=[0.3, 0.3, 0.3, 0.02, 0.02, 0.02])
        opening = (d.qpos[self.fq[0]] + d.qpos[self.fq[1]]) / (2 * FINGER_OPEN)
        shape = np.zeros(3)
        shape[self.k] = 1
        o = np.concatenate([
            (q - Q_HOME) / np.pi, qd / np.pi,
            [opening, self.grip_cmd],
            np.clip(touch / 20.0, 0, 2) + self.rng.normal(scale=0.01, size=2),
            (tcp - self.tcp_home) * 3, Rt[:, 1], Rt[:, 2],
            rel * 5, Rt.T @ rel * 5,
            p["R"][:, 2], p["R"][:, 0], shape,
            p["half"] * 20,
            np.clip(v1, -2, 2), [float(p["vis"])],
            (p["hand"] - tcp) * 3,
            np.clip(ft[:3] / 20.0, -3, 3), np.clip(ft[3:] / 2.0, -3, 3),
            self.prev_action,
            np.clip(v5, -2, 2), np.clip(v10, -2, 2),
        ])
        return o.astype(np.float32)

    # ------------------------------------------------------------- control
    def _contacts(self):
        d, m = self.d, self.m
        og = self.obj_geom[self.k]
        pad_obj = [False, False]
        hand_f = 0.0
        obj_table = False
        obj_robot = False
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            pair = {g1, g2}
            if og in pair:
                other = g2 if g1 == og else g1
                if other == self.pad[0]:
                    pad_obj[0] = True
                elif other == self.pad[1]:
                    pad_obj[1] = True
                elif other == self.table_geom:
                    obj_table = True
                if m.geom_bodyid[other] in self.robot_bodies:
                    obj_robot = True
            if (g1 in self.hand_geoms) or (g2 in self.hand_geoms):
                mujoco.mj_contactForce(m, d, i, f6)
                hand_f = max(hand_f, abs(f6[0]))
        return pad_obj, hand_f, obj_table, obj_robot

    def step(self, action):
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        tcp = d.site_xpos[self.tcp].copy()

        # --- safety layer: speed & separation monitoring, accel limits
        v = a[:3] * V_MAX
        w = a[3:6] * W_MAX
        hand_d = np.linalg.norm(self._percept()["hand"] - tcp)
        vcap = V_COLLAB if hand_d < SSM_DIST else V_MAX
        n = np.linalg.norm(v)
        if n > vcap:
            v *= vcap / n
        dv = v - self.v_prev
        lim = A_MAX * CTRL_DT
        if np.linalg.norm(dv) > lim:
            v = self.v_prev + dv * lim / np.linalg.norm(dv)
        dw = w - self.w_prev
        lim = ALPHA_MAX * CTRL_DT
        if np.linalg.norm(dw) > lim:
            w = self.w_prev + dw * lim / np.linalg.norm(dw)
        self.v_prev, self.w_prev = v, w

        # --- differential IK (damped least squares) with joint limits
        mujoco.mj_jacSite(m, d, self._jac[:3], self._jac[3:], self.tcp)
        J = self._jac[:, self.vadr]
        lam = 0.03
        qd = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), np.r_[v, w])
        r = np.max(np.abs(qd) / VEL_LIMITS)
        if r > 1:
            qd /= r
        nxt = np.clip(self.q_tgt + qd * CTRL_DT, JOINT_LIMITS[:, 0] + 0.03, JOINT_LIMITS[:, 1] - 0.03)
        qd = (nxt - self.q_tgt) / CTRL_DT
        q = d.qpos[self.qadr]
        self.q_tgt = np.clip(nxt, q - 0.12, q + 0.12)  # anti wind-up under contact

        # --- gripper (binary command, finite closing speed)
        self.grip_cmd = 1.0 if a[6] > 0 else 0.0
        goal = 0.0 if self.grip_cmd else FINGER_OPEN
        step = 0.5 * 0.15 * CTRL_DT  # per finger, 150 mm/s stroke
        self.finger_tgt += np.clip(goal - self.finger_tgt, -step, step)

        # --- servo loop at 500 Hz (ctrl = position target with velocity feed-forward)
        ff = self.kv / self.kp * qd
        q_from = d.ctrl[:6].copy() - 0  # previous target (incl. ff)
        q_to = self.q_tgt + ff
        f_from = d.ctrl[6]
        for i in range(5):
            s = (i + 1) / 5
            d.ctrl[:6] = q_from + s * (q_to - q_from)
            d.ctrl[6:8] = f_from + s * (self.finger_tgt - f_from)
            mujoco.mj_step(m, d, nstep=N_SUB // 5)
            self.mt["torque_sat"] += int(np.any(np.abs(d.actuator_force[:6]) > 0.98 * TORQUE_LIMITS))
            self.mt["sub"] += 1
        self.t += CTRL_DT
        self.steps += 1

        # --- human
        pad_obj, hand_f, obj_table, obj_robot = self._contacts()
        both = pad_obj[0] and pad_obj[1]
        grip_force = min(abs(d.actuator_force[6]), abs(d.actuator_force[7]))
        gripped = both and grip_force > 5.0
        if gripped:
            if self.mt["contact_both_t"] is None:
                self.mt["contact_both_t"] = self.t
        else:
            self.mt["contact_both_t"] = None
        if (self.release_t is None and gripped and not self.distractor
                and self.t - self.mt["contact_both_t"] >= self.reaction):
            self.release_t = self.t
            self.release_hand = d.mocap_pos[self.hand_mocap].copy()
            d.eq_active[self.eq_hold[self.k]] = 0
            if self.mt["grasp_t"] is None:
                self.mt["grasp_t"] = self.t
        d.mocap_pos[self.hand_mocap] = self._hand_pos(self.t)
        self._perceive()

        # --- metrics
        tcp_new = d.site_xpos[self.tcp]
        sp = np.linalg.norm(tcp_new - tcp) / CTRL_DT
        mt = self.mt
        mt["path"] += np.linalg.norm(tcp_new - tcp)
        mt["max_tcp_speed"] = max(mt["max_tcp_speed"], sp)
        mt["max_qd_ratio"] = max(mt["max_qd_ratio"], float(np.max(np.abs(d.qvel[self.vadr]) / VEL_LIMITS)))
        qn = d.qpos[self.qadr]
        mt["jl_hits"] += int(np.any((qn < JOINT_LIMITS[:, 0] + 0.01) | (qn > JOINT_LIMITS[:, 1] - 0.01)))
        if hand_f > 0:
            mt["hand_contact"] = True
        mt["max_hand_force"] = max(mt["max_hand_force"], hand_f)
        dist_home = np.linalg.norm(tcp_new - self.tcp_home)
        mt["max_dist_home"] = max(mt["max_dist_home"], dist_home)
        mt["obj_touched"] |= obj_robot

        # --- reward / termination
        g, R, c = self._true_object()
        dist = np.linalg.norm(tcp_new - g)
        Rt = d.site_xmat[self.tcp].reshape(3, 3)
        axis = R[:, 2]
        perp = 1 - abs(Rt[:, 2] @ axis)
        if self.k == 1:
            close_ok = max(abs(Rt[:, 1] @ R[:, 0]), abs(Rt[:, 1] @ R[:, 1]))
        else:
            close_ok = 1 - abs(Rt[:, 1] @ axis)
        orient = perp * close_ok
        rew = 0.0
        done = False
        success = False
        fail = None
        da = a - self.prev_action
        rew -= 0.02 * float(da @ da)
        if hand_f > 0:
            rew -= 1.0 + 0.02 * hand_f
        if hand_f > HAND_FORCE_LIMIT:
            done, fail = True, "unsafe_contact"
            rew -= 20
        presented = self.t >= self.t_present - 0.3
        if self.distractor:
            rew -= 2.0 * max(0.0, dist_home - 0.08) + 0.1 * self.grip_cmd
            if obj_robot:
                rew -= 1.0
            if self.steps >= int(5.0 / CTRL_DT):
                done = True
                success = (mt["max_dist_home"] < 0.15) and not mt["obj_touched"] and not mt["hand_contact"]
                rew += 5.0 if success else 0.0
        else:
            if not presented and self.release_t is None:
                rew -= 0.5 * max(0.0, dist_home - 0.10)
            else:
                rew += 1.0 - np.tanh(4 * dist)
                rew += 0.5 * orient * (1.0 if dist < 0.15 else 0.3)
                if self.release_t is None:
                    if dist < 0.025 and orient > 0.8:
                        rew += 0.5 * self.grip_cmd
                    elif dist > 0.06:
                        rew -= 0.3 * self.grip_cmd
            if both:
                rew += 1.5
            if self.release_t is not None:
                held = both and not obj_table and dist < 0.08
                if held:
                    rew += 2.0
                    mt["hold_steps"] += 1
                else:
                    mt["hold_steps"] = 0
                if mt["hold_steps"] >= 20:  # held securely for 1 s after the human let go
                    done, success = True, True
                    rew += 30.0
                if c[2] < 0.03 or np.linalg.norm(c - tcp_new) > 0.2:
                    done, fail = True, "drop"
                    rew -= 15.0
        timeout = False
        if not done and self.steps >= self.max_steps:
            done, timeout = True, True
            fail = fail or "timeout"
        self.prev_action = a.copy()
        info = {}
        if done:
            info = dict(
                success=success, distractor=self.distractor, fail=fail, timeout=timeout,
                obj_type=int(self.k), grasped=self.mt["grasp_t"] is not None,
                released=self.release_t is not None,
                time_to_grasp=(self.mt["grasp_t"] - self.t_present) if (
                    self.mt["grasp_t"] is not None and np.isfinite(self.t_present)) else np.nan,
                episode_time=self.t, hand_contact=mt["hand_contact"],
                max_hand_force=mt["max_hand_force"], max_tcp_speed=mt["max_tcp_speed"],
                max_qd_ratio=mt["max_qd_ratio"], torque_sat_frac=mt["torque_sat"] / max(mt["sub"], 1),
                jl_frac=mt["jl_hits"] / self.steps, false_reach=bool(self.distractor and mt["max_dist_home"] >= 0.15),
                path=mt["path"], mass=self.mass,
            )
        return self._obs(), float(rew), done, info
