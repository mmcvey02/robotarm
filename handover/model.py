"""MuJoCo model of the handover work cell.

Robot: 6-DOF collaborative arm with UR5e-class kinematics, link masses,
inertias, joint torque limits (150 Nm base/shoulder/elbow, 28 Nm wrist),
rotor armature and joint friction. Mounted on a table top (z = 0).

End effector: 6-axis force/torque sensor + Robotiq 2F-85-class parallel-jaw
gripper (85 mm stroke, force limited, finite closing speed) with tactile
(touch) sensing on both finger pads and a wrist RGB-D camera.

Human: the hand is a kinematic (mocap) body that carries the offered object
through a weld constraint; the human "lets go" by deactivating the weld once
the robot's grip is established. The torso is a non-colliding visual.

Objects: three free bodies (cylinder, box, capsule). Only one is active per
episode; size, mass and friction are randomised at runtime. NOTE: the XML sizes
are the *maximum* randomised sizes, because MuJoCo's mid-phase bounding volumes
are computed at compile time and must enclose every runtime size.
"""

import numpy as np

# collision bit masks
ROBOT, OBJ, HAND, TABLE = 1, 2, 4, 8

ARM_JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
]
# position limits (rad) -- tightened from the UR5e +-2pi to prevent cable wrap
# and elbow self-collision, as is common in cobot safety configurations.
JOINT_LIMITS = np.array([
    [-np.pi, np.pi],
    [-np.pi, 0.0],
    [-2.8, 2.8],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
])
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])  # Nm (UR5e)
VEL_LIMITS = np.array([np.pi] * 6)  # rad/s (UR5e: 180 deg/s)
GRIPPER_STROKE = 0.085  # m (full opening)
GRIPPER_SPEED = 0.15  # m/s stroke rate (2F-85: 20-150 mm/s)
OBJECT_NAMES = ["obj_cyl", "obj_box", "obj_cap"]


def build_xml() -> str:
    return f"""
<mujoco model="handover_cell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"
          noslip_iterations="0"/>
  <size memory="16M"/>

  <visual>
    <global offwidth="960" offheight="720" azimuth="135" elevation="-20"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".55 .55 .58" rgb2=".5 .5 .53"
             width="512" height="512"/>
    <material name="table" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="arm" rgba="0.75 0.78 0.82 1"/>
    <material name="joint" rgba="0.25 0.45 0.75 1"/>
    <material name="grip" rgba="0.15 0.15 0.17 1"/>
    <material name="pad" rgba="0.9 0.4 0.1 1"/>
    <material name="skin" rgba="0.87 0.68 0.55 1"/>
    <material name="shirt" rgba="0.3 0.5 0.35 0.5"/>
    <material name="obj" rgba="0.95 0.8 0.1 1"/>
  </asset>

  <default>
    <joint armature="0.1" damping="2" frictionloss="1"/>
    <geom contype="{ROBOT}" conaffinity="{ROBOT|OBJ|HAND|TABLE}" condim="3" friction="0.8 0.01 0.001"
          solref="0.01 1"/>
    <default class="big">
      <joint actuatorgravcomp="true"/>
      <general gaintype="fixed" biastype="affine" gainprm="3000" biasprm="0 -3000 -150"
               forcerange="-150 150" ctrlrange="-6.3 6.3"/>
    </default>
    <default class="small">
      <joint actuatorgravcomp="true" armature="0.05" damping="0.5" frictionloss="0.3"/>
      <general gaintype="fixed" biastype="affine" gainprm="600" biasprm="0 -600 -20"
               forcerange="-28 28" ctrlrange="-6.3 6.3"/>
    </default>
    <default class="finger">
      <!-- armature = reflected inertia of the non-backdrivable finger drive (motor + worm gear) -->
      <joint type="slide" range="0 0.0425" armature="0.5" damping="10" frictionloss="0.5"/>
      <general gaintype="fixed" biastype="affine" gainprm="6000" biasprm="0 -6000 -100"
               forcerange="-40 40" ctrlrange="0 0.0425"/>
    </default>
  </default>

  <worldbody>
    <light pos="0.5 -1 2.5" dir="0 0.3 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="1.5 1 2" dir="-0.4 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="table" type="box" size="1.0 0.8 0.02" pos="0.5 0 -0.02" material="table"
          contype="{TABLE}" conaffinity="{ROBOT|OBJ}"/>
    <camera name="overview" pos="0.55 -1.45 0.85" xyaxes="1 0 0 0 0.45 0.89" fovy="50"/>
    <camera name="side" pos="1.6 -1.1 1.0" xyaxes="0.6 0.8 0 -0.35 0.26 0.9" fovy="45"/>
    <!-- external stereo depth camera used for human / object tracking -->
    <camera name="scene_depth" pos="-0.25 0 1.1" xyaxes="0 -1 0 0.8 0 0.6" fovy="70"/>

    <!-- ===================== robot ===================== -->
    <body name="base" pos="0 0 0">
      <inertial mass="4.0" pos="0 0 0.05" diaginertia="0.0044 0.0044 0.0072"/>
      <geom type="cylinder" size="0.075 0.045" pos="0 0 0.05" material="arm"
            contype="0" conaffinity="0"/>
      <body name="shoulder_link" pos="0 0 0.163" gravcomp="1">
        <inertial mass="3.7" pos="0 0 0" diaginertia="0.0103 0.0103 0.0067"/>
        <joint name="shoulder_pan" class="big" axis="0 0 1" range="{JOINT_LIMITS[0,0]} {JOINT_LIMITS[0,1]}"/>
        <geom type="cylinder" size="0.06 0.07" material="joint" contype="0" conaffinity="0"/>
        <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0" gravcomp="1">
          <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.1339 0.1339 0.0151"/>
          <joint name="shoulder_lift" class="big" axis="0 1 0" range="{JOINT_LIMITS[1,0]} {JOINT_LIMITS[1,1]}"/>
          <geom type="cylinder" size="0.06 0.065" quat="1 1 0 0" material="joint"/>
          <geom type="capsule" fromto="0 0 0.06 0 0 0.40" size="0.054" material="arm"/>
          <body name="forearm_link" pos="0 -0.131 0.425" gravcomp="1">
            <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0312 0.0312 0.0041"/>
            <joint name="elbow" class="big" axis="0 1 0" range="{JOINT_LIMITS[2,0]} {JOINT_LIMITS[2,1]}"/>
            <geom type="cylinder" size="0.05 0.06" quat="1 1 0 0" material="joint"/>
            <geom type="capsule" fromto="0 0 0.07 0 0 0.37" size="0.04" material="arm"/>
            <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0" gravcomp="1">
              <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0026 0.0026 0.0022"/>
              <joint name="wrist_1" class="small" axis="0 1 0" range="{JOINT_LIMITS[3,0]} {JOINT_LIMITS[3,1]}"/>
              <geom type="cylinder" size="0.04 0.05" pos="0 0.05 0" quat="1 1 0 0" material="joint"
                    contype="0" conaffinity="0"/>
              <body name="wrist_2_link" pos="0 0.127 0" gravcomp="1">
                <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0026 0.0026 0.0022"/>
                <joint name="wrist_2" class="small" axis="0 0 1" range="{JOINT_LIMITS[4,0]} {JOINT_LIMITS[4,1]}"/>
                <geom type="cylinder" size="0.04 0.05" pos="0 0 0.05" material="joint"/>
                <body name="wrist_3_link" pos="0 0 0.1" gravcomp="1">
                  <inertial mass="0.1889" pos="0 0.0771683 0" quat="1 0 0 1"
                            diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
                  <joint name="wrist_3" class="small" axis="0 1 0" range="{JOINT_LIMITS[5,0]} {JOINT_LIMITS[5,1]}"/>
                  <geom type="cylinder" size="0.04 0.04" pos="0 0.06 0" quat="1 1 0 0" material="joint"
                        contype="0" conaffinity="0"/>
                  <!-- tool flange: z axis = approach direction, y axis = finger closing direction -->
                  <body name="tool" pos="0 0.1 0" quat="-1 1 0 0" gravcomp="1">
                    <site name="ft_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
                    <!-- F/T sensor (0.3 kg) + gripper base & coupling (0.6 kg) -->
                    <inertial mass="0.9" pos="0 0 0.05" diaginertia="0.0012 0.0012 0.0009"/>
                    <geom name="ft_body" type="cylinder" size="0.04 0.0175" pos="0 0 0.0175" material="grip"/>
                    <geom name="palm" type="box" size="0.03 0.05 0.03" pos="0 0 0.065" material="grip"/>
                    <camera name="wrist_cam" pos="0.055 0 0.05" xyaxes="0 -1 0 0.97 0 0.26" fovy="87"/>
                    <geom type="box" size="0.012 0.02 0.012" pos="0.045 0 0.05" rgba="0.1 0.1 0.1 1"
                          contype="0" conaffinity="0"/>
                    <site name="tcp" pos="0 0 0.14" size="0.006" rgba="0 1 0 0.6"/>
                    <body name="finger_l" pos="0 0 0" gravcomp="1">
                      <inertial mass="0.06" pos="0 0.02 0.11" diaginertia="2e-5 2e-5 1e-5"/>
                      <joint name="finger_l" class="finger" axis="0 1 0"/>
                      <geom type="box" size="0.011 0.006 0.025" pos="0 0.012 0.115" material="grip"/>
                      <geom name="pad_l" type="box" size="0.011 0.004 0.022" pos="0 0.004 0.14"
                            material="pad" condim="4" friction="1.0 0.02 0.002" priority="1"
                            solref="0.005 1" solimp="0.95 0.99 0.001"/>
                      <site name="pad_l_site" type="box" size="0.012 0.006 0.023" pos="0 0.004 0.14"
                            rgba="0 0 0 0"/>
                    </body>
                    <body name="finger_r" pos="0 0 0" gravcomp="1">
                      <inertial mass="0.06" pos="0 -0.02 0.11" diaginertia="2e-5 2e-5 1e-5"/>
                      <joint name="finger_r" class="finger" axis="0 -1 0"/>
                      <geom type="box" size="0.011 0.006 0.025" pos="0 -0.012 0.115" material="grip"/>
                      <geom name="pad_r" type="box" size="0.011 0.004 0.022" pos="0 -0.004 0.14"
                            material="pad" condim="4" friction="1.0 0.02 0.002" priority="1"
                            solref="0.005 1" solimp="0.95 0.99 0.001"/>
                      <site name="pad_r_site" type="box" size="0.012 0.006 0.023" pos="0 -0.004 0.14"
                            rgba="0 0 0 0"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ===================== human ===================== -->
    <body name="torso" pos="1.32 0 0.25">
      <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.17" material="shirt" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 0.85" size="0.11" material="skin" contype="0" conaffinity="0"/>
    </body>
    <body name="hand" mocap="true" pos="1.0 0 0.1">
      <geom name="hand_fist" type="ellipsoid" size="0.045 0.05 0.04" material="skin"
            contype="{HAND}" conaffinity="{ROBOT}"/>
      <geom name="hand_forearm" type="capsule" fromto="0.03 0 -0.01 0.30 0 -0.12" size="0.035"
            material="skin" contype="{HAND}" conaffinity="{ROBOT}"/>
    </body>
    <body name="storage" mocap="true" pos="0 1.5 -1"/>

    <!-- ===================== objects ===================== -->
    <body name="obj_cyl" pos="0 1.5 -1">
      <freejoint name="obj_cyl"/>
      <geom name="obj_cyl" type="cylinder" size="0.034 0.11" mass="0.3" material="obj" condim="4"
            contype="{OBJ}" conaffinity="{ROBOT|TABLE}" friction="0.8 0.02 0.002"/>
    </body>
    <body name="obj_box" pos="0.2 1.5 -1">
      <freejoint name="obj_box"/>
      <geom name="obj_box" type="box" size="0.034 0.045 0.11" mass="0.3" material="obj" condim="4"
            contype="{OBJ}" conaffinity="{ROBOT|TABLE}" friction="0.8 0.02 0.002"/>
    </body>
    <body name="obj_cap" pos="0.4 1.5 -1">
      <freejoint name="obj_cap"/>
      <geom name="obj_cap" type="capsule" size="0.03 0.08" mass="0.3" material="obj" condim="4"
            contype="{OBJ}" conaffinity="{ROBOT|TABLE}" friction="0.8 0.02 0.002"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="base" body2="shoulder_link"/>
    <exclude body1="base" body2="upper_arm_link"/>
    <exclude body1="upper_arm_link" body2="wrist_1_link"/>
    <exclude body1="finger_l" body2="finger_r"/>
  </contact>

  <equality>
    <weld name="hold_cyl" body1="hand" body2="obj_cyl" active="false" solref="0.02 1"/>
    <weld name="hold_box" body1="hand" body2="obj_box" active="false" solref="0.02 1"/>
    <weld name="hold_cap" body1="hand" body2="obj_cap" active="false" solref="0.02 1"/>
    <weld name="store_cyl" body1="storage" body2="obj_cyl" relpose="0 0 0 1 0 0 0" active="true"/>
    <weld name="store_box" body1="storage" body2="obj_box" relpose="0.2 0 0 1 0 0 0" active="true"/>
    <weld name="store_cap" body1="storage" body2="obj_cap" relpose="0.4 0 0 1 0 0 0" active="true"/>
  </equality>

  <actuator>
    <general name="a_shoulder_pan" joint="shoulder_pan" class="big"/>
    <general name="a_shoulder_lift" joint="shoulder_lift" class="big"/>
    <general name="a_elbow" joint="elbow" class="big"/>
    <general name="a_wrist_1" joint="wrist_1" class="small"/>
    <general name="a_wrist_2" joint="wrist_2" class="small"/>
    <general name="a_wrist_3" joint="wrist_3" class="small"/>
    <general name="a_finger_l" joint="finger_l" class="finger"/>
    <general name="a_finger_r" joint="finger_r" class="finger"/>
  </actuator>

  <sensor>
    <force name="ft_force" site="ft_site"/>
    <torque name="ft_torque" site="ft_site"/>
    <touch name="touch_l" site="pad_l_site"/>
    <touch name="touch_r" site="pad_r_site"/>
  </sensor>
</mujoco>
"""
