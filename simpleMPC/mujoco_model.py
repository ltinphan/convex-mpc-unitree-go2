import mujoco as mj
import mujoco.viewer
import numpy as np
from pathlib import Path
import time
from go2_robot_data import PinGo2Model
import mujoco.viewer as mjv

class MuJoCo_GO2_Model:
    def __init__(self):
        # Set this to your MJCF file directory
        repo = Path(__file__).resolve().parents[1]
        scene_path = repo / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml"

        # Load the MuJoCo model
        self.model = mj.MjModel.from_xml_path(str(scene_path))
        self.data = mj.MjData(self.model)
        self.viewer = None

        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    def update_with_q_pin(self, q_pin):
        px, py, pz, qx, qy, qz, qw, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 = q_pin[:]
        self.data.qpos[:] = [px, py, pz, qw, qx, qy, qz, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]
        # self.data.qpos[:3] = q_pin[:3]
        # self.data.qpos[6:] = q_pin[6:]

        mj.mj_forward(self.model, self.data)   # recompute derived quantities

    def start_viewer(self):
        self.viewer = mj.viewer.launch_passive(self.model, self.data).__enter__()
        self.viewer.sync()

    def hold_viewer(self):
        while self.viewer.is_running():
            #mujoco.mj_step(self.model, self.data) #Comment this line for passive viewer
            self.viewer.sync()

    def set_leg_joint_torque(self, leg: str, torque):

        aid_hip = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip")
        aid_thigh = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_thigh")
        aid_calf = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_calf")

        self.data.ctrl[aid_hip] = torque[0]
        self.data.ctrl[aid_thigh] = torque[1]
        self.data.ctrl[aid_calf] = torque[2]

    def replay_simulation(self, time_log_s, q_log, tau_log_Nm, RENDER_DT, REALTIME_FACTOR):
        model = self.model
        data_replay = mj.MjData(model)

        with mjv.launch_passive(model, data_replay) as viewer:

            # 1) Pick the body to track (change "base" to your torso/body name)
            base_id = model.body("base_link").id   # or e.g. model.body("torso").id

            # 2) Configure camera as a tracking camera
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
            viewer.cam.fixedcamid = -1       # not using a fixed camera slot

            # Optional: nice initial view
            viewer.cam.distance = 2.0        # how far from the body
            viewer.cam.elevation = -20       # vertical angle (deg)
            viewer.cam.azimuth = 90          # horizontal angle (deg)

            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True


            while viewer.is_running():           # loop until the window is closed

                start_wall = time.perf_counter()
                t0 = time_log_s[0]
                next_render_t = t0

                k = 0
                T = len(time_log_s)

                # One full replay
                while k < T and viewer.is_running():
                    t = time_log_s[k]

                    # time to render a frame?
                    if t >= next_render_t:
                        data_replay.qpos[:] = q_log[k]
                        data_replay.ctrl[:] = tau_log_Nm[k]
                        mj.mj_forward(model, data_replay)
                        viewer.sync()

                        # real-time pacing
                        target_wall = start_wall + (t - t0) / REALTIME_FACTOR
                        now = time.perf_counter()
                        sleep_time = target_wall - now# 2) Compute the dynamics and desir ed trajectory

                        if sleep_time > 0:
                            time.sleep(sleep_time)

                        next_render_t += RENDER_DT

                    k += 1

                time.sleep(1)

    def set_joint_torque(self, torque):

        self.set_leg_joint_torque("FL", torque[0:3])
        self.set_leg_joint_torque("FR", torque[3:6])
        self.set_leg_joint_torque("RL", torque[6:9])
        self.set_leg_joint_torque("RR", torque[9:12])

    def get_leg_joint_pos(self, leg: str):

        q_leg = np.zeros(3, dtype=float)

        jid_hip = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
        jid_thigh = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
        jid_calf = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_calf_joint")

        q_idx_hip = self.model.jnt_qposadr[jid_hip]
        q_idx_thigh = self.model.jnt_qposadr[jid_thigh]
        q_idx_calf = self.model.jnt_qposadr[jid_calf]

        q_leg[0] = float(self.data.qpos[q_idx_hip])
        q_leg[1] = float(self.data.qpos[q_idx_thigh])
        q_leg[2] = float(self.data.qpos[q_idx_calf])

        return q_leg
    
    def get_leg_joint_vel(self, leg: str):

        v_leg = np.zeros(3, dtype=float)

        jid_hip = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
        jid_thigh = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")
        jid_calf = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_calf_joint")

        v_idx_hip = self.model.jnt_dofadr[jid_hip]
        v_idx_thigh = self.model.jnt_dofadr[jid_thigh]
        v_idx_calf = self.model.jnt_dofadr[jid_calf]

        v_leg[0] = float(self.data.qpos[v_idx_hip])
        v_leg[1] = float(self.data.qpos[v_idx_thigh])
        v_leg[2] = float(self.data.qpos[v_idx_calf])

        return v_leg
    
    def update_pin_with_mujoco(self, go2:PinGo2Model):

        mujoco_q  = np.asarray(self.data.qpos, dtype=float).reshape(-1)   # (19,)
        mujoco_dq = np.asarray(self.data.qvel, dtype=float).reshape(-1)   # (18,)
        qw, qx, qy, qz = mujoco_q[3:7]

        # Convert to Pin
        q  = np.concatenate([mujoco_q[0:3], [qx, qy, qz, qw], mujoco_q[7:]])
        dq = mujoco_dq

        go2.update_model(q, dq)











