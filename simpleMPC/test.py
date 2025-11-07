import numpy as np
from go2_model import Pin_Go2_Model
from mujoco_model import MuJoCo_GO2_Model
import mujoco as mj
import time

pin_go2 = Pin_Go2_Model()
mujoco_go2 = MuJoCo_GO2_Model()

# pin_go2.inverse_kinematics("FL", np.array([0.1934, 0.142, -0.17]))
# pin_go2.inverse_kinematics("FR", np.array([0.1934, -0.142, -0.27]))
# pin_go2.inverse_kinematics("RL", np.array([-0.1934, 0.142, -0.27]))
# pin_go2.inverse_kinematics("RR", np.array([-0.1934, -0.142, -0.27]))

# J1 = pin_go2.computeFootJacobian("FL")
# print(J1)

def pd_controller_static(leg: str, q_des):

    q  = mujoco_go2.get_leg_joint_pos(leg)
    v = mujoco_go2.get_leg_joint_vel(leg)

    Kp = np.array([100.0, 100.0, 100.0])
    Kd = np.array([5.0, 5.0, 5.0])

    # PD torque (positive toward q_des)
    tau_pd = Kp * (q_des - q) - Kd * v

    # Add MuJoCo's bias forces (gravity, Coriolis, etc.) for the leg's DoFs
    # Assumes the wrapper exposes model/data as .model / .data
    m, d = mujoco_go2.model, mujoco_go2.data
    jnames = [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"]
    tau_bias = np.zeros(3)
    for i, jn in enumerate(jnames):
        jid  = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, jn)
        didx = m.jnt_dofadr[jid]               # DoF index for this joint
        tau_bias[i] = d.qfrc_bias[didx]        # MuJoCo bias torque at this DoF
    tau = tau_pd + tau_bias

    return tau

# Set Torque
# Visualize with Mujoco
mujoco_go2.update_with_q_pin(pin_go2.current_config.compute_q())


# Simulation
control_hz = 1000
render_hz  = 60
dt = 1.0 / control_hz
mujoco_go2.model.opt.timestep = 1.0 / control_hz
steps_per_render = max(1, int(control_hz // render_hz))
next_t = time.perf_counter()
torque = np.zeros(12, dtype=float)

q_des = pin_go2.q_init

mujoco_go2.start_viewer()
#mujoco_go2.hold_viewer()
i = 0
print(f"Running simulation at {control_hz} Hz (dt={dt:.6f}s)...")

while mujoco_go2.viewer.is_running():

    torque[0:3] = pd_controller_static("FL", q_des[7:10])
    torque[3:6] = pd_controller_static("FR", q_des[10:13])
    torque[6:9] = pd_controller_static("RL", q_des[13:16])
    torque[9:12] = pd_controller_static("RR", q_des[16:19])

    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_leg_joint_torque("FL", torque[0:3])
    mujoco_go2.set_leg_joint_torque("FR", torque[3:6])
    mujoco_go2.set_leg_joint_torque("RL", torque[6:9])
    mujoco_go2.set_leg_joint_torque("RR", torque[9:12])
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    if (i % steps_per_render) == 0:
        mujoco_go2.viewer.sync()

    next_t += dt
    sleep = next_t - time.perf_counter()
    if sleep > 0:
        time.sleep(sleep)
    else:
        # fell behind, reset clock
        next_t = time.perf_counter()

    i += 1
