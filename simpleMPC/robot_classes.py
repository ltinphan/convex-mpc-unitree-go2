from dataclasses import dataclass
import numpy as np
import pinocchio as pin

@dataclass
class RigidBodyState:
    x_pos: float = 0.0
    y_pos: float = 0.0
    z_pos: float = 0.27
    x_vel: float = 0.0
    y_vel: float = 0.0
    z_vel: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    def compute_x_vec(self):
        x_vec = np.array([
            self.x_pos, self.y_pos, self.z_pos,
            self.roll, self.pitch, self.yaw,
            self.x_vel, self.y_vel, self.z_vel,
            self.roll_rate, self.pitch_rate, self.yaw_rate
        ])
        x_vec = x_vec.reshape(-1, 1)
        return x_vec

@dataclass
class ConfigurationState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.27
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    theta1: float = 0.0
    theta2: float = 0.9
    theta3: float = -1.8
    theta4: float = 0.0
    theta5: float = 0.9
    theta6: float = -1.8
    theta7: float = 0.0
    theta8: float = 0.9
    theta9: float = -1.8
    theta10: float = 0.0
    theta11: float = 0.9
    theta12: float = -1.8

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    dtheta1: float = 0.0
    dtheta2: float = 0.0
    dtheta3: float = 0.0
    dtheta4: float = 0.0
    dtheta5: float = 0.0
    dtheta6: float = 0.0
    dtheta7: float = 0.0
    dtheta8: float = 0.0
    dtheta9: float = 0.0
    dtheta10: float = 0.0
    dtheta11: float = 0.0
    dtheta12: float = 0.0


    
    def compute_q(self):
        q = np.array([self.x, self.y, self.z,                      
                    self.qx, self.qy, self.qz, self.qw,                 
                    self.theta1, self.theta2, self.theta3,
                    self.theta4, self.theta5, self.theta6,
                    self.theta7, self.theta8, self.theta9,
                    self.theta10, self.theta11, self.theta12], dtype=float) 
        return q
    
    def compute_v(self):
        v = np.array([self.dx, self.dy, self.dz,                      
                    self.wx, self.wy, self.wz,                 
                    self.dtheta1, self.dtheta2, self.dtheta3,
                    self.dtheta4, self.dtheta5, self.dtheta6,
                    self.dtheta7, self.dtheta8, self.dtheta9,
                    self.dtheta10, self.dtheta11, self.dtheta12], dtype=float) 
        return v
    
    def get_euler_angle(self):

        q_eig = pin.Quaternion(self.qw, self.qx, self.qy, self.qz)
        R = q_eig.toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(R)

        return np.array(rpy) 

    def update_config(self, q):
        self.x = q[0]
        self.y = q[1]
        self.z = q[2]
        self.qx = q[3]
        self.qy = q[4]
        self.qz = q[5]
        self.qw = q[6]
        self.theta1 = q[7]
        self.theta2 = q[8]
        self.theta3 = q[9]
        self.theta4 = q[10]
        self.theta5 = q[11]
        self.theta6 = q[12]
        self.theta7 = q[13]
        self.theta8 = q[14]
        self.theta9 = q[15]
        self.theta10 = q[16]
        self.theta11 = q[17]
        self.theta12 = q[18]

    def update_dconfig(self, v):
        self.dx = v[0]
        self.dy = v[1]
        self.dz = v[2]
        self.wx = v[3]
        self.wy = v[4]
        self.wz = v[5]
        self.dtheta1 = v[6]
        self.dtheta2 = v[7]
        self.dtheta3 = v[8]
        self.dtheta4 = v[9]
        self.dtheta5 = v[10]
        self.dtheta6 = v[11]
        self.dtheta7 = v[12]
        self.dtheta8 = v[13]
        self.dtheta9 = v[14]
        self.dtheta10 = v[15]
        self.dtheta11 = v[16]
        self.dtheta12 = v[17]
    
    def get_simplified_full_state(self):
        current_state = RigidBodyState()

        current_state.x_pos = self.x
        current_state.y_pos = self.y
        current_state.z_pos = self.z

        rpy = self.get_euler_angle()
        current_state.roll = rpy[0]
        current_state.pitch = rpy[1]
        current_state.yaw = rpy[2]

        current_state.x_vel= self.dx
        current_state.y_vel = self.dy
        current_state.z_vel = self.dz

        current_state.roll_rate= self.wx
        current_state.pitch_rate = self.wy
        current_state.yaw_rate = self.wz

        return current_state

         