import mujoco
import numpy as np


class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """

        self.m = m
        self.d = d

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        """
        :param pos: numpy array of joint positions in range [-pi, pi]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return (pos / 3.14159 + 1.) * 2048

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        """
        :param pwm: numpy array of pwm values in range [0, 4096]
        :return: numpy array of joint positions in range [-pi, pi]
        """
        return (pwm / 2048 - 1) * 3.14159265359

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of pwm values in range [0, 4096]
        :return: numpy array of values in range [0, 1]
        """
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of values in range [0, 1]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return x * 4096

    def read_position(self) -> np.ndarray:
        """
        :return: numpy array of current joint positions in range [0, 4096]
        """
        return self.d.qpos[:6]  # 5-> 6

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def read_ee_pos(self, joint_name='end_effector'):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        joint_id = self.m.body(joint_name).id
        return self.d.geom_xpos[joint_id]

    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos

    def inverse_kinematics(self, current_position, ee_target_pos, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]

        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)

        # compute target joint velocities
        qpos = current_position[:6]
        # 5->6 due to increased njoints
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos)

        # apply the joint velocities
        q_target_pos = qpos + qdot * rate
        return q_target_pos

    def inverse_kinematics_rot(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(6)
        integration_dt = 1.0

        # compute the jacobian
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)

        # compute target joint velocities
        jac = np.vstack([jacp, jacr])

        # Orientation error.

        mujoco.mju_mat2Quat(site_quat, ee_rot)
        # print("size test:", site_quat,site_quat.shape,ee_rot, ee_rot.shape)  
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)

        mujoco.mju_negQuat(site_quat_conj, site_quat)

        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)

        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)
        # print("type testing:", ee_pos, ee_target_pos,error_rot)
        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])

        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        # print("dq:",dq)
        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        # Set the control signal.
        np.clip(q[:6], *self.m.jnt_range.T[:, :6], out=q[:6])
        self.d.ctrl[:6] = q[:6]
        # Step the simulation.
        mujoco.mj_step(self.m, self.d)
        return q[:6]
    
    def forward_kinematics(self, qpos):
        """
        Computes the end-effector position given joint positions.
        :param qpos: Joint positions (numpy array)
        :return: End-effector position (numpy array)
        """
        # Update the simulation with the current joint positions
        self.d.qpos[:len(qpos)] = qpos
        mujoco.mj_fwdPosition(self.m, self.d)

        # Get the end-effector position
        ee_body_name = 'joint6'  # Replace with your end-effector link name
        ee_body_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_XBODY, ee_body_name)
        ee_pos = self.d.xpos[ee_body_id].copy()

        return ee_pos
