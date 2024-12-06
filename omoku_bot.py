import time
# import keyboard

import numpy as np
import mujoco
import mujoco.viewer
from interface import SimulatedRobot
from robot import Robot, OperatingMode
from scipy.spatial.transform import Rotation as R

# Default positions for illustrative purposes
HOME_POSITION = [0.0, 0.16, 0.14]          # Example "home" hover position
TAKE_PICK_POSITION = [-0.08, 0.085, 0.14]  # Example pick position
DEFAULT_UP_Z = 0.14                        # "Safe" hover height
DEFAULT_DOWN_Z = 0.07                      # Lower position for grasp/release


class OmokuBot:
    def __init__(self):
        """
        Initialize the OmokuBot class, set up the robot, and define workspace boundaries.
        """
        self.robot = None
        self.r = None
        self.d = None
        self.m = None
        self.viewer = None
        self.positions = None

        # Robot setup
        self.robot_setup()
        self.positions = self.robot.read_position()

        # Additional parameters
        self.telescope_rate = 1.5e-2
        self.tolerance = 1e-5
        self.x_max_distance = 0.07
        self.x_min_distance = -0.07
        self.y_max_distance = 0.24
        self.y_min_distance = 0.10

        print("ROBOT SETUP DONE")

    def robot_setup(self):
        """
        Set up the Mujoco model and SimulatedRobot interface, and connect to the real robot.
        """
        self.m = mujoco.MjModel.from_xml_path(
            'omoku_bot/low_cost_robot/low_cost_robot.xml')
        self.d = mujoco.MjData(self.m)
        self.r = SimulatedRobot(self.m, self.d)
        self.robot = Robot(device_name='/dev/ttyACM0')

    def init_robot(self):
        """
        Initialize the robot viewer and move to a default position.
        """
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        # Ensure gripper is open at start
        self.release()
        self.move_ee_position(TAKE_PICK_POSITION)

    def move_joint(self, positions):
        """
        Move the robot joints to the given PWM positions with interpolation.
        """
        pwm_values = np.array(positions)
        current_position = self.robot.read_position()
        interpolated_positions = self.robot.get_interpolate_pose(
            current_position, pwm_values
        )

        for pos in interpolated_positions:
            pos = np.array(pos)
            self.robot.set_goal_pos(pos)
            current_qpos = self.r._pwm2pos(pos)
            self.d.qpos[:6] = current_qpos[:6]
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            time.sleep(0.01)

    def move_ee_position(self, destination):
        """
        Move the end-effector (EE) to the specified 3D position using inverse kinematics.
        """
        destination = np.array(destination)

        # Define target orientation (fixed orientation in this example)
        target_ee_rot = R.from_euler(
            'z', 0, degrees=True).as_matrix().flatten()

        current_ee_pos = self.d.xpos[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, 'joint6')]

        # Interpolate between current_ee_pos and destination
        num_waypoints = 100
        waypoints = np.linspace(current_ee_pos, destination, num_waypoints)

        for waypoint in waypoints:
            max_iterations = 50000
            tolerance = 5e-2

            for _ in range(max_iterations):
                qpos_ik = self.r.inverse_kinematics_rot(
                    waypoint, target_ee_rot, rate=0.05, joint_name='joint6'
                )
                self.d.qpos[:5] = qpos_ik[:5]
                mujoco.mj_forward(self.m, self.d)

                ee_pos = self.d.xpos[mujoco.mj_name2id(
                    self.m, mujoco.mjtObj.mjOBJ_BODY, 'joint6')]
                error = np.linalg.norm(waypoint - ee_pos)
                if error < tolerance:
                    break
            else:
                print(f"IK did not converge for waypoint {waypoint}")
                continue

            # Convert joint positions to PWM values
            pwm_values = self.r._pos2pwm(qpos_ik[:5]).astype(int)

            # Keep current gripper position
            current_gripper_pwm = self.positions[5]
            full_pwm_values = np.concatenate(
                (pwm_values, [current_gripper_pwm]))

            # Send command to the robot
            self.robot.set_goal_pos(full_pwm_values)
            time.sleep(0.02)

            # Update simulation with actual robot position
            current_position = self.robot.read_position()
            converted_positions = self.r._pwm2pos(np.array(current_position))
            self.d.qpos[:6] = converted_positions[:6]
            mujoco.mj_forward(self.m, self.d)

            if self.viewer is not None:
                self.viewer.sync()

    def gripper(self, mode):
        """
        Control the gripper.
        mode: "open" or "close"
        """
        if mode == "open":
            self.positions[5] = 2200
        elif mode == "close":
            self.positions[5] = 1965

        self._move_gripper()

    def _move_gripper(self):
        """
        Internal method to move the gripper to the currently set positions[5].
        """
        pwm_values = np.array(self.positions)
        current_position = self.robot.read_position()
        interpolated_positions = self.robot.get_interpolate_pose(
            current_position, pwm_values
        )

        for pos in interpolated_positions:
            pos = np.array(pos)
            self.robot.set_goal_pos(pos)
            current_qpos = self.r._pwm2pos(pos)
            self.d.qpos[:6] = current_qpos[:6]
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            time.sleep(0.01)

    def get_ee_xyz(self):
        """
        Get the current end-effector XYZ position from the real robot.
        """
        positions = np.array(self.robot.read_position())
        current_qpos = self.r._pwm2pos(positions)
        current_ee_xyz = self.r.forward_kinematics(current_qpos)
        return current_ee_xyz

    def get_joint_rad(self):
        """
        Get the current joint angles (in radians).
        """
        positions = np.array(self.robot.read_position())
        current_qpos = self.r._pwm2pos(positions)
        return current_qpos

    def set_workspace(self, x_min, x_max, y_min, y_max):
        """
        Set workspace boundaries.
        """
        self.x_min_distance = x_min
        self.x_max_distance = x_max
        self.y_min_distance = y_min
        self.y_max_distance = y_max

    def move_to_grid(self, grid_x, grid_y, z_plane=0.08):
        """
        Move the robot to a specific grid position on a 9x9 grid.
        grid_x, grid_y: integer coordinates between 0 and 8
        z_plane: target z height
        """
        if not (0 <= grid_x < 9 and 0 <= grid_y < 9):
            raise ValueError("Grid coordinates must be between 0 and 8")

        # Define workspace bounds
        x_min, x_max = self.x_min_distance, self.x_max_distance
        y_min, y_max = self.y_min_distance, self.y_max_distance

        # Calculate the grid position
        x_position = x_min + (x_max - x_min) * (grid_x / 8)
        y_position = y_min + (y_max - y_min) * (grid_y / 8)
        destination = np.array([x_position, y_position, z_plane])

        self.move_ee_position(destination)

    def grasp(self):
        """
        Perform a grasp:
        1. Move down from current Z=0.14 to Z=0.07
        2. Close gripper
        3. Move up back to Z=0.14
        """
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        # Move down
        self.move_ee_position([x, y, DEFAULT_DOWN_Z])
        # Close gripper
        self.gripper("close")
        # Move up
        self.move_ee_position([x, y, DEFAULT_UP_Z])

    def release(self):
        """
        Perform a release:
        1. Move down from current Z=0.14 to Z=0.07
        2. Open gripper
        3. Move up back to Z=0.14
        """
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        # Move down
        self.move_ee_position([x, y, DEFAULT_DOWN_Z])
        # Open gripper
        self.gripper("open")
        # Move up
        self.move_ee_position([x, y, DEFAULT_UP_Z])

    def move_up(self):
        """
        Move the robot end-effector straight up to Z=0.14 from current position.
        """
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position([x, y, DEFAULT_UP_Z])

    def move_down(self):
        """
        Move the robot end-effector straight down to Z=0.07 from current position.
        """
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position([x, y, DEFAULT_DOWN_Z])

    def _pwm2pos(self, positions):
        # Placeholder if needed for custom PWM-to-position logic
        pass


# Example usage (uncomment if you want to run this standalone):
# if __name__ == "__main__":
#     robot = OmokuBot()
#     robot.init_robot()
#     robot.set_workspace(-0.07, 0.07, 0.10, 0.24)
#     robot.move_to_grid(4, 4)
#     robot.grasp()
#     robot.release()
