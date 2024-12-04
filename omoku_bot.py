import time
# import keyboard

import numpy as np
import mujoco
import mujoco.viewer
from interface import SimulatedRobot
from robot import Robot, OperatingMode
from scipy.spatial.transform import Rotation as R


class omoku_bot:
    def __init__(self):
        self.max_distance = 0.30986731312447995
        self.min_distance = 0.10611490569775514
        self.positions = [3084, 2038, 1984, 1013, 2036, 1969]
        self.robot = None  # Initialize your robot object
        self.r = None  # Initialize your robot's specific attributes
        self.d = None  # Initialize your Mujoco data object
        self.m = None  # Initialize your Mujoco model object
        self.robot_setup()
        self.etc()
        # self.enroll_workspace()
        # self.mode_selection()

    def robot_setup(self):
        self.m = mujoco.MjModel.from_xml_path(
            'omoku_bot/low_cost_robot/low_cost_robot.xml')
        self.d = mujoco.MjData(self.m)
        self.r = SimulatedRobot(self.m, self.d)
        # self.robot = Robot(device_name='/dev/ttyACM0')
        self.robot = Robot(device_name='/dev/tty.usbmodem58760435301')

    def init_robot(self):
        self.move_joint([3084, 2038, 1984, 1013, 2036, 1969])
        self.release()

    def etc(self):
        self.gripper("test")
        print("ROBOT SETUP DONE")
        self.telescope_rate = 1.5e-2
        self.tolerance = 1e-5
        self.max_distance = 0.30986731312447995
        self.min_distance = 0.10611490569775514

    # def mode_selection(self):

    #     with mujoco.viewer.launch_passive(self.m, self.d) as self.viewer:
    #         while self.viewer.is_running():

    #             mode = input("Enter the mode: ")
    #             if mode == "t":
    #                 self.telescope_mode()
    #             elif mode == "g":
    #                 self.target_mode()
    #             elif mode == "d":
    #                 self.debug_mode()

    def telescope_mode(self):
        while True:
            key = input()
            # key = keyboard.read_event()
            destination = self.get_ee_xyz()
            if key == "w":
                destination[1] += self.telescope_rate
            elif key == "s":
                destination[1] -= self.telescope_rate
            elif key == "a":
                destination[0] -= self.telescope_rate
            elif key == "d":
                destination[0] += self.telescope_rate
            elif key == "q":
                destination[2] += self.telescope_rate
            elif key == "e":
                destination[2] -= self.telescope_rate
            elif key == "z":
                break
            print("destination:", self.get_ee_xyz(), destination)
            if self.check_workspace(destination) == True:
                self.move("telescope_mode", destination)
            else:
                print("Out of workspace")

    def target_mode(self):
        while True:
            key = input("input target position:")
            if key == "z":
                break
            destination = key.split(" ")
            destination = list(map(float, destination))

            if self.check_workspace(destination) == True:
                self.move("target_mode", destination)
                self.gripper("close")
                # time.sleep(2)s
                self.gripper("open")
            else:
                print("Out of workspace")

    def debug_mode(self):
        self.robot._disable_torque()
        while True:
            cmd = input(
                "Debug mode: check the current position by just pressing enter, to quit, press z")

            print(f"\nEE_xyz: {self.get_ee_xyz()}")
            print(f"Joint_rad: {self.get_joint_rad()}")
            print(f"PWM_value: {np.array(self.robot.read_position())} \n")
            self.viewer.sync()
            if cmd == "z":
                self.robot._enable_torque()
                break

    def move_joint(self, positions):
        pwm_values = np.array(positions)
        current_position = self.robot.read_position()
        interpolated_positions = self.robot.get_interpolate_pose(
            current_position, pwm_values)

        for positions in interpolated_positions:
            positions = np.array(positions)
            self.robot.set_goal_pos(positions)
            current_qpos = self.r._pwm2pos(positions)
            self.d.qpos[:6] = current_qpos[:6]
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            time.sleep(0.01)

    def move_ee_postition(self, mode, destination):

        destination = np.array(destination)
        target_ee_rot = R.from_euler(
            'z', 0, degrees=True).as_matrix().flatten()
        max_iterations = 100 if mode == "telescope_mode" else 200
        for _ in range(max_iterations):
            qpos_ik = self.r.inverse_kinematics_rot(destination, target_ee_rot,
                                                    rate=0.05, joint_name='joint6')

            self.d.qpos[:5] = qpos_ik[:5]
            current_position = qpos_ik
            mujoco.mj_forward(self.m, self.d)
            pwm_values = self.r._pos2pwm(qpos_ik).astype(int)
            time.sleep(0.02)
            ee_pos = self.d.xpos[mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_BODY, 'joint6')]
            error = np.linalg.norm(destination - ee_pos)
            if error < self.tolerance:
                print(f"Converged to target position with error: {error}")
                break
        pwm_values = self.r._pos2pwm(qpos_ik).astype(int)
        current_position = self.robot.read_position()
        interpolated_positions = self.robot.get_interpolate_pose(
            current_position, pwm_values)
        # print(pwm_values)

        for positions in interpolated_positions:
            # Convert PWM positions back to joint angles
            self.positions = positions
            positions = np.array(positions)
            self.robot.set_goal_pos(positions)
            # print(f"Setting goal position: {positions}")
            current_qpos = self.r._pwm2pos(positions)

            # Update simulation state
            self.d.qpos[:5] = current_qpos[:5]

            # Step and render
            mujoco.mj_forward(self.m, self.d)
            self.viewer.sync()

            # Add small delay for visualization
            time.sleep(0.015)
            mujoco.mj_step(self.m, self.d)

        self.viewer.sync()
        mujoco.mj_step(self.m, self.d)

    def gripper(self, mode):
        if mode == "open":
            self.positions[5] = 2200
        elif mode == "close":
            self.positions[5] = 1965

        pwm_values = np.array(self.positions)
        current_position = self.robot.read_position()
        interpolated_positions = self.robot.get_interpolate_pose(
            current_position, pwm_values)
        for positions in interpolated_positions:
            positions = np.array(positions)
            self.robot.set_goal_pos(positions)
            current_qpos = self.r._pwm2pos(positions)
            self.d.qpos[:6] = current_qpos[:6]
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            time.sleep(0.01)

    def get_ee_xyz(self):
        positions = np.array(self.robot.read_position())
        current_qpos = self.r._pwm2pos(positions)
        current_ee_xyz = self.r.forward_kinematics(current_qpos)
        return current_ee_xyz

    def get_joint_rad(self):
        positions = np.array(self.robot.read_position())
        current_qpos = self.r._pwm2pos(positions)
        return current_qpos

    def enroll_workspace(self):
        distance_list = []
        for _ in range(4):
            case = input("set the workspace")
            x, y, z = self.get_ee_xyz()
            distance = np.linalg.norm([x, y, z])
            distance_list.append(distance)

        distance_list = np.array(distance_list)
        self.max_distance = np.max(distance_list)
        self.min_distance = np.min(distance_list)
        print(f"max distance: {self.max_distance}")
        print(f"min distance: {self.min_distance}")

    def check_workspace(self, destination):
        # print(destination)
        distance = np.linalg.norm(destination)
        if distance > self.max_distance or distance < self.min_distance:
            return False
        else:
            return True

    def move_to_grid(self, grid_x, grid_y):
        """
        Move the robot to a specific grid position on a 9x9 grid.
        grid_x: x-coordinate of the grid (0-8)
        grid_y: y-coordinate of the grid (0-8)
        """
        if not (0 <= grid_x < 9 and 0 <= grid_y < 9):
            raise ValueError("Grid coordinates must be between 0 and 8")

        # Calculate the position based on the grid coordinates
        # Scale the grid coordinates to fit within the workspace bounds
        x_position = self.min_distance + \
            (self.max_distance - self.min_distance) * (grid_x / 8)
        y_position = self.min_distance + \
            (self.max_distance - self.min_distance) * (grid_y / 8)
        z_position = 0.03  # Assuming a fixed z position for simplicity

        destination = np.array([x_position, y_position, z_position])
        print(f"Calculated destination: {destination}")

        if self.check_workspace(destination):
            # Move the robot to the calculated position
            print(f"Moving to grid position: ({grid_x}, {grid_y})")
            self.move_ee_postition("target_mode", destination)
        else:
            print("Destination is out of workspace bounds")

    def grasp(self):
        self.positions[5] = 1965
        self._move_gripper()

    def release(self):
        self.positions[5] = 2200
        self._move_gripper()

    def _move_gripper(self):
        pwm_values = np.array(self.positions)
        current_position = self.robot.read_position()
        interpolated_positions = self.robot.get_interpolate_pose(
            current_position, pwm_values)

        for positions in interpolated_positions:
            positions = np.array(positions)
            self.robot.set_goal_pos(positions)
            current_qpos = self.r._pwm2pos(positions)
            self.d.qpos[:6] = current_qpos[:6]
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            time.sleep(0.01)

    def _pwm2pos(self, positions):
        # Convert PWM to position
        pass

# # Example usage
# if __name__ == "__main__":
#     robot = omoku_bot()
#     robot.init_robot()
#     robot.enroll_workspace()
#     robot.move_to_grid(4, 4)
