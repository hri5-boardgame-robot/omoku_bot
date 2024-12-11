import time
# import keyboard
import os
import numpy as np
import mujoco
import mujoco.viewer
import json
from interface import SimulatedRobot
from robot import Robot, OperatingMode
from scipy.spatial.transform import Rotation as R

# Default positions for illustrative purposes
HOME_POSITION = [0.0, 0.15, 0.15]          # Example "home" hover position
DEFAULT_UP_Z = 0.14                        # "Safe" hover height
DEFAULT_DOWN_Z = 0.095                    # Lower position for grasp/release

class OmokuBot:
    def __init__(self, use_real_robot=True, device_name='/dev/ttyACM0', workspace_enroll=False):
        """
        Initialize the OmokuBot class, set up the robot, and define workspace boundaries.
        If use_real_robot=False, code will run simulation only.
        """
        self.use_real_robot = use_real_robot
        self.robot = None
        self.sim = None
        self.d = None
        self.m = None
        self.viewer = None
        self.position_pwm = None
        self.usb_device = device_name
        self.workspace_enroll = workspace_enroll
        # Robot setup
        self.robot_setup()

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
        Set up the Mujoco model and SimulatedRobot interface, and connect to the real robot if enabled.
        """
        self.m = mujoco.MjModel.from_xml_path(
            'low_cost_robot/scene.xml')
        self.d = mujoco.MjData(self.m)
        self.sim = SimulatedRobot(self.m, self.d)

        if self.use_real_robot:
            self.robot = Robot(device_name=self.usb_device)
            if self.workspace_enroll:
                self.enroll_workspace()
            with open("workspace.json", "r") as json_file:
                workspace = json.load(json_file)
                self.set_workspace(workspace)
            self.reload()
            # self.calibration()
        else:
            # No real robot: robot remains None
            self.robot = None

    def init_robot(self):
        """
        Initialize the robot viewer and move to a default position.
        """
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        # Ensure gripper is open at start
        robot_pwm = np.array([int(i) for i in self.robot.read_position()])
        self.position_pwm = self.sim._pwm2pos(robot_pwm)
        mujoco.mj_forward(self.m, self.d)
        self.viewer.sync()
        self.move_ee_position(HOME_POSITION)
        if self.viewer is not None:
            self.viewer.sync()

    def move_ee_position(self, destination):
        destination = np.array(destination)
        target_ee_rot = R.from_euler('x', -90, degrees=True).as_matrix()

        current_ee_pos = self.get_ee_xyz()
        real_robot_position = self.robot.read_position()
        # Interpolate between current_ee_pos and destination

        num_waypoints = 100
        waypoints = np.linspace(current_ee_pos, destination, num_waypoints)

        for waypoint in waypoints:
            qpos_ik = self.sim.inverse_kinematics_rot(
                waypoint, target_ee_rot, rate=0.005, joint_name='joint6'
            )
            if self.use_real_robot:
                # Convert joint positions to PWM values
                pwm_values = self.sim._pos2pwm(qpos_ik[:5]).astype(int)
                current_gripper_pwm = real_robot_position[5]
                full_pwm_values = np.concatenate(
                    (pwm_values, [current_gripper_pwm]))
                full_pwm_values = [int(i) for i in full_pwm_values]

                self.robot.set_goal_pos(full_pwm_values)
                mujoco.mj_forward(self.m, self.d)
                time.sleep(0.002)

                # Update simulation from actual robot position
                current_position = self.robot.read_position()
                converted_positions = self.sim._pwm2pos(
                    np.array(current_position))
                self.d.qpos[:5] = converted_positions[:5]
            else:
                # Simulation-only
                self.d.qpos[:5] = qpos_ik[:5]

            mujoco.mj_forward(self.m, self.d)

            if self.viewer is not None:
                self.viewer.sync()

    def gripper(self, mode):
        """
        Control the gripper.
        mode: "open" or "close"
        """
        if mode == "open":
            self.position_pwm[5] = 2200
        elif mode == "close":
            self.position_pwm[5] = 1865

        self._move_gripper()

    def _move_gripper(self):
        """
        Internal method to move the gripper to the currently set positions[5].
        """
        if self.use_real_robot:
            pwm_values = np.array(self.position_pwm)
            current_position = self.robot.read_position()
            interpolated_positions = self.robot.get_interpolate_pose(
                current_position, pwm_values
            )
            for pos in interpolated_positions:
                pos = np.array(pos)
                self.robot.set_goal_pos(pos)
                current_qpos = self.sim._pwm2pos(pos)
                self.d.qpos[:6] = current_qpos[:6]
                mujoco.mj_forward(self.m, self.d)
                mujoco.mj_step(self.m, self.d)
                time.sleep(0.01)
        else:
            # Simulation-only: update qpos directly

            target_qpos = self.position_pwm
            current_qpos = self.sim.read_position()

            self.d.qpos[:6] = self.sim._pwm2pos(target_qpos)[:6]
            mujoco.mj_forward(self.m, self.d)
            if self.viewer is not None:
                self.viewer.sync()
            time.sleep(0.02)

    def get_ee_xyz(self):
        """
        Get the current end-effector XYZ position.
        If running simulation-only, returns simulated EE position.
        """
        if self.use_real_robot:
            positions = np.array(self.robot.read_position())
        else:
            # Simulation-only: Use stored positions or current qpos
            positions = self.sim.read_position()

        current_qpos = self.sim._pwm2pos(positions)
        current_ee_xyz = self.sim.forward_kinematics(current_qpos)
        return current_ee_xyz

    def get_joint_rad(self):
        """
        Get the current joint angles (in radians).
        """
        if self.use_real_robot:
            positions = np.array(self.robot.read_position())
        else:
            positions = self.position_pwm
        current_qpos = self.sim._pwm2pos(positions)
        return current_qpos

    def set_workspace(self,workspace):
        """
        Set workspace boundaries.
        """
        self.x_min_distance = workspace['0_0'][0]
        self.x_max_distance = workspace['8_8'][0]
        self.y_min_distance = workspace['8_8'][1]
        self.y_max_distance = workspace['0_0'][1]
        self.reload_position = workspace['starting']

    def calibration(self):
        full_grid_point = [[j,i] for j in [0,8] for i in [0,8]]
        calibartion_value = {}
        tolerance = 0.001
        telescope_rate = 3.5e-4
        # print(full_grid_point)
        for grid in full_grid_point:
            calibartion_value[str(grid)] = [0,0,0]
            dest = self.grid_to_xyz(grid[0], grid[1])
            print("dest: ",dest, grid[0], grid[1])    
            for i in range(2):
                count = 0
                while True:
                    self.move_ee_position(np.array([dest[0]+calibartion_value[str(grid)][0], 
                                                    dest[1]+calibartion_value[str(grid)][1],
                                                    dest[2]+calibartion_value[str(grid)][2]]))

                    cur = self.get_ee_xyz()
                    if np.abs(cur[i] - dest[i]) > tolerance:
                        count = 0
                        calibartion_value[str(grid)][i] -= (cur[i] - dest[i])/np.abs(cur[i] - dest[i]) * telescope_rate
                        
                        time.sleep(1)
                        self.move_ee_position(np.array([0,0.2,0.15]))
                        print("calibration: ",calibartion_value,"x: ",cur[0]-dest[0],"y: ",cur[1]-dest[1],"z: ",cur[2]-dest[2])
                    else:
                        print("calibration success")
                        count +=1
                        if count > 2:
                            break
                        else:
                            self.move_ee_position(np.array([0,0.2,0.15]))
        

    def grid_to_xyz(self, grid_y, grid_x, z_plane=0.14):
        x_min, x_max = self.x_min_distance, self.x_max_distance
        y_min, y_max = self.y_min_distance, self.y_max_distance
        
        if not (0 <= grid_x < 9 and 0 <= grid_y < 9):
            raise ValueError("Grid coordinates must be between 0 and 8")
        else:
            x_position = x_min + (x_max - x_min) * (grid_x / 8)
            y_position = y_min + (y_max - y_min) * (1-(grid_y / 8))
            destination = np.array([x_position, y_position, z_plane])
            return destination
        
    def move_to_grid(self, grid_y, grid_x, z_plane=0.14):
        """
        Move the robot to a specific grid position on a 9x9 grid.
        grid_x, grid_y: integer coordinates between 0 and 8
        z_plane: target z height
        """
        destination = self.grid_to_xyz(grid_y, grid_x, z_plane)
        self.move_ee_position(destination)

    def grasp(self):
        # Example: just close gripper
        self.position_pwm = self.sim.read_position()
        self.gripper("close")

    def release(self):
        # Example: just open gripper
        self.position_pwm = self.sim.read_position()
        self.gripper("open")
        # time.sleep(1000)

    def robot_playing(self,grid_y, grid_x):
        self.move_to_grid(grid_y, grid_x)
        time.sleep(0.5)
        self.move_down()
        time.sleep(0.5)
        self.release()
        control_robot_position = self.robot.read_position()
        print("control_robot_position: ",control_robot_position)    
        control_robot_position[2] += 300
        # control_robot_position[3] -= 300
        print("edited_robot_position: ",control_robot_position) 
        self.robot.set_goal_pos(control_robot_position)
        time.sleep(0.5)
        # if grid_y > 1:
        #     self.move_up()
        #     time.sleep(0.5)
        self.move_to_grid(4, 4)
        time.sleep(1)
        self.reload()
        # self.move_to_grid(4, 4)

    def reload(self):
        # self.move_to_grid(4, 4)
        x,y,z = self.reload_position
        self.move_ee_position([x,y,DEFAULT_UP_Z]) 
        time.sleep(0.5)
        self.release()
        time.sleep(0.5)
        self.move_down(z)
        time.sleep(0.5)
        self.grasp()
        time.sleep(0.5)
        control_robot_position = self.robot.read_position()
        control_robot_position[2] += 400
        self.robot.set_goal_pos(control_robot_position)
        print("done")
        self.move_to_grid(8, 0) 
        
        time.sleep(0.5)
        print("reload done")

    def move_up(self,z=DEFAULT_UP_Z):
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position([x, y, z])

    def move_down(self,z=DEFAULT_DOWN_Z):
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position([x, y, z])

    def enroll_workspace(self):
        workspace = {}
        print("enroll")
        sequence = ['0_0','8_0','0_8','8_8', 'starting']
        for grid in sequence:
            self.robot._disable_torque()
            key = input(f"Press enter the any key to enroll {grid} position")        
            x,y,z = self.get_ee_xyz()
            workspace[grid] = [x,y,z]
        with open("workspace.json", "w") as json_file:
            json.dump(workspace, json_file)
        self.robot._enable_torque()


# Example usage (uncomment if you want to run this standalone):
# if __name__ == "__main__":
#     # Set use_real_robot=False to run simulation only
#     robot = OmokuBot(use_real_robot=False)
#     robot.init_robot()
#     robot.set_workspace(-0.07, 0.07, 0.10, 0.24)
#     robot.move_to_grid(4, 4)
#     robot.grasp()
#     robot.release()
