import time
import numpy as np
import mujoco
import mujoco.viewer

from interface import SimulatedRobot
from robot import Robot, OperatingMode

# Set this to False when you don't want to interact with the real robot
USE_REAL_ROBOT = False

# Load the MuJoCo model
m = mujoco.MjModel.from_xml_path(
    '/home/ahrilab/Downloads/koch_ik2/low_cost_robot/low_cost_robot.xml')
d = mujoco.MjData(m)

# Initialize simulated robot
r = SimulatedRobot(m, d)
robot = Robot(device_name='/dev/ttyACM0')


with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        positions = robot.read_position()
        positions = np.array(positions)
        current_qpos = r._pwm2pos(positions)

        # Compute forward kinematics to get the current end-effector position
        current_ee_pos = r.forward_kinematics(current_qpos)

        # Display the current end-effector position
        print(f"Current end-effector position: {current_ee_pos}")

        # Update the simulation with the computed joint positions

        # Send the position commands to the real robot
        if USE_REAL_ROBOT:  # Define the target end-effector position
            target_ee_pos = np.array([0.1, 0.0, 0.1])

            # Compute the inverse kinematics to get joint positions
            qpos_ik = r.inverse_kinematics(
                target_ee_pos, rate=0.2, joint_name='joint6')
            d.qpos[:6] = qpos_ik[:6]
            desired_positions = r._pos2pwm(qpos_ik[:6]).astype(int)

            print(f"Desired joint positions: {qpos_ik[:6]}")
            robot.set_goal_pos(desired_positions)

        # Step the simulation
        mujoco.mj_step(m, d)
        viewer.sync()

        # Sleep to maintain real-time simulation
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
