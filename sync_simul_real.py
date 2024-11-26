import time
import numpy as np
import mujoco
import mujoco.viewer

from interface import SimulatedRobot
from robot import Robot, OperatingMode

# Set this to 'FK' for forward kinematics testing or 'IK' for inverse kinematics testing
TEST_MODE = 'IK'

# Load the MuJoCo model
m = mujoco.MjModel.from_xml_path(
    '/home/ahrilab/Downloads/koch_ik2/low_cost_robot/low_cost_robot.xml')
d = mujoco.MjData(m)


# Initialize simulated robot
mujoco.mj_resetDataKeyframe(m, d, 1)
mujoco.mj_forward(m, d)
r = SimulatedRobot(m, d)

# Initialize real robot
robot = Robot(device_name='/dev/ttyACM0')

# Read initial positions and set initial simulation state
positions = robot.read_position()
positions = np.array(positions)
d.qpos[:6] = r._pwm2pos(positions)  # Adjust if necessary

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if TEST_MODE == 'FK':
            # Read the current positions from the real robot
            positions = robot.read_position()
            positions = np.array(positions)
            current_qpos = r._pwm2pos(positions)  # Adjust if necessary

            # Compute forward kinematics to get the current end-effector position
            current_ee_pos = r.forward_kinematics(current_qpos)

            # Display the current end-effector position
            print(f"Current end-effector position: {current_ee_pos}")

        elif TEST_MODE == 'IK':
            # Define the target end-effector position
            target_ee_pos = np.array([0.04745835,  0.15619665,  0.05095877])

            # Compute the inverse kinematics to get joint positions
            qpos_ik = r.inverse_kinematics(
                target_ee_pos, rate=0.2, joint_name='joint6')

            interpolated_positions = robot.move_joints(qpos_ik)

            # Visualize each interpolated position in the simulation
            for positions in interpolated_positions:
                # Convert PWM positions back to joint angles
                positions = np.array(positions)

                # Update simulation state
                r.d.qpos[:6] = positions

                # Step and render
                mujoco.mj_forward(r.m, r.d)
                viewer.sync()

                # Add small delay for visualization
                # Adjust this value to control visualization speed
                time.sleep(0.02)

            # break

        # Step the simulation
        mujoco.mj_step(m, d)
        viewer.sync()

        # Sleep to maintain real-time simulation
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
