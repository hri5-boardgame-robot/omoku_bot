from omoku_bot import OmokuBot


class TestOmokuBot:
    def __init__(self):
        self.robot = OmokuBot()

    def run(self):
        self.robot.init_robot()
        # self.robot.enroll_workspace()
        # self.robot.move_to_grid(3, 4)  # Example move to grid position (4, 4)
        self.robot.grasp()

        # Example move to end effector position
        # self.robot.move_ee_postition([-0.05304016, 0.16248379, 0.09452285])


if __name__ == "__main__":
    test = TestOmokuBot()
    test.run()
