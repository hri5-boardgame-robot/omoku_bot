from omoku_bot import omoku_bot


class TestOmokuBot:
    def __init__(self):
        self.robot = omoku_bot()

    def run(self):
        self.robot.init_robot()
        # self.robot.enroll_workspace()
        self.robot.move_to_grid(4, 4)  # Example move to grid position (4, 4)


if __name__ == "__main__":
    test = TestOmokuBot()
    test.run()
