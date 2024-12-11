from omoku_bot import OmokuBot
import time

# Example pick position          # Example "home" hover position
TAKE_PICK_POSITION = [-0.08, 0.085, 0.14]


class TestOmokuBot:
    def __init__(self):
        self.robot = OmokuBot(use_real_robot = False)
        self.robot.init_robot()
        # self.robot.robot_setup()

    def run(self):

        # self.robot.enroll_workspace()
        # self.robot.move_to_grid(3, 4)  # Example move to grid position (4, 4)
        while True:
            self.robot.move_ee_position(TAKE_PICK_POSITION)
            time.sleep(1)
            #move grid (0,4)..... (8,4)
            self.robot.move_to_grid(0, 4)
            time.sleep(1)
            self.robot.move_to_grid(1, 4)
            time.sleep(1)
            self.robot.move_to_grid(2, 4)
            time.sleep(1)
            self.robot.move_to_grid(3, 4)
            time.sleep(1)
            self.robot.move_to_grid(4, 4)
            time.sleep(1)
            self.robot.move_to_grid(5, 4)
            time.sleep(1)
            self.robot.move_to_grid(6, 4)
            time.sleep(1)
            self.robot.move_to_grid(7, 4)
            time.sleep(1)
            self.robot.move_to_grid(8, 4)



        # self.robot.move_ee_postition([-0.05304016, 0.16248379, 0.09452285])


if __name__ == "__main__":
    test = TestOmokuBot()
    test.run()
