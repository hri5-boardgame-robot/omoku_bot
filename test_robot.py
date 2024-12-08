from omoku_bot import OmokuBot
import time


class TestOmokuBot:
    def __init__(self):
        self.robot = OmokuBot(use_real_robot=False)
        self.robot.init_robot()
        # self.robot.robot_setup()

    def run(self):

        # self.robot.enroll_workspace()
        # self.robot.move_to_grid(3, 4)  # Example move to grid position (4, 4)
        while True:
            # Example move to grid position (4, 4)
            self.robot.move_to_grid(4, 4)
            time.sleep(1)
            print("1")
            self.robot.move_down()
            time.sleep(1)
            print("2")
            self.robot.grasp()
            time.sleep(1)
            print("3")
            self.robot.move_up()
            time.sleep(1)
            print("4")
            self.robot.move_to_grid(1, 4)

            time.sleep(1)
            print("5")
            self.robot.move_down()
            time.sleep(1)
            print("6")
            self.robot.release()
            time.sleep(1)
            self.robot.move_up()
            self.robot.move_to_grid(8, 8)
            print("Done")
        # self.robot.move_ee_postition([-0.05304016, 0.16248379, 0.09452285])


if __name__ == "__main__":
    test = TestOmokuBot()
    test.run()
