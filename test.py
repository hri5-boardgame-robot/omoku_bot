from omoku_bot_v2 import OmokuBot
import time

# Example pick position          # Example "home" hover position
TAKE_PICK_POSITION = [-0.08, 0.085, 0.14]


class TestOmokuBot:
    def __init__(self):
        self.robot = OmokuBot(use_real_robot = True, device_name = "/dev/tty.usbmodem58760435301",
                              workspace_enroll=True)
        self.robot.init_robot()
        # self.robot.robot_setup()

    def run(self):
        # self.robot.move_to_grid(3, 4)  # Example move to grid position (4, 4)
        while True:
            key = input("manual test: ")
            key = [int(i) for i in key.split(" ")]
            if len(key) == 2:
                self.robot.robot_playing(key[0], key[1])
                # self.robot.move_to_grid(key[0], key[1])
            if len(key) == 1:
                if key[0] == 0:
                    self.robot.grasp()
                elif key[0] == 1:
                    self.robot.release()
                elif key[0] == 2:
                    self.robot.reload()
                elif key[0] == 3:
                    self.robot.move_down()
                elif key[0] == 4:
                    self.robot.move_up()
                else:
                    print(self.robot.get_ee_xyz())

         


if __name__ == "__main__":
    test = TestOmokuBot()
    test.run()
