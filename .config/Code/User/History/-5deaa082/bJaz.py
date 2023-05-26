from typing import Tuple

from touch_sdk import WatchManager

class MyWatchManager(WatchManager):
    
    ### Why doesn't this work?
    # def __init__(self):
    #     self.quaternion = None
    #     print("OK")

    def on_gyro(self, angularVelocity):
        pass
        # print('gyroscope', angularVelocity)

    def on_acc(self, acceleration):
        pass
        # print('acceleration', acceleration)

    def on_grav(self, gravityVector):
        pass
        # print('gravity', gravityVector)

    def on_quat(self, quaternion):
        self.quaternion = quaternion
        # print('quat', quaternion)

    def on_tap(self):
        print('tap')

    def on_touch_down(self, x, y):
        pass
        # print('touch down', x, y)

    def on_touch_up(self, x, y):
        pass
        # print('touch up', x, y)

    def on_touch_move(self, x, y):
        pass
        # print('touch move', x, y)

    def on_rotary(self, direction):
        pass
        # print('rotary', direction)

    def on_back_button(self):
        pass
        # print('back button')

# wm = MyWatchManager()
# wm.start()
# input(f"found devices:{print(wm.found_devices)}")

