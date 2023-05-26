from touch_sdk import WatchManager

class MyWatchManager(WatchManager):
    def __init__(self):
        self.quaternion = None
        input("HEP")

    # def on_gyro(self, angularVelocity):
    #     print('gyroscope', angularVelocity)

    # def on_acc(self, acceleration):
    #     print('acceleration', acceleration)

    # def on_grav(self, gravityVector):
    #     print('gravity', gravityVector)

    def on_quat(self, quaternion):
        print('quat', quaternion)
        self.quaternion = quaternion
        print(type(self.quaternion))

    def on_tap(self):
        print('tap')

    # def on_touch_down(self, x, y):
    #     print('touch down', x, y)

    # def on_touch_up(self, x, y):
    #     print('touch up', x, y)

    # def on_touch_move(self, x, y):
    #     print('touch move', x, y)

    # def on_rotary(self, direction):
    #     print('rotary', direction)

    # def on_back_button(self):
    #     print('back button')

# wm = MyWatchManager()
# wm.start()