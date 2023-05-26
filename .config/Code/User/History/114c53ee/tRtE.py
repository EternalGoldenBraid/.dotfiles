from touch_sdk import WatchManager

class MyWatchManager(WatchManager):

    # def __init__(self):
    #     super().__init
    #     self.quat = None

    def on_quat(self, quaternion):
        self.quat = quaternion
        print('quat', quaternion)

    def on_tap(self):
        print('tap')

wm = MyWatchManager()
wm.start()
# input(f"found devices:{print(wm.found_devices)}")

