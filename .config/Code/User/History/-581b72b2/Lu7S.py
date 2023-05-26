import asyncio
import struct

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError


# GATT characteristic UUIDs

# These are found under the service with a
# service UUID of "4b574af0-72d7-45d2-a1bb-23cd0ec20c57"
ACC_UUID = "4b574af2-72d7-45d2-a1bb-23cd0ec20c57"
GYRO_UUID = "4b574af1-72d7-45d2-a1bb-23cd0ec20c57"
GRAV_UUID = "4b574af3-72d7-45d2-a1bb-23cd0ec20c57"
QUAT_UUID = "4b574af4-72d7-45d2-a1bb-23cd0ec20c57"
DATAFRAME_UUID = "4c574af1-72d7-45d2-a1bb-23cd0ec20c57"


SERVICE_UUID = "008e74d0-7bb3-4ac5-8baf-e5e372cced76"

# These are found under the service with a
# service UUID of "008e74d0-7bb3-4ac5-8baf-e5e372cced76"
GESTURE_UUID = "008e74d1-7bb3-4ac5-8baf-e5e372cced76"
TOUCH_UUID = "008e74d2-7bb3-4ac5-8baf-e5e372cced76"
MOTION_UUID = "008e74d3-7bb3-4ac5-8baf-e5e372cced76"


TOUCH_TYPES = {0: "Down", 1: "Up", 2: "Move"}
MOTION_TYPES = {0: "Rotary", 1: "Back button"}
ROTARY_INFOS = {0: "clockwise", 1: "counterclockwise"}
GESTURES = {0: "None", 1: "Tap"}


class WatchManager:
    def __init__(self):
        self.found_devices = []
        self.last_device = None
        self.scanner = None

    def start(self):
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            pass

    async def run(self):
        self.scanner = BleakScanner(
            self._detection_callback, service_uuids=[SERVICE_UUID]
        )
        await self.scanner.start()
        while True:
            await asyncio.sleep(1)

    async def _detection_callback(self, device, advertisement_data):
        name = (
            advertisement_data.manufacturer_data.get(0xFFFF, bytearray()).decode(
                "utf-8"
            )
            or advertisement_data.local_name
        )

        if SERVICE_UUID in advertisement_data.service_uuids:
            if device in self.found_devices:
                return

            self.found_devices.append(device)

            client = BleakClient(device)
            if client.is_connected:
                return

            print(f"Found {name}")
            try:
                await self._do_connect(client, device)
            except asyncio.exceptions.CancelledError:
                print("connection cancelled from", name)
            except BleakError:
                pass

    async def _do_connect(self, client, device):
        await client.connect()

        def wrapper(function):
            async def wrapped(_, data):
                self.last_device = device
                await self._disconnect_non_last()
                await function(_, data)

            return wrapped

        await client.start_notify(DATAFRAME_UUID, wrapper(self._raw_on_dataframe))
        await client.start_notify(GESTURE_UUID, wrapper(self._raw_on_gesture))
        await client.start_notify(TOUCH_UUID, wrapper(self._raw_on_touch))
        await client.start_notify(MOTION_UUID, wrapper(self._raw_on_motion))

    async def _disconnect_non_last(self):
        await self.scanner.stop()
        for device in self.found_devices:
            if device != self.last_device:
                client = BleakClient(device)
                await client.disconnect()

    async def _raw_on_dataframe(self, _, data):
        dataframe = struct.unpack(">13fq", data)
        self.on_acc(dataframe[:3])
        self.on_grav(dataframe[3:6])
        self.on_gyro(dataframe[6:9])
        self.on_quat(dataframe[9:13])

    def on_gyro(self, angular_velocity):
        pass

    def on_acc(self, acceleration):
        pass

    def on_grav(self, gravity_vector):
        pass

    def on_quat(self, quaternion):
        pass

    async def _raw_on_gesture(self, _, data):
        gesture = struct.unpack(">b", data)
        if GESTURES[gesture[0]] == "Tap":
            self.on_tap()

    def on_tap(self):
        pass

    async def _raw_on_touch(self, _, data):
        touch = struct.unpack(">b2f", data)
        touch_type = TOUCH_TYPES[touch[0]]
        x = touch[1]
        y = touch[2]

        if touch_type == "Down":
            self.on_touch_down(x, y)
        if touch_type == "Up":
            self.on_touch_up(x, y)
        if touch_type == "Move":
            self.on_touch_move(x, y)

    def on_touch_down(self, x, y):
        pass

    def on_touch_up(self, x, y):
        pass

    def on_touch_move(self, x, y):
        pass

    async def _raw_on_motion(self, _, data):
        motion = struct.unpack(">2b", data)
        motion_type = MOTION_TYPES[motion[0]]
        if motion_type == "Rotary":
            value = 1 - 2 * motion[1]  # 0,1 -> +1,-1
            self.on_rotary(value)
        if motion_type == "Back button":
            self.on_back_button()

    def on_rotary(self, direction):
        pass

    def on_back_button(self):
        pass
