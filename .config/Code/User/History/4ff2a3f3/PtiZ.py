import os
from pathlib import Path

import time

import numpy 
import open3d as o3d

from visualizer_utils import (Viewer3D, create_image_frame, draw_image_on_image_frame,
                                )
from utils import (load_object_pointcloud,)

img_width: int = 666
img_height: int = 666

pcl1: o3d.t.geometry.Pointcloud = load_object_pointcloud()

viewer = Viewer3D(title='vis')

for i in range(10):
    viewer.run_one_tick()
    time.sleep(0.1)