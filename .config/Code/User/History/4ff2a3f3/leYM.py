import os
from typing import List, Dict
from pathlib import Path

import time

import numpy 
import open3d as o3d

from visualizer_utils import (Viewer3D, create_image_frame, draw_image_on_image_frame,
                                )
from utils import (load_object_pointcloud,)

DEVICE = o3d.core.Device('CPU:0')


# Load objects
pcl1: o3d.t.geometry.PointCloud = load_object_pointcloud(device=DEVICE)

geoms: List[Dict] = [
    {'name': 'pcl1', 'geometry': pcl1},
]

viewer = Viewer3D(title='vis')
viewer.setup_point_clouds(geoms=geoms)

# Projection image setup
img_width: int = 666
img_height: int = 666

# for i in range(10):
    # viewer.run_one_tick()
    # time.sleep(0.1)
while True:
    viewer.run_one_tick()