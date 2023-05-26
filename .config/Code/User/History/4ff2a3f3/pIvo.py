import os
from typing import List, Dict
from pathlib import Path

import time

import numpy as np
import open3d as o3d
from open3d.core import Tensor, float32

from visualizer_utils import (Viewer3D, create_image_frame, draw_image_on_image_frame,
                                )
from utils import (load_object_pointcloud,)

DEVICE = o3d.core.Device('CPU:0')


# Load objects
pcl1: o3d.t.geometry.PointCloud = load_object_pointcloud(device=DEVICE)

viewer = Viewer3D(title='vis')
viewer.setup_point_clouds(geoms=geoms)

# Projection image setup
img_width: int = 666
img_height: int = 666

### Rendering params
dist: float = 10.0
x:float = 0.0
y:float = 0.0
z:float = dist
depth_max: float = 10000.0
depth_scale: float = 15.0
intrinsic_scale: float = 1.0
c_scale: float = 120000//1000
rx = ry = rz = 0

extrinsics: Tensor = Tensor(np.eye(4),float32)
extrinsics[:3,3] = (x,y,z)
extrinsics[:3,:3] = np.eye(3)
intrinsics: Tensor = Tensor([[c_scale,  0.0,   img_width*0.5],
                            [ 0.0, c_scale,    img_height*0.5],
                            [ 0.0,  0.0,   1.0]],
                            float32)

### Construct frame to render image on.
frame_mesh: o3d.t.geometry.TriangleMesh = create_image_frame(width=img_width, height=img_height) 
frame_mesh.translate((0,0,dist))

geoms: List[Dict] = [
    {'name': 'pcl1', 'geometry': pcl1},
    {'name': 'frame_mesh', 'geometry': frame_mesh},
]

while True:

    # img:  = pcl.project_to_depth_image(width=cam_width, height=cam_height,
    img: o3d.t.geometry.RGBDImage = pcl1.project_to_rgbd_image(
                        width=img_width, height=img_height,
                        extrinsics=extrinsics, intrinsics=intrinsics,
                        depth_scale=depth_scale, depth_max=depth_max
                        )
    draw_image_on_image_frame(frame=frame_mesh, image=img.color, 
                            frame_size=(img_height, img_width))
    viewer.run_one_tick()