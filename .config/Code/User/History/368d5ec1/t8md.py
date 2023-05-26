"""
Animate the rotation of a point cloud
"""

from typing import Dict, List
import time
from math import pi
from threading import Thread

import cv2

import open3d as o3d
from open3d.core import Tensor, float32
from open3d.visualization import O3DVisualizer
from open3d.visualization import gui, rendering

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.transform import Rotation as sp_R

from apple_pygatt.examples.basic import MyWatchManager as Watch
# from touch_sdk import WatchManager

from utils import load_object_pointcloud
from visualizer_utils import (visualizer_setup, create_image_frame,
                            draw_image_on_image_frame, Viewer3D)
from geom_utils import (rotate_axis_angle, generate_rotation_matrices, 
                        create_plane, project_to_plane, create_sphere_mesh,
                        Intermediate_Projections)

DEVICE = o3d.core.Device('CPU:0')
# DEVICE = o3d.core.Device('CUDA:0')

radius=1000//2
# radius=0

### Create sphere
sphere_mesh, mat_sphere = create_sphere_mesh(coords=(0,0,0),  radius=radius)

### Load object to be rotated
print("Loading PCL")
object_pcl = load_object_pointcloud(device=DEVICE, n_points=20000)
object_center = Tensor(np.array([0,0,0]), float32, device=DEVICE)
object_pcl.translate(object_center)
print("PCL min/max")
print(object_pcl.point.positions.min(), object_pcl.point.positions.max())
print(30*"#")

### Construct projection plane (camera)
# frame_width: int = 2000
# frame_height: int = 2000
frame_width: int = 640
frame_height: int = 480
cam_width: int = 640
cam_height: int = 480

### Rendering params
# x, y, z, = -0.44999999999999996, -0.5, 0
x, y, z, = 0.0, 0.0, radius
depth_max = 10000.0
depth_scale = 15.0
intrinsic_scale = 1.0
c_scale = 120000//500
rx = ry = rz = 0

extrinsics: Tensor = Tensor(np.eye(4),float32)
extrinsics[:3,3] = (x,y,z)
extrinsics[:3,:3] = np.eye(3)
intrinsics: Tensor = Tensor([[c_scale,  0.0,   cam_width*0.5],
                            [ 0.0, c_scale,    cam_height*0.5],
                            [ 0.0,  0.0,   1.0]],
                            float32)

### Construct frame to render image on.
frame_mesh: o3d.t.geometry.TriangleMesh = create_image_frame(width=frame_width, height=frame_height) 
frame_mesh.translate((0,0,radius))

### Virtual Camera pos
# rot_axis_mesh: o3d.t.geometry.TriangleMesh  = create_image_frame(width=frame_width, height=frame_height)
# rot_axis_mesh.translate((0,0, radius))

layered_projections = Intermediate_Projections(main_pcl=object_pcl,
                    main_pcl_center=object_center, n_basis=5, device=DEVICE,
                    extrinsics=extrinsics, intrinsics=intrinsics,
                    )
projections: List[Dict[str, o3d.t.geometry.PointCloud]] = layered_projections.bases

geoms: List[Dict] = [
    # {'name': 'coords', 'geometry': coords},
    # # {'name': 'tangent_coords', 'geometry': tangent_coord},
    # {'name': 'rot_axis', 'geometry': rot_axis_mesh, 'material': mat_rot_axis},
    # {'name': 'rot_axis', 'geometry': rot_axis_mesh},
    {'name': 'object', 'geometry': object_pcl},
    {'name': 'image_frame', 'geometry': frame_mesh},
    {'name': 'sphere', 'geometry': sphere_mesh, 'material': mat_sphere},
]
for proj in projections:
    geoms.append(proj)

viewer = Viewer3D(title='demo', width=1440, height=720)
viewer.setup_point_clouds(geoms=geoms)
viewer.setup_watch()

axis = np.array([0, 0, 1])
rot_idx = 0
t_old = 0.0
R_updated: NDArray = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
R_prev: NDArray = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)

from time import sleep
tick = -1
while True:
    tick += 1

    # w.update_geometry("mesh", mesh_pcl, 0)
    # w.remove_geometry("object")
    # w.add_geometry(name="object", geometry=pcl, time=time)
    # w.show_geometry("mesh", True)
    
    # rotate_axis_angle(old_axis=axis, pcl=pcl)
    # pcl.rotate(rotations[rot_idx % num_rotations], center=object_center)

    # Is is w.r.t. to the canonical watch frame. Needs to be updated to take into account previous rotations.
    R = o3d.geometry.get_rotation_matrix_from_quaternion(
        [viewer.watch.quaternion[-1], *viewer.watch.quaternion[:-1]])
    print(R)

    ### NOTE: Invert the pose after applying it
    object = object_pcl.rotate(
        R @ R_prev.T,
        center=object_center)

    R_prev = R

    rgbd_img = object_pcl.project_to_rgbd_image(width=cam_width, height=cam_height,
                            extrinsics=extrinsics,
                             intrinsics=intrinsics,
                            depth_scale=depth_scale,
                            depth_max=depth_max
                            )
    img = rgbd_img.color
    # img.create_normal_map()

    draw_image_on_image_frame(frame=frame_mesh, frame_size=(frame_height, frame_width),
            # image=rgbd_img.color.create_normal_map(),
            image=img,
     )
    
    if tick % 100 == 0: print("Tick:", tick)
    viewer.main_vis.remove_geometry("object")
    viewer.main_vis.add_geometry(name="object", geometry=object_pcl)

    viewer.main_vis.remove_geometry("image_frame")
    viewer.main_vis.add_geometry(name="image_frame", geometry=frame_mesh)

    viewer.run_one_tick()