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
from visualizer_utils import visualizer_setup, create_image_frame, draw_image_on_image_frame
from geom_utils import (rotate_axis_angle, generate_rotation_matrices, 
                        create_plane, project_to_plane, create_sphere_mesh)

# class Watch(WatchManager):
#     def __init__(self):
#         self.quaternion = None

#     def on_quat(self, quaternion):
#         print('quat', quaternion)
#         self.quaternion = quaternion
#         print(type(self.quaternion))

#     def on_tap(self):
#         print('tap')

watch = Watch()
from time import sleep
# # watch.start()
# # watch.run()
watch_thread = Thread(target=watch.start)
watch_thread.start()
# while True:
#     print("Quat:", watch.quaternion)
#     sleep(0.5)

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

axis = np.array([0, 0, 1])
num_rotations = 100
rotation = Tensor([ [1, 0, 0], [0, 1, 0], [0, 0, 1]])
rot_idx = 0
t_old = 0.0
R_updated: NDArray = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
R_prev: NDArray = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
id: NDArray = np.eye(3, dtype=float)

# def animate(w, time: float, old_axis: List[NDArray]=old_axis, pcl=object_pcl):
# TODO: Wrap this in a class signifying the window.
def animate(w, time: float, pcl=object_pcl):
    # TODO: Get rid of these globals for this callback.
    global rotations
    global rot_idx
    global n_rotations
    global plane_pcl
    global K
    global watch
    global R_prev
    global R_updated

    # w.update_geometry("mesh", mesh_pcl, 0)
    w.remove_geometry("object")
    # w.add_geometry(name="object", geometry=pcl, time=time)
    # w.show_geometry("mesh", True)
    
    # rotate_axis_angle(old_axis=axis, pcl=pcl)
    # pcl.rotate(rotations[rot_idx % num_rotations], center=object_center)

    # Is is w.r.t. to the canonical watch frame. Needs to be updated to take into account previous rotations.
    R = o3d.geometry.get_rotation_matrix_from_quaternion([watch.quaternion[-1], *watch.quaternion[:-1]])
    
    ### NOTE: Keep changing the canonical frame
    # R_updated = R@R_updated
    # print(f"R: {R}")
    # print("Rot idx:", rot_idx)
    # pcl = pcl.rotate(
    #     R_updated,
    #     center=object_center)
    # w.add_geometry(name="object", geometry=pcl, time=time)

    ### NOTE: Invert the pose after applying it
    # R_prev = R
    pcl = pcl.rotate(
        R @ R_prev.T,
        center=object_center)
    w.add_geometry(name="object", geometry=pcl, time=time)
    R_prev = R

    # if 
    # R = o3d.geometry.get_rotation_matrix_from_quaternion(watch.quaternion)
    # if not np.allclose(R @ np.linalg.inv(R_prev), id, atol=atol):
    #     print("Rotated")
    #     pcl.rotate(
    #         # o3d.geometry.get_rotation_matrix_from_quaternion(watch.quaternion),
    #         # o3d.geometry.Geometry3D.get_rotation_matrix_from_quaternion(watch.quaternion),
    #         R,
    #         center=object_center)
    # R_prev = R

    rot_idx += 1
    
    rgbd_img = pcl.project_to_rgbd_image(width=cam_width, height=cam_height,
    # img = pcl.project_to_depth_image(width=cam_width, height=cam_height,
                            extrinsics=extrinsics,
                             intrinsics=intrinsics,
                            depth_scale=depth_scale,
                            depth_max=depth_max
                            )
    img = rgbd_img.color
    img.create_normal_map()
                                      
    # cv2.imshow('projection', np.asarray(img))
    # cv2.waitKey(1)

    # This 
    draw_image_on_image_frame(frame=frame_mesh, frame_size=(frame_height, frame_width),
            # image=rgbd_img.color.create_normal_map(),
            image=img,
     )
    w.remove_geometry("image_frame")
    w.add_geometry("image_frame", frame_mesh, time=time)
    
    
    ### Only way I managed to retain the object in the scene during animation is to set is_animating flag to false
    # for rotation step
    # w.post_redraw()
    # w.is_animating = False
    w.is_animating = True
    return O3DVisualizer.TickResult.REDRAW

geoms: List[Dict] = [
    # {'name': 'coords', 'geometry': coords},
    # # {'name': 'tangent_coords', 'geometry': tangent_coord},
    # {'name': 'rot_axis', 'geometry': rot_axis_mesh, 'material': mat_rot_axis},
    # {'name': 'rot_axis', 'geometry': rot_axis_mesh},
    {'name': 'object', 'geometry': object_pcl},
    {'name': 'image_frame', 'geometry': frame_mesh},
    {'name': 'sphere', 'geometry': sphere_mesh, 'material': mat_sphere},
]


# o3d.visualization.draw_geometries_with_animation_callback(geoms, animate)
# o3d.visualization.draw_geometries_with_animation_callback([object_pcl], animate)
visualizer_setup(geoms=geoms, callback=animate, 
                # raw_mode=True,
                show_skybox=False
                )