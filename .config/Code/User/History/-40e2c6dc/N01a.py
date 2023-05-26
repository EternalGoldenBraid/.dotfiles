"""
Animate the rotation of a point cloud
"""
from typing import Dict, List

import cv2

import open3d as o3d
from open3d.core import Tensor, float32
from open3d.visualization import O3DVisualizer
from open3d.visualization import gui, rendering

import numpy as np
from numpy.typing import NDArray
import time
from math import pi

from utils import load_object_pointcloud
from visualizer_utils import visualizer_setup, create_image_frame, draw_image_on_image_frame
from geom_utils import (rotate_axis_angle, generate_rotation_matrices, 
                        create_plane, project_to_plane, create_sphere_mesh)

DEVICE = o3d.core.Device('CPU:0')
# DEVICE = o3d.core.Device('CUDA:0')


radius=1000//2
# radius=0

### Create sphere
sphere_mesh, mat_sphere = create_sphere_mesh(coords=(0,0,0),  radius=radius)

### Load object to be rotated
print("Loading PCL")
object_pcl = load_object_pointcloud(device=DEVICE, n_points=2000)
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
c_scale = 120000//1000
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

axis = np.array([0, 0, 1])
num_rotations = 100
rotations = generate_rotation_matrices(initial_axis=axis, num_rotations=num_rotations)
rot_idx = 0
t_old = 0.0

# def animate(w, time: float, old_axis: List[NDArray]=old_axis, pcl=object_pcl):
def animate(w, time: float, pcl=object_pcl):
    global rotations
    global rot_idx
    global n_rotations
    global plane_pcl
    global K

    # w.update_geometry("mesh", mesh_pcl, 0)
    w.remove_geometry("object")
    w.add_geometry(name="object", geometry=pcl, time=time)
    # w.show_geometry("mesh", True)
    
    
    # rotate_axis_angle(old_axis=axis, pcl=pcl)
    pcl.rotate(rotations[rot_idx % num_rotations], center=object_center)
    rot_idx += 1
    
    # img = pcl.project_to_rgbd_image(width=cam_width, height=cam_height,
    img = pcl.project_to_depth_image(width=cam_width, height=cam_height,
                            extrinsics=extrinsics,
                             intrinsics=intrinsics,
                            depth_scale=depth_scale,
                            depth_max=depth_max
                            )
                                      
    cv2.imshow('projection', np.asarray(img))
    cv2.waitKey(1)

    draw_image_on_image_frame(frame=frame_mesh, image=img, frame_size=(frame_height, frame_width))
    w.remove_geometry("image_frame")
    w.add_geometry("image_frame", frame_mesh, time=time)
    
    
    ### Only way I managed to retain the object in the scene during animation is to set is_animating flag to false
    # for rotation step
    # w.post_redraw()
    # w.is_animating = False
    w.is_animating = True
    return O3DVisualizer.TickResult.REDRAW

geoms = [
    # {'name': 'coords', 'geometry': coords},
    # # {'name': 'tangent_coords', 'geometry': tangent_coord},
    # {'name': 'rot_axis', 'geometry': rot_axis, 'material': mat_rot_axis},
    {'name': 'object', 'geometry': object_pcl},
    {'name': 'image_frame', 'geometry': frame_mesh},
    {'name': 'sphere', 'geometry': sphere_mesh, 'material': mat_sphere},
]


# def visualizer_setup(geoms: List[Dict], title='demo', width=1024, height=768, 
#         bg_color= (1.0, 1.0, 1.0, 1.0), bg_image = None, callback=None):

#     gui.Application.instance.initialize()
#     w = O3DVisualizer(title, width, height)
#     w.set_background(bg_color, bg_image)
#     w.show_settings = True
    
    
#     # w.add_action("animate", animate)
#     w.animation_time_step = 0.01
#     w.animation_duration = 10
#     w.set_on_animation_frame(animate)
#     # w.animation_frame_delay=0.1
#     # w.set_on_animation_tick(animate)

#     ### Add mesh to visualizer
#     w.add_geometry("object", object_pcl)
    
#     w.reset_camera_to_default()  # make sure far/near get setup nicely
#     gui.Application.instance.add_window(w)
#     gui.Application.instance.run()
    
#     ### Animate the rotation of the point cloud

visualizer_setup(geoms=geoms, callback=animate)
