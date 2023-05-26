from typing import Dict, List, Tuple

import open3d as o3d
from open3d.visualization import O3DVisualizer
from open3d.visualization import gui, rendering

import numpy as np
from numpy.typing import NDArray
import time
from math import pi


def visualizer_setup(geoms: List[Dict], title='demo', width=640, height=480,
                     bg_color=(1.0, 1.0, 1.0, 1.0), bg_image=None, callback=None):

    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)
    w.set_background(bg_color, bg_image)
    w.show_settings = True

    # w.add_action("animate", animate)
    # w.animation_time_step = 0.01
    w.animation_time_step = 0.1
    w.animation_duration = 1
    w.set_on_animation_frame(callback)
    # w.animation_frame_delay=0.1
    # w.set_on_animation_tick(animate)

    # Add mesh to visualizer
    for g in geoms:
        w.add_geometry(g)
    # w.add_geometry("object", object_pcl)

    w.reset_camera_to_default()  # make sure far/near get setup nicely
    gui.Application.instance.add_window(w)
    gui.Application.instance.run()
    

def create_image_frame(width: int, height: int) -> o3d.t.geometry.TriangleMesh:
   # Read the image
    # map_image = o3d.t.io.read_image(o3d.data.JuneauImage().path)
    # map_resy = map_image.rows
    # map_resx = map_image.columns
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int64
    # Make a triangle mesh to frame the image
    triangle_mesh = o3d.t.geometry.TriangleMesh()
    triangle_mesh.vertex.positions = o3d.core.Tensor(
        [
            [-width/2,  height/2, 0.0], # Left upper
            [-width/2, -height/2, 0.0],  # Left lower
            [ width/2,  height/2, 0.0], # Right upper 
            [ width/2, -height/2, 0.0], # Right lower
        ],
        dtype_f)
    # dtype_f) / max(width, height)
    triangle_mesh.triangle.indices = o3d.core.Tensor([
                                                        [3, 2, 1],
                                                        [0, 1, 2]],
                                                     dtype_i)
    triangle_mesh.vertex.texture_uvs = o3d.core.Tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype_f)

    return triangle_mesh


def draw_image_on_image_frame(frame: o3d.t.geometry.TriangleMesh, frame_size: Tuple[int, int],
                              image: o3d.t.geometry.RGBDImage):

    material = frame.material
    material.material_name = "defaultLit"

    image_size = (image.rows, image.columns)
    if frame_size == image_size:
        # Add image to frame as the albedo
        material.texture_maps["albedo"] = image
    elif frame_size[0] > image_size[0] or frame_size[1] > image_size[1]:
        raise NotImplementedError("Not sure about this stuff yet")
        start_w = (frame_size[1] - image_size[0])//2
        start_h = (frame_size[0] - image_size[0])//2
        
        output: NDArray = np.zeros((*frame_size, image.channels), np.uint8)
        output[start_w:start_w+image_size[0], start_h:start_h+image_size[1]] = image.as_tensor()
        material.texture_maps["albedo"] = o3d.t.geometry.Image(output)

