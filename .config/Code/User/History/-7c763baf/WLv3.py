from typing import Dict, List, Tuple, Any
from threading import Thread

import open3d as o3d
from open3d.visualization import O3DVisualizer
from open3d.visualization import gui, rendering

import numpy as np
from numpy.typing import NDArray
import time
from math import pi

from apple_pygatt.examples.basic import MyWatchManager as Watch

class Viewer3D(object):
    """
    TODO: Why object?
    From: https://github.com/isl-org/Open3D/issues/5501
    """
    
    def __init__(self, title: str ='demo', width: int =640, height: int = 480,
                bg_color=(1.0, 1.0, 1.0, 1.0), bg_image=None, raw_mode: bool = False,
                show_skybox: bool = True):
        
        self.apple_watch_thread: Any = None
        
        self.geoms: List[Dict] = []
        app = gui.Application.instance
        app.initialize()
        
        self.main_vis: O3DVisualizer = O3DVisualizer(title, width, height)
        app.add_window(self.main_vis)
        self.main_vis.set_background(bg_color, bg_image)
        self.main_vis.show_settings = True
        
        # TODO: What are these?
        self.setup_depth_streaming()
        # self.setup_o3d_scene()

    def setup_depth_streaming(self):
        # TODO: setup your depth / point cloud streaming source here
        pass

    def setup_point_clouds(self, geoms: List[Dict]) -> None:
        # for geom in geoms: self.geoms.append(geom) 
        for geom in geoms: self.main_vis.add_geometry(geom)
        self.main_vis.reset_camera_to_default()
        
    def setup_watch(self):
        self.watch = Watch()
        self.watch_thread = Thread(target=self.watch.start)
        self.watch_thread.start()

        # TODO: Figure out how to use self.client in watch_manager.py to check for is_Connected.
        # print("Connecting")
        # while not self.watch.client.is_connected:
            # pass
        print("Apple watch setup done")
            # raise RuntimeError("Could not connect to watch.")
        
    def update_point_clouds(self):
        # update your point cloud data here: convert depth to point cloud / filter / etc.
        pass

    def setup_o3d_scene(self) -> None:
        # center, eye, up
        self.main_vis.setup_camera(60,
                                    [4, 2, 5],
                                    [0, 0, -1.5],
                                    [0, 1, 0])
        
    def run_one_tick(self) -> bool:
        "TODO: Change to run in thread. See:http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.Application.html#open3d.visualization.gui.Application.run_in_thread"
        app = gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return


def visualizer_setup(geoms: List[Dict], title='demo', width=640, height=480,
                     bg_color=(1.0, 1.0, 1.0, 1.0), bg_image=None, callback=None,
                     raw_mode: bool = False, show_skybox: bool = True):

    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)
    w.set_background(bg_color, bg_image)
    w.show_settings = True
    
    if raw_mode == True:
        w.enable_raw_mode(True)
    if show_skybox == True:
        w.show_skybox(True)

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

