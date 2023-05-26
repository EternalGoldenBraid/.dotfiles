import os
from time import perf_counter
import sys
import warnings
import json
from itertools import product
from pathlib import Path
from numba import njit, prange
import cv2
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
import pyrealsense2 as rs
from scipy import stats

import h5py

from ipdb import iex
from matplotlib import pyplot as plt

import mediapy as media

from os.path import join as pjoin
from bop_toolkit_lib import inout
warnings.filterwarnings("ignore")

#breakpoint()
#base_path = os.path.dirname(os.path.abspath("."))
#sys.path.append(base_path)
#
#breakpoint()
from utility import (timeit,  cam_control,
                     #load_segmentation_model,
                     load_segmentation_model_chroma)
from utility.load_pose_estimator import PoseEstimator

from lib.render_cloud import load_cloud, render_cloud # TODO Use this
from lib import ove6d

from dataset import demo_dataset
from configs import config as cfg

#DEVICE = torch.device('cuda')
#DEVICE = torch.device('cpu')
DEVICE = 'cpu'
#DEVICE = 'cuda'

def run(args):

    cfg.DATASET_NAME = 'huawei_box'        # dataset name
    cfg.USE_ICP = args.icp

    # Load data
    grp = args.group
    file_in = f"{args.file_in}.hdf5"
    f = h5py.File(f"data/recordings/{file_in}", "r")
    if args.obj_name not in f[grp]:
        print("Object name not found in recordings:", f[grp].keys())
        return -1
    depth_scale = f['meta']["depth_scale"][:]
    cam_K = torch.tensor(f['meta']["camera_intrinsic"][:])
    fps = f['meta']["framerate"][0]
    
    # Load recordings
    #for args.obj_name in f[grp][args.obj_name]:
    obj_path = Path(f.name, grp, args.obj_name)
    color_frames = f[str(obj_path/'color_frames')][:]
    depth_frames = f[str(obj_path/'depth_frames')][:]
    f.close()

    print("Loaded recordings:")
    print(f"Number of frames: {color_frames.shape[0]}")
    print(f"Recording framerate {fps}")

    # load segmentation module
    if args.segment_method == 'chromakey':
        segmentator = load_segmentation_model_chroma.load(cfg=cfg, device=DEVICE)
    else:
        raise NotImplementedError("Enable segmentator selection (detectron)")
        #segmentator = load_segmentation_model.load(model=args.segment_method, cfg=cfg, device=DEVICE)

    # Load mesh models
    #timeit.log("Loading data.")
    dataroot: Path = Path(os.path.dirname(__file__))/Path(cfg.DATA_PATH)
    #dataroot = Path(os.path.realpath(__file__)).parent.parent/Path(cfg.DATA_PATH)
    dataset = demo_dataset.Dataset(data_dir=dataroot/ cfg.DATASET_NAME, cfg=cfg,
                cam_K=cam_K, cam_height=cfg.RENDER_HEIGHT, cam_width=cfg.RENDER_WIDTH,
                n_triangles=args.n_triangles)

    # Load pose estimation module
    obj_id: int = args.obj_id.value
    codebook_path: Path = dataroot/'object_codebooks'/ cfg.DATASET_NAME / \
        'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR) / \
        'views_{}'.format(str(cfg.RENDER_NUM_VIEWS))

    #timeit.endlog()
    pose_estimator = PoseEstimator(cfg=cfg, cam_K=dataset.cam_K, obj_id=obj_id,
            model_path=Path('checkpoints','OVE6D_pose_model.pth'),
            device=DEVICE, dataset=dataset, codebook_path=codebook_path)

    # TODO Renderer usable for renering mesh 
    # Passing initialized renderer? Implications?
    #dataset.object_renderer = pose_estimator.obj_renderer

    # Initialize write/output parameters
    rendered_frames = np.empty_like(color_frames, dtype=np.uint8)
    n_frames = color_frames.shape[0]
    duration = int(n_frames/fps)
    poses = np.eye(4, 4)[None,...].repeat(repeats=n_frames, axis=0) # In homo form. : ndarray[Any, dtype[float64]]
    
    
    # Streaming loop
    mod_count: int = 0
    #wait_time = int(1000/fps)
    wait_time = int(1)
    
    d_max = 1 # max depth in meters
    count = -1
    saving = False

    try:
        while not saving:
            for frame in range(n_frames):
                # Careful with overflow
                count += 1

                fps_start = perf_counter()
                
                depth_image = depth_frames[count % n_frames]
                color_image = color_frames[count % n_frames].astype(np.uint8)

                depth_image[depth_image*depth_scale > d_max] = 0
                depth_image[depth_image*depth_scale <= 0] = 0

                masks, masks_gpu, scores = segmentator(color_image)

                #breakpoint()
                #breakpoint()
                cv2.imshow('mask', masks[0].astype(float))

                #depth_colormap = cv2.applyColorMap(
                #        cv2.convertScaleAbs(depth_image,
                #                            #alpha=0.03),
                #                            alpha=0.9, beta=0.0),
                #        cv2.COLORMAP_JET) 
                #depth_colormap = (255*((depth_image - d_max)/d_max)).astype(np.uint8)[..., None].repeat(repeats=3,axis=2)
                #depth_colormap = cv2.applyColorMap(
                #        (255*((depth_image - d_max)/d_max)).astype(np.uint8),
                #        cv2.COLORMAP_JET) 
                depth_colormap = cv2.applyColorMap(
                        #(255*depth_image/depth_image.max() ).astype(np.uint8),
                        (255*depth_image/(d_max/depth_scale) ).astype(np.uint8),
                        cv2.COLORMAP_JET) 
                if masks.size != 0:

                    ### TODO: Can we get depth_image dircetly to gpu from sensor and skip gpu --> cpu with <mask>
                    R, t = pose_estimator.estimate_pose(obj_mask=masks_gpu[0][None,...],
                                obj_depth=torch.tensor(
                                    (depth_image*masks[0]*depth_scale).astype(np.float32)).squeeze()[None,...]
                                )

                    ### TODO Multi object support.
                    #obj_depths = torch.tensor([(depth_image*mask*depth_scale).astype(np.float32) for mask in masks])
                    #R, t = pose_estimator.estimate_poses(obj_masks=masks_gpu, scores=scores,
                    #            obj_depths=obj_depths.squeeze())

                    #timeit.endlog()
                    #timeit.log("Rendering.")

                    for transform_idx in range(R.shape[0]):
                        #color_image, done = dataset.render_cloud(obj_id=obj_id, 
                        #        R=R[transform_idx].numpy().astype(np.float32), 
                        #        t=t[transform_idx].numpy()[...,None].astype(np.float32),
                        #        image=color_image)

                        #color_image, done = dataset.render_mesh(obj_id=obj_id, 
                        #         R=R[transform_idx].numpy().astype(np.float32), 
                        #         t=t[transform_idx].numpy()[...,None].astype(np.float32),
                        #         image=color_image.copy())
                        color_image, done = dataset.render_mesh_slow(obj_id=obj_id, 
                                 R=R[transform_idx].numpy().astype(np.float32), 
                                 t=t[transform_idx].numpy()[...,None].astype(np.float32),
                                 image=color_image.copy())

                        #color_image, done = dataset.render_depth(obj_id=obj_id, 
                        #         R=R[transform_idx].numpy().astype(np.float32), 
                        #         t=t[transform_idx].numpy()[...,None].astype(np.float32),
                        #         image=color_image.copy())


                    ### For demo visualization only
                    masked_depth = depth_colormap.copy()
                    masked_depth[masks[0].astype(bool)] = [10, 57, 255]
                    colors_masked = color_image.copy()
                    colors_masked[masks[0].astype(bool)] = [10, 57, 255]

                    images = np.hstack([ 
                        #color_image, 
                        cv2.addWeighted(depth_colormap, 0.7, masked_depth, 0.3, 0) ,
                        cv2.addWeighted(color_image, 0.7, colors_masked, 0.3, 0)
                        #color_image*np.array(masks.sum(axis=0, dtype=np.uint8)[...,None]) 
                        ])
                else:
                    images = np.hstack((color_image, depth_colormap))

                
                cv2.putText(images, f"fps: {(1/(perf_counter()-fps_start)):2f}", (10,10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,0,0), 1)
                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', images)
                key = cv2.waitKey(wait_time)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord('s'):
                    print("Saving")
                    saving = True
                    wait_time = 1

                rendered_frames[count % n_frames] = color_image

    finally:
        
        cv2.destroyAllWindows()
        if args.to_save:
            #path = '/tmp/video.mp4'
            path = f'data/results/{args.obj_name}.mp4'
            print(f". Writing video to {path}")
            print(f"Duration: {duration}")
            #foo = cv2.cvtColor(rendered_frames, cv2.COLOR_BGR2RGB)
            #breakpoint()
            media.write_video(path, rendered_frames[...,::-1], fps=fps)

if __name__=="__main__":
    import argparse

    from enum import Enum, unique
    class ArgTypeMixin(Enum):

        @classmethod
        def argtype(cls, s: str) -> Enum:
            try:
                return cls[s]
            except KeyError:
                raise argparse.ArgumentTypeError(
                    f"{s!r} is not a valid {cls.__name__}")

        def __str__(self):
            return self.name

    @unique
    class ObjectIds(ArgTypeMixin, Enum):
        box = 1
        head_phones = 3
        engine_main = 4
        dual_sphere = 5
        tea = 6
        bolt = 7
        wrench = 8
        lego = 9
        eraser_lowq = 10
        eraser_highq = 11
        #eraser_lowq = 10
        box_synth = 12
        gear_assembled = 13
        clipper = 14
        pot = 15
        phone_stand = 16


    
    parser = argparse.ArgumentParser(prog='demo',
            description='Superimpose rotated pointcloud onto video.')

    parser.add_argument('-o','--object', dest='obj_id',
                        type=ObjectIds.argtype, default=ObjectIds.box,
                        choices=ObjectIds,
                        help='Object names')
    #parser.add_argument('-o', '--object ', dest='obj_id',
    #                    required=False,default='box',
    #                    choices = ['box', 'head_phones', 'engine_main', 'dual_sphere','6', 'box'],
    #                    help='Object names')
    parser.add_argument('-b', '--buffer_size', dest='buffer_size',  
                        type=int, required=False, default=3,
                        help='Frame buffer for smoothing.')
    parser.add_argument('-n', '--n_triangles', dest='n_triangles',
                        type=int, required=False, default=2000,
                        help='Number of triangles for cloud/mesh.')
    parser.add_argument('-s', '--segmentation', dest='segment_method',
                        required=False, default='maskrcnn',
                        choices = ['chromakey','bgs', 'bgs_hsv', 'bgsMOG2', 'bgsKNN', 'contour', 'maskrcnn', 'point_rend'],
                        help="""Method of segmentation.
                        contour: OpenCV based edge detection ...,
                        TODO:
                        """)
    ### Python < 3.9 TODO: Set this up.
    #parser.add_argument('--feature', action='store_true', dest='render_mesh')
    #parser.add_argument('--no-feature', dest='render_mesh', action='store_false')
    #parser.set_defaults(render_mesh=True)
    ### Python >= 3.9
    parser.add_argument('-rm', '--render-mesh', dest='render_mesh', action=argparse.BooleanOptionalAction)
    parser.add_argument('-icp', dest='icp', action=argparse.BooleanOptionalAction)

    
    args = parser.parse_args()

    run(args)
