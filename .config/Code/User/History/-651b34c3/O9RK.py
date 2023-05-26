"""
For running in batches on recordings.
"""
import subprocess
import argparse
from enum import Enum, unique

from demo_record import run as run_ove6d_captured_chroma

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

class Params():
    """
    Parameters
    ----------
    obj_id: ObjectIds
        Object cad model to be used for pose estimation and rendering.
    obj_name:
        Recorded object name if ./data/recordings/file_in
    segment_method:
        Segmentation method. e.g. chromakey (green screen)
    group:
        TODO SCENARIOS
    n_triangles: int
        TODO
    segment_method: str
        TODO
    file_in: str
        Input file name from directory ./data/recordings/
    file_out: str
        Output file name to directory ./data/renderings/
    render_mesh: bool
        To render mesh on screen.
    save_rendered: bool
        To save final video of estimation.
    icp: bool =
        To use ICP for OBE6D Pose refinement.
        Internal to OVE6D TODO: Change to open3d.
    """

    obj_id: ObjectIds = ObjectIds.box
    obj_name: str = 'test_box'
    group: str = 'single_object'
    n_triangles: int = 2000
    segment_method: str = 'chromakey'
    file_in: str = 'test'
    file_out: str = 'test_rendered'
    render_mesh: bool = True
    save_rendered: bool = False
    icp: bool = False
    icp_track_max_iters: int = 2
    icp_ove6d_max_iters: int = 2
    buffer_size=1

args = Params()
args.to_save = True
#args.to_save = False 
args.n_triangles = 4000

#args.file_out = ""
#args.file_in ="test"
args.file_in ="demo_videos"

args.group = 'single_object'

#args.obj_name='clipper'
#args.obj_id=ObjectIds.clipper

#args.obj_name='test_box'
#args.obj_id=ObjectIds.box

#args.obj_name = "gear_assembled"
#args.obj_id = ObjectIds.gear_assembled

args.obj_name = "phone_stand"
args.obj_id = ObjectIds.phone_stand

#args.obj_name = "head_phones"
#args.obj_id = ObjectIds.head_phones

#args.obj_name = "pot"
#args.obj_id = ObjectIds.pot

#args.icp_track_max_iters=10
args.icp_ove6d_max_iters=20
args.icp = True

args.segment_method='chromakey'
#segmentation=histogram_plane

#run_ove6d_captured(args)
run_ove6d_captured_chroma(args)
#run_ove6d_icp_point2plane_captured(args)
