import open3d as o3d
from open3d.core import Tensor, float32

import numpy as np

from numpy.typing import NDArray

def convert_material_record(mat_record):
    mat = o3d.visualization.Material('defaultLit')
    # Convert scalar parameters
    mat.vector_properties['base_color'] = mat_record.base_color
    mat.scalar_properties['metallic'] = mat_record.base_metallic
    mat.scalar_properties['roughness'] = mat_record.base_roughness
    mat.scalar_properties['reflectance'] = mat_record.base_reflectance
    # mat.texture_maps['albedo'] = o3d.t.geometry.Image.from_legacy(
    #     mat_record.albedo_img)
    # mat.texture_maps['normal'] = o3d.t.geometry.Image.from_legacy(
    #     mat_record.normal_img)
    # mat.texture_maps['ao_rough_metal'] = o3d.t.geometry.Image.from_legacy(
    #     mat_record.ao_rough_metal_img)
    return mat

def load_object_pointcloud(device: o3d.core.Device, n_points: int = 2048):

    ### Read mesh into pcl
    mesh = o3d.t.io.read_triangle_mesh("models/obj_000003.ply", print_progress=True)
    # In mm
    mesh.vertex.positions = (mesh.vertex.positions-mesh.vertex.positions.mean(dim=0))*1000.
    # mesh.vertex.positions = (mesh.vertex.positions-mesh.vertex.positions.mean(dim=0))
    
    mesh_pcl = o3d.t.geometry.PointCloud(device)
    mesh_pcl.point.positions = mesh.vertex.positions
    # Downsample to n_points

    ratio: float = n_points/len(mesh_pcl.point.positions)
    if ratio < 1.0 and ratio > 0.0:
        mesh_pcl = mesh_pcl.random_down_sample(sampling_ratio=ratio)
    mesh_pcl = mesh_pcl.paint_uniform_color(Tensor([1.0, 0.0, 0.0], float32))
    # mesh_pcl.paint_uniform_color([0.5, 0, 0.1])
    
    return mesh_pcl

def create_sphere_pcl(n_pts_sphere=100, coords=(0,0,0)):

    sphere_pts = []
    for i in range(n_pts_sphere):
        for j in range(n_pts_sphere):
            x = 100*np.cos(np.pi * i / 100) * np.cos(2 * np.pi * j / 100)
            y = 100*np.cos(np.pi * i / 100) * np.sin(2 * np.pi * j / 100)
            z = 100*np.sin(np.pi * i / 100)
            sphere_pts.append([x, y, z])
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_pts)
    sphere_pcd.translate(coords, relative=True)
    
    return sphere_pcd

def create_sphere_mesh(coords: tuple =(0,0,0), scale: float =1., radius=1.0):
    """
    Create TriangleMesh sphere and PBR materials.
    Return mesh and materials.
    """
    mat_sphere = o3d.visualization.rendering.MaterialRecord()
    # mat_sphere.shader = 'defaultLitTransparency'
    mat_sphere.shader = 'defaultLitSSR'
    mat_sphere.base_color = [0.467, 0.467, 0.467, 0.2]
    mat_sphere.base_roughness = 0.0
    mat_sphere.base_reflectance = 0.0
    mat_sphere.base_clearcoat = 1.0
    mat_sphere.thickness = 1.0
    mat_sphere.transmission = 1.0
    mat_sphere.absorption_distance = 100
    mat_sphere.absorption_color = [0.5, 0.5, 0.5]

    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, 
                        create_uv_map=True)
    # sphere_mesh = o3d.t.geometry.TriangleMesh.from_legacy(sphere_mesh)
    # sphere_mesh.compute_vertex_normals()
    sphere_mesh.translate(coords)

    # mat_sphere = convert_material_record(mat_record = mat_sphere)
    # sphere_mesh.material = mat_sphere
    # sphere_mesh.vertices = o3d.utility.Vector3dVector(
    #     (np.asarray(sphere_mesh.vertices) - np.asarray(sphere_mesh.vertices).mean(axis=0))*scale)
    
    return sphere_mesh, mat_sphere

def setup_canvas(width=640, height=480, color=(0,0,0)):
    plane_pts = np.array([[x, y, 0] for x in range(width) for y in range(height)])
    plane_pts = plane_pts - [width//2, height//2, 0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plane_pts)

    return pcd

def get_axis_angle_from_matrix(R: np.array):

    angle = np.arccos((np.trace(R)-1)/2)
    u_skew_sym = 1/(2*np.sin(angle))*(R-R.T) # https://thenumb.at/Exponential-Rotations/
    
    u = 1/(2*np.sin(angle))*(
        np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0] - R[0,1]])) # https://thenumb.at/Exponential-Rotations/
    
    return (u, angle)

import math
def get_rotating_transform(u, v):
    """
    Given two unit vectors u, v,
    we compute the rotation R such that v = R @ u
    """
    assert np.isclose(np.linalg.norm(u), 1)
    assert np.isclose(np.linalg.norm(v), 1)
    n = np.cross(u, v)
    while np.allclose(n, 0): # u and v are on the same line -.-
        rand_dir = np.random.randn(3)
        n = np.cross(u, rand_dir)
    n /= np.linalg.norm(n)
    t = np.cross(n, u)
    T = np.stack([u, t, n], axis=1)
    alpha = math.atan2(v @ t, v @ u)
    R = np.array([
        [math.cos(alpha), -math.sin(alpha), 0],
        [math.sin(alpha), math.cos(alpha), 0],
        [0, 0, 1]
    ])
    transform = T @ R @ T.T
    assert np.allclose(v, transform @ u)
    return transform