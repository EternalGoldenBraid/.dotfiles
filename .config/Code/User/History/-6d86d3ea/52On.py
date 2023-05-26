import time

import open3d as o3d
import numpy as np
import geomstats as gm
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


from utils import (create_sphere_pcl, create_sphere_mesh,
                    setup_canvas, get_axis_angle_from_matrix)

### Setup canvas
z_init = 1000
s=1000.
# height = 480
# width = 640

# Create tangent plane
height = 255
width = 255
plane_pcd = setup_canvas(width=width, height=height)
plane_pcd.translate((0,0,s))
tangent_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=100, origin=(0,0,s))

# coords = o3d.geometry.create_mesh_coordinate_frame(size=1.0, origin=(0,0,0))
# coords = o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=1.0, origin=(0,0,0))
coords = o3d.geometry.TriangleMesh().create_coordinate_frame(size=40, origin=(0,0,0))

# Create an instance of the SpecialOrthogonal group SO(3)
SO3 = SpecialOrthogonal(3)

rot_point = o3d.geometry.PointCloud()
rot_point.points = o3d.utility.Vector3dVector([[0.,0.,1000.]])
rot_axis = o3d.geometry.TriangleMesh().create_arrow(
    cylinder_radius=10, cylinder_height=200.,
    cone_radius=10., cone_height=20.)
mat_rot_axis = o3d.visualization.rendering.MaterialRecord()
mat_rot_axis.base_color = [1., 0., 0., 0.5]
rot_axis.translate((0,0,s))

### Add mesh

# sphere_mesh, mat_sphere = create_sphere_mesh(coords=(0,0,z_init), scale=s)
radius = s
sphere_mesh, mat_sphere = create_sphere_mesh(coords=(0,0,0), radius=radius)

mesh = o3d.io.read_triangle_mesh("models/obj_000004.ply")
# Scale and center
mesh.vertices = o3d.utility.Vector3dVector(
    (np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(axis=0))*s)

R = SO3.random_uniform()
axis, angle = get_axis_angle_from_matrix(R)
print(f"axis: {axis}, angle: {angle}")
### Animate
K = np.eye(3)
from utils import get_rotating_transform as gR
def animate_projection(vis, mesh=mesh, K=K):

    # Sample rotation
    global SO3
    R = SO3.random_uniform()
    axis, angle = get_axis_angle_from_matrix(R)
    R_ = gR(u=rot_point.points/np.linalg.norm(rot_point.points), v=axis/np.linalg.norm(axis))
    print(f"axis: {axis}, angle: {angle}")
    
    # Move the mesh closer to the plane
    # mesh.translate((0, 0, -1), relative=True)
    # Transform the mesh using the projection matrix
    mesh.rotate(R)
    rot_axis.rotate(R_)
    # rot_axis.verts = o3d.utility.Vector3dVector(SO3.right_translate_by_rotation(rot_axis.vertices, R))
    # verts = np.array(mesh.vertices)
    # mesh.vertices = o3d.utility.Vector3dVector( verts/(t*verts[:,-1][..., None]))
    # mesh.vertices = o3d.utility.Vector3dVector( verts/((1-t)*initial_coords[...]))


    # Update the visualization
    vis.update_geometry(mesh)
    vis.update_geometry(rot_axis)
    time.sleep(0.1)

# Create a visualizer and set the background color to white
vis = o3d.visualization.Visualizer()
# vis.get_render_option().background_color = np.ones(4)
vis.create_window()


# vis.create_window()
# vis.add_geometry(coords)
# vis.add_geometry(plane_pcd)
# vis.add_geometry(mesh)
# vis.add_geometry(rot_axis)
# # vis.add_geometry(sphere_pcl)
vis.add_geometry(sphere_mesh)
# geoms = [
#     {'name': 'coords', 'geometry': coords},
#     # {'name': 'tangent_coords', 'geometry': tangent_coord},
#     {'name': 'rot_axis', 'geometry': rot_axis, 'material': mat_rot_axis},
#     {'name': 'plane', 'geometry': plane_pcd},
#     {'name': 'object', 'geometry': mesh},
#     {'name': 'sphere', 'geometry': sphere_mesh, 'material': mat_sphere},
# ]
# vis.add_geometry(geoms[-1])

# Register the animate_projection function as the animation callback
# vis.register_animation_callback(animate_projection)

# Run the visualizer
vis.run()