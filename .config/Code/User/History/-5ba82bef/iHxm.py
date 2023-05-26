import open3d as o3d
import numpy as np


def setup_canvas(width=640, height=480, color=(0,0,0)):
    plane_pts = np.array([[x, y, 0] for x in range(width) for y in range(height)])
    plane_pts = plane_pts - [width//2, height//2, 0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plane_pts)

    return pcd

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
    sphere_pcd.translate((0, 0, z_init), relative=True)
    
    return sphere_pcd

def create_sphere_mesh(coords: tuple =(0,0,0), scale: float =1.):
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

    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere()
    sphere_mesh.compute_vertex_normals()
    sphere_mesh.translate((0,0,z_init))
    sphere_mesh.vertices = o3d.utility.Vector3dVector(
        (np.asarray(sphere_mesh.vertices) - np.asarray(sphere_mesh.vertices).mean(axis=0))*s)
    
    return sphere_mesh, mat_sphere

### Setup canvas
# height = 480
# width = 640
height = 255
width = 255
plane_pcd = setup_canvas(width=width, height=height)

# coords = o3d.geometry.create_mesh_coordinate_frame(size=1.0, origin=(0,0,0))
# coords = o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=1.0, origin=(0,0,0))
coords = o3d.geometry.TriangleMesh().create_coordinate_frame(size=40, origin=(0,0,0))

### Add mesh
z_init = 1000
s=1000.

sphere_mesh, mat_sphere = create_sphere_mesh(coords=(0,0,z_init), scale=s)

mesh = o3d.io.read_triangle_mesh("models/obj_000004.ply")
# Scale and center
mesh.vertices = o3d.utility.Vector3dVector(
    (np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(axis=0))*s)
# Lift
mesh.translate((0, 0, z_init), relative=True)
initial_coords = np.array(mesh.vertices)

# sphere_pcl = create_sphere_pcl(n_pts_sphere=100, coords=(0,0,z_init))

### Projection
K = np.eye(4)

n_steps_max=100
# height_at_time = lambda t, z_0=z_init, n_steps_max=n_steps_max: z_0*(1-t/n_steps_max)
height_at_time = lambda t, z_0=z_init, n_steps_max=n_steps_max: z_0*(1-t)
t = 0
step_size_metric = z_init/n_steps_max
step_size_temporal = step_size_metric/n_steps_max
# Function to animate the mesh being projected onto the plane
def animate_projection(vis, mesh=mesh, K=K, initial_coords=initial_coords):
    
    # Move the mesh closer to the plane
    # mesh.translate((0, 0, -1), relative=True)
    global t
    z = height_at_time(t)
    print(f"time: {t}, z: {z}")
    mesh.translate((0, 0, height_at_time(t)), relative=False)
    # t = t + step_size_temporal
    t = t + 1/n_steps_max

    # If the mesh has reached the plane, stop the animation
    if mesh.get_min_bound()[2] <= 0:
        vis.register_animation_callback(None)

    # Transform the mesh using the projection matrix
    mesh.transform(K)
    verts = np.array(mesh.vertices)
    # mesh.vertices = o3d.utility.Vector3dVector( verts/(t*verts[:,-1][..., None]))
    mesh.vertices = o3d.utility.Vector3dVector( verts/((1-t)*initial_coords[...]))


    # Update the visualization
    vis.update_geometry(mesh)


# Create a visualizer and set the background color to white
vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.get_render_option().background_color = np.ones(4)

# Add the point cloud and the mesh to the visualizer
geoms = [
    {'name': 'coords', 'geometry': coords},
    {'name': 'plane', 'geometry': plane_pcd},
    {'name': 'object', 'geometry': mesh},
    {'name': 'sphere', 'geometry': sphere_mesh, 'material': mat_sphere},

]
# vis.add_geometry(coords)
# vis.add_geometry(plane_pcd)
# vis.add_geometry(mesh)
# # vis.add_geometry(sphere_pcl)
# vis.add_geometry(sphere_mesh)

o3d.visualization.draw(geoms)

# Register the animate_projection function as the animation callback
vis.register_animation_callback(animate_projection)

# Run the visualizer
vis.run()