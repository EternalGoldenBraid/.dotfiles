import open3d as o3d

# Create the mesh
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

# Create the window and get the unique ID of the O3DVisualizer object
vis_uid = o3d.visualization.draw([{'name': 'object',  'geometry': mesh}],
                                  non_blocking_and_return_uid=True)

# Get a reference to the O3DVisualizer object using the unique ID
vis = o3d.visualization.gui.get_visualizer(vis_uid)

# Render and save an image of the window
vis.take_screen_shot('image.png')
