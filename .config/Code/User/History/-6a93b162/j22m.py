import open3d as o3d

# Read the image
map_image = o3d.t.io.read_image(o3d.data.JuneauImage().path)
map_resy = map_image.rows
map_resx = map_image.columns
dtype_f = o3d.core.float32
dtype_i = o3d.core.int64
# Make a triangle mesh to frame the image
triangle_mesh = o3d.t.geometry.TriangleMesh()
triangle_mesh.vertex.positions = o3d.core.Tensor(
    [
        [0.0, 0.0, 0.0],  #
        [map_resx, 0.0, 0.0],  #
        [0.0, map_resy, 0.0],  #
        [map_resx, map_resy, 0.0]
    ],
    dtype_f) / max(map_resx, map_resy)
triangle_mesh.triangle.indices = o3d.core.Tensor([[0, 1, 3], [3, 2, 0]],
                                                 dtype_i)
triangle_mesh.vertex.texture_uvs = o3d.core.Tensor(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype_f)

# Add image to frame as the albedo
material = triangle_mesh.material
material.material_name = "defaultLit"
material.texture_maps["albedo"] = map_image
o3d.visualization.draw(triangle_mesh)