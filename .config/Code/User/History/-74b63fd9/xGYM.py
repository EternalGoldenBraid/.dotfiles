"""
This script computes the optimal SE(3) transformation between corresponding 2D and 3D points using non-linear optimiziation.
"""
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import scipy

from optimization_utils import (project, f, optimize,
                                g1, g2, g3, kkt)

# Define the 3D points in world coordinates
points_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

# Define the 2D points in image coordinates
points_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the camera intrinsic matrix
K = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])

# Define the initial camera extrinsic parameters
R = np.eye(3)
t = np.zeros(3)

# Define the initial guess
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Define the bounds
bnds = ((-np.inf, np.inf), (-1, 1), (-1, 1), (-np.inf, np.inf), (-np.inf, np.inf))

# Define the constraints
cons = ({'type': 'eq', 'fun': g1}, {'type': 'eq', 'fun': g2}, {'type': 'ineq', 'fun': g3})

# Perform the optimization
# res = scipy.optimize.minimize(f, x0, args=(points_3d, points_2d), 
#     # method='SLSQP',
#     method='lm',
#     bounds=bnds, constraints=cons)

res = optimize(f=f, x0=x0, points_3d=points_3d, points_2d=points_2d, K=K)

# Print the results
print(res)

# Unpack the angle-axis vector and translation vector
theta = res.x[0]
axis = res.x[1:3]
t = res.x[3:]

# Compute the rotation matrix
R = o3d.geometry.get_rotation_matrix_from_axis_angle(theta*axis)

# Print the rotation matrix
print(R)

# Print the translation vector
print(t)

# Project the 3D points onto the image plane
points_2d_pred = project(R, t, K, points_3d)

# Print the predicted 2D points
print(points_2d_pred)

# Print the observed 2D points
print(points_2d)

# Plot the 3D points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
o3d.visualization.draw_geometries([pcd])

# Plot the 2D points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_2d)
o3d.visualization.draw_geometries([pcd])

# Plot the predicted 2D points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_2d_pred)
o3d.visualization.draw_geometries([pcd])

