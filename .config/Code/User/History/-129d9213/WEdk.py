import open3d as o3d
import numpy as np
from scipy.optimize import least_squares

# Define the 3D points in world coordinates
points_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

# Define the 2D points in image coordinates
points_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the camera intrinsic matrix
K = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])

# Define the initial camera extrinsic parameters
R = np.eye(3)
t = np.zeros(3)

# Define a function to project the 3D points onto the image plane
def project(R, t, K, points_3d):
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_2d_homo = np.dot(K, np.dot(R, points_3d_homo.T) + t)
    points_2d = points_2d_homo[:2, :].T / points_2d_homo[2, :]
    return points_2d

# Define the optimization objective function
def objective(x, points_3d, points_2d):
    # Unpack the angle-axis vector and translation vector
    theta = x[:3]
    t = x[3:]
    
    # Compute the rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(theta)
    
    # Project the 3D points onto the image plane
    points_2d_pred = project(R, t, K, points_3d)
    
    # Compute the difference between the predicted and observed 2D points
    diff = points_2d_pred - points_2d
    diff = diff.reshape(-1)
    return diff

# Define the initial guess for the camera parameters
x0 = np.zeros(6)

# Use the Levenberg-Marquardt optimization method to find the optimal camera parameters
res = least_squares(objective, x0, args=(points_3d, points_2d), method='lm')

# Unpack the optimized camera parameters
theta_opt = res.x[:3]
t_opt = res.x[3:]