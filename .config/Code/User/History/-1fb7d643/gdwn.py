"""
This script computes the optimal SE(3) transformation between corresponding 2D and 3D points using non-linear optimiziation.
"""
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import scipy

def project(R: NDArray, t: NDArray, K: NDArray, points_3d: NDArray) -> NDArray:
    """
    This function projects 3D points onto the image plane.
    """
    # Compute the projection matrix
    P = K @ np.hstack([R, t.reshape(-1, 1)])
    
    # Convert the 3D points to homogeneous coordinates
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # Project the 3D points onto the image plane
    points_2d = P @ points_3d_h.T
    points_2d = points_2d.T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    
    return points_2d

# Define the optimization objective function
def f(x: NDArray, points_3d, points_2d, K):
    """
    This function computes the difference between the predicted and observed 2D points.
    """
    # Unpack the angle-axis vector and translation vector
    theta = x[0]
    axis = x[1:3]
    t = x[3:]
    
    # Compute the rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(theta*axis)
    
    # Project the 3D points onto the image plane
    points_2d_pred = project(R, t, K, points_3d)
    
    # Compute the difference between the predicted and observed 2D points
    diff = points_2d_pred - points_2d
    diff = diff.reshape(-1)
    return diff

# Define the constraint functions
def g1(x: NDArray):
    """
    This function computes the constraint that the angle-axis vector must have unit norm.
    """
    return np.linalg.norm(x[1:3]) - 1

def g2(x: NDArray):
    """
    This function computes the constraint that the translation vector must have unit norm.
    """
    return np.linalg.norm(x[3:]) - 1

# Define the angle constraint
def g3(x: NDArray):
    """
    This function computes the constraint that the angle must be between 0 and pi/2.
    """
    return x[0] - np.pi/2

# Define KKT conditions
def kkt(x: NDArray, points_3d, points_2d, K):
    """
    This function computes the KKT conditions.
    """
    # Compute the Jacobian of the objective function
    J = scipy.optimize.approx_fprime(x, f, 1e-8, points_3d, points_2d, K)
    
    # Compute the Jacobian of the constraint functions
    J1 = scipy.optimize.approx_fprime(x, g1, 1e-8)
    J2 = scipy.optimize.approx_fprime(x, g2, 1e-8)
    J3 = scipy.optimize.approx_fprime(x, g3, 1e-8)
    
    # Compute the KKT conditions
    kkt = np.hstack([J, J1, J2, J3])
    return kkt

def optimize(f, x0, points_3d, points_2d, K):
    """
    This function optimizes the objective function using the Modified Newton algorithm.
    """

    # Define the initial guess
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Define the bounds on the variables
    bounds = scipy.optimize.Bounds([0.0, -1.0, -1.0, -1.0, -1.0, -1.0], [np.pi/2, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Define the constraints
    # constraints = scipy.optimize.NonlinearConstraint(kkt, 0, 0, jac='2-point', args=(points_3d, points_2d, K))

    # Run the optimization
    max_iter: int = 100
    
    # Define the initial step size
    alpha: float = 1.0

    # Define the initial guess
    x: NDArray = x0

    # Define the initial objective function value
    f0: float = f(x, points_3d, points_2d, K)

    # Define the initial gradient
    g: NDArray = scipy.optimize.approx_fprime(x, f, 1e-8, points_3d, points_2d, K)

    # Define the initial Hessian
    H: NDArray = scipy.optimize.approx_fprime(x, lambda x: scipy.optimize.approx_fprime(x, f, 1e-8, points_3d, points_2d, K), 1e-8, points_3d, points_2d, K)
    
    # Define the initial KKT conditions
    kkt0: NDArray = kkt(x, points_3d, points_2d, K)

    # Define the initial step direction
    d: NDArray = -np.linalg.solve(H, g)
    
    # Define min loss
    min_loss: float = 1e-6

    # Define the initial loss
    loss: float = np.linalg.norm(kkt0)
    
    iter: int = 0
    while loss > min_loss:
        if iter > max_iter:
            break

        # Compute the step size
        # alpha = scipy.optimize.line_search(f, lambda x: scipy.optimize.approx_fprime(x, f, 1e-8, points_3d, points_2d, K), x, d, g, alpha)[0]
        alpha = scipy.optimize.fminbound(lambda alpha: f(x + alpha*d, points_3d, points_2d, K), 0, 2)
        

    return res