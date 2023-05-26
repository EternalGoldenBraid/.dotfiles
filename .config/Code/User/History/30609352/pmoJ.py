import cv2
import numpy as np
import open3d as o3d


if __name__ == '__main__':

    def get_intrinsic(width, height, focal_dist=6064):
        return o3d.core.Tensor([[focal_dist, 0     , width * 0.5], 
                                [0     , focal_dist, height * 0.5],
                                [0     , 0     , 1]])

    def get_extrinsic(x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0):
        extrinsic = np.eye(4)
        extrinsic[:3,  3] = (x, y, z)
        extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(np.radians(np.asarray((rx, ry, rz))))
        return extrinsic

    def compute_show_reprojection(pcd, width, height, intrinsic, extrinsic, depth_max=10.0, depth_scale=15.0, window_wait=3000):
        depth_reproj = pcd.project_to_depth_image(width,
                                                  height,
                                                  intrinsic,
                                                  extrinsic,
                                                  depth_scale=depth_scale,
                                                  depth_max=depth_max)

        
        depth_mat = np.asarray(depth_reproj.to_legacy())
        cv2.imshow("depth", depth_mat)

        return cv2.waitKey(window_wait), depth_mat
        
    
    width, height = 640, 480
    
    points = np.random.rand(100, 3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    pcd = o3d.t.geometry.PointCloud.from_legacy(point_cloud)

    key = ' '
    pcd = pcd.cuda()
    print('pcd.is_cuda', pcd.is_cuda)
    
    x, y, z, = -0.44999999999999996, -0.5, 300.0
    depth_max = 10000.0
    depth_scale = 15.0
    intrinsic_scale = 1.0
    c_scale = 120000
    rx = ry = rz = 0
    
    while key != ord('q'):
        # update intrinsic, extrinsic matrices
        intrinsic = get_intrinsic(width * intrinsic_scale, height * intrinsic_scale, focal_dist = c_scale)
        extrinsic = get_extrinsic(x, y, z, rx, ry, rz)
        # reproject (and display) depth map from point cloud
        key, depth_img = compute_show_reprojection(pcd.cpu(), width, height, intrinsic, extrinsic, \
                                                   depth_max=depth_max, depth_scale=depth_scale, window_wait=40)
        
        if key == ord('x'):
            x -= 0.1
            print(f"x, y, z = {x, y, z}")
    
        if key == ord('X'):
            x += 0.1
            print(x, y, z)
    
        if key == ord('y'):
            y -= 0.1
            print(f"x, y, z = {x, y, z}")
    
        if key == ord('Y'):
            y += 0.1
            print(f"x, y, z = {x, y, z}")
    
        if key == ord('z'):
            z -= 0.1
            print(f"x, y, z = {x, y, z}")
    
        if key == ord('Z'):
            z += 0.1
            print(f"x, y, z = {x, y, z}")

        if key == ord('m'):
            depth_max -= 0.1
            print('depth_max', depth_max)
    
        if key == ord('M'):
            depth_max += 0.1
            print('depth_max', depth_max)
    
    
        if key == ord('s'):
            depth_scale -= 0.1
            print('depth_scale', depth_scale)
    
        if key == ord('S'):
            depth_scale += 0.1
            print('depth_scale', depth_scale)
            # cv2.imwrite('depth_reproj.png', depth_img)
    
        if key == ord('i'):
            intrinsic_scale -= 0.1
            print('intrinsic_scale', intrinsic_scale)
        if key == ord('I'):
            intrinsic_scale += 0.1
            print('intrinsic_scale', intrinsic_scale)

        if key == ord('c'):
            c_scale -= 100
            print('c_scale', c_scale)

        if key == ord('C'):
            c_scale += 100
            print('c_scale', c_scale)

        if key == ord('u'):
            rx += 10    
    
        if key == ord('j'):
            rx -= 10  

        if key == ord('i'):
            ry += 10    
    
        if key == ord('k'):
            ry -= 10  

        if key == ord('o'):
            rz += 10    
    
        if key == ord('l'):
            rz -= 10    
    