def plane_to_camera_dist(plane_pcd, intrinsic):
    intrinsic_tensor = o3d.core.Tensor(intrinsic.intrinsic_matrix)
    plane_arr = np.asarray(plane_pcd.points)
    plane_pcd_tensor = o3d.t.geometry.PointCloud(o3d.core.Tensor(plane_arr.astype(np.float32)))
    plane_depth_image = plane_pcd_tensor.project_to_depth_image(640, 480, intrinsic_tensor)

    return np.mean(np.asarray(plane_depth_image))
