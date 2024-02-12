# coding:utf-8            
# @Time    : 27/01/2024 22:37
# @Author  : Tyrone Chen HU

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

import matplotlib.pyplot as plt
import cv2
from utils.data_process import remove_outliers_3d, crop_matrix, mean_filter_edges
from utils.context_analysis import context_analysis, split_mulit_objects, split_plane_object
from utils.visualizer import visualize_grasp_points
from utils.constants import ALL_SIM_OBJECT
from utils.measure import plane_to_camera_dist
from utils.tools import dict_to_csv

REAL_TIME = False


OUT_PATH = r'D:\Kings\Projects\Point Cloud Context Analysis for Rehabilitationi Grasping Assistance\Exprimental results'
OBJ_NAME = '053_mini_soccer_ball'

def main():
    # Create RealSense pipline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # start pipeline
    profile = pipeline.start(config)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    results_dict = {'grasp_points_dist':[], 'camera_dist':[]}
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_scale = device.query_sensors()[0].get_option(rs.option.depth_units)
            depth_image_np = np.asanyarray(depth_frame.get_data()) * depth_scale
            color_image_np = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense RGB', color_image_np)

            depth_image = crop_matrix(depth_image_np, 80)
            depth_image = mean_filter_edges(depth_image, 50, 11)
            depth_image = np.pad(depth_image, 80, 'constant', constant_values=0)
            # depth_image_o3d = o3d.t.geometry.Image(depth_image.astype(np.float32))
            depth_image_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
            o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_intrinsics.width, depth_intrinsics.height,
                                                                 depth_intrinsics.fx, depth_intrinsics.fy,
                                                                 depth_intrinsics.ppx, depth_intrinsics.ppy)
            intrinsic_tensor = o3d.core.Tensor(o3d_camera_intrinsic.intrinsic_matrix)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, o3d_camera_intrinsic)

            object_pcd, plane_pcd = split_plane_object(pcd, 0.01, 40, 1000)

            filtered_object_pcd = remove_outliers_3d(object_pcd, 1500, 0.5)

            cluster_list = split_mulit_objects(filtered_object_pcd, 0.02, 10)
            aabb_list, grasp_points_list, grasp_points_dist_list = context_analysis(cluster_list)
            visualize_grasp_points(grasp_points_list, aabb_list, cluster_list)

            # vis.add_geometry(inlier_cloud)
            plane_arr = np.asarray(object_pcd.points)
            plane_tensor = o3d.t.geometry.PointCloud(o3d.core.Tensor(plane_arr.astype(np.float32)))
            plane_depth_image = plane_tensor.project_to_depth_image(640, 480, intrinsic_tensor)
            camera_dist = plane_to_camera_dist(plane_depth_image)
            results_dict['grasp_points_dist'].append(np.round(np.float32(grasp_points_dist_list*100), 4))
            results_dict['camera_dist'].append(np.round(camera_dist/10, 4))

            if REAL_TIME:
                for label in range(cluster_list):
                    vis.update_geometry(cluster_list[label])
                    vis.update_geometry(aabb_list[label])
                    vis.update_geometry(grasp_points_list[label])
                vis.poll_events()
                vis.update_renderer()
                # vis.register_key_callback(256, on_key_press)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    vis.close()
                    break
            else:
                vis.run()
                vis.destroy_window()

        dict_to_csv(results_dict, OUT_PATH, OBJ_NAME)

    finally:
        # Turn off RealSense stream
        # dict_to_csv(results_dict, OUT_PATH, OBJ_NAME)
        pipeline.stop()

if __name__ == '__main__':
    main()
