# coding:utf-8            
# @Time    : 29/01/2024 09:33
# @Author  : Tyrone Chen HU

import numpy as np
from solvers.pn2_inference import inference
import open3d as o3d
from utils.data_process import arr_to_pcd, remove_outliers_3d
from utils.context_analysis import split_mulit_objects, context_analysis
from utils.visualizer import visualize_grasp_points
from utils.constants import classes

def main():
    data_root = r'./datasets/biorob_3d/'
    print('Start inference...')
    handle_points, object_points = inference(data_root, [8, 9, 10]) # classes number
    handle_pcd, object_pcd = arr_to_pcd(handle_points), arr_to_pcd(object_points)
    handle_pcd.paint_uniform_color([0, 0, 1])

    object_pcd.paint_uniform_color([0, 0, 1])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    filtered_handle_pcd = remove_outliers_3d(handle_pcd, 1500, 0.5)
    filtered_object_pcd = remove_outliers_3d(object_pcd, 1500, 0.5)
    cluster_list = split_mulit_objects(filtered_handle_pcd, 0.02, 10)
    aabb_list, grasp_points_list, grasp_points_dist_list = context_analysis(cluster_list, 200)

    print('Distance between two grasp points: ', grasp_points_dist_list)
    visualize_grasp_points(grasp_points_list, aabb_list, cluster_list, vis)
    vis.add_geometry(filtered_object_pcd)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.2, 0.2, 0.2])
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
