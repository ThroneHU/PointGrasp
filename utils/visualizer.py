# coding:utf-8            
# @Time    : 27/01/2024 20:22
# @Author  : Tyrone Chen HU

import open3d as o3d
import numpy as np

import time
from utils.data_process import remove_outliers_3d, pred_to_handle_pcd
from utils.context_analysis import context_analysis, split_mulit_objects, split_plane_object

def visualize_np_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def visualize_grasp_points(grasp_points_list, aabb_list, cluster_list, visulizer):
    for idx in range(len(grasp_points_list)):
        visulizer.add_geometry(cluster_list[idx])
        visulizer.add_geometry(aabb_list[idx])
        visulizer.add_geometry(grasp_points_list[idx])



if __name__ == '__main__':
    points = np.load(r'./datasets/biorob_3d/test/test_arr.npy')
    points = points[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)


    handle_pcd = remove_outliers_3d(pcd, 1500, 0.5)
    cluster_list = split_mulit_objects(handle_pcd, 0.02, 10)
    aabb_list, grasp_points_list, grasp_points_dist_list = context_analysis(cluster_list)
    visualize_grasp_points(grasp_points_list, aabb_list, cluster_list)
