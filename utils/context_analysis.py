# coding:utf-8            
# @Time    : 27/01/2024 20:30
# @Author  : Tyrone Chen HU

import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter
import pandas as pd


def save_npy(filename, arr):
    np.save(filename, arr)

def save_excel(filename, result_dict):
    df = pd.DataFrame(result_dict)
    df.to_excel(filename, index=False)

def arr_to_pcd(arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def pcd_to_arr_d(pcd):
    arr = np.asarray(pcd.points)
    return arr

def pcd_to_arr_rgbd(pcd):
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    xyzrgb = np.concatenate((xyz, rgb), axis=1)
    return xyzrgb

def remove_outliers_3d(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_points = pcd.select_by_index(ind)
    filtered_points.paint_uniform_color([0, 0, 1])
    return filtered_points


def pred_to_handle_pcd(pred_label, handle_num, whole_scene_data):
    handle_index = np.where(pred_label == handle_num)
    handle_points = whole_scene_data[handle_index]
    handle_points = handle_points[:, 0:3]
    handle_pcd = arr_to_pcd(handle_points)
    return handle_pcd

def crop_matrix(matrix, n):
    if len(matrix.shape) not in [2, 3]:
        raise ValueError("Matrix must be either 2D or 3D.")

    if n >= min(matrix.shape[:2]):
        raise ValueError("Invalid crop size. Crop size should be smaller than "
                         "both dimensions of the matrix.")

    if len(matrix.shape) == 2:  # gray
        rows, cols = matrix.shape
        cropped_matrix = matrix[n:rows-n, n:cols-n]
    else:  # RGB
        rows, cols, _ = matrix.shape
        cropped_matrix = matrix[n:rows-n, n:cols-n, :]

    return cropped_matrix


def mean_filter_edges(matrix, n, filter_size):
    # Check if the matrix is either 2D or 3D
    if len(matrix.shape) not in [2, 3]:
        raise ValueError("Matrix must be either 2D or 3D.")

    # Get the dimensions of the matrix
    if len(matrix.shape) == 2:  # For grayscale images
        rows, cols = matrix.shape
    else:  # For RGB images
        rows, cols, _ = matrix.shape

    # Ensure n is smaller than the dimensions of the matrix
    if n > rows or n > cols:
        raise ValueError("Invalid value of n. It should be smaller than the dimensions of the matrix.")

    # Copy the original matrix
    filtered_matrix = matrix.copy()

    # Apply mean filter to the edges of the image
    if len(matrix.shape) == 2:
        # Apply filter to rows
        filtered_matrix[:n, :] = uniform_filter(filtered_matrix[:n, :], size=filter_size, mode='reflect')
        filtered_matrix[-n:, :] = uniform_filter(filtered_matrix[-n:, :], size=filter_size, mode='reflect')
        # Apply filter to columns
        filtered_matrix[:, :n] = uniform_filter(filtered_matrix[:, :n], size=filter_size, mode='reflect')
        filtered_matrix[:, -n:] = uniform_filter(filtered_matrix[:, -n:], size=filter_size, mode='reflect')
    else:
        # Apply filter to each color channel separately
        for c in range(matrix.shape[2]):
            filtered_matrix[:n, :, c] = uniform_filter(filtered_matrix[:n, :, c], size=filter_size, mode='reflect')
            filtered_matrix[-n:, :, c] = uniform_filter(filtered_matrix[-n:, :, c], size=filter_size, mode='reflect')
            filtered_matrix[:, :n, c] = uniform_filter(filtered_matrix[:, :n, c], size=filter_size, mode='reflect')
            filtered_matrix[:, -n:, c] = uniform_filter(filtered_matrix[:, -n:, c], size=filter_size, mode='reflect')

    return filtered_matrix




if __name__ == '__main__':
    pass
