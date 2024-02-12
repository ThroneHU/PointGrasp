# coding:utf-8            
# @Time    : 27/01/2024 20:08
# @Author  : Tyrone Chen HU

import os
import sys
from pathlib import Path
from glob import glob
import os.path as osp
import numpy as np

BASE_DIR = r"./utils"
ROOT_DIR = r"./"
DATASET_DIR = r"./datasets/YCBSub2"

sys.path.append(BASE_DIR)

def collect_point_label():
    pass

def find_txt_files(root_dir):
    return [str(file) for file in Path(root_dir).rglob('*.txt')]

def find_dir(root_dir):
    all_files_and_folders = os.listdir(root_dir)
    folders = [name for name in all_files_and_folders if os.path.isdir(os.path.join(root_dir, name))]
    return folders

def parse_rgb_from_int(rgb_int):
    # 按位与操作并右移来提取RGB分量
    blue = rgb_int & 0x0000ff
    green = (rgb_int & 0x00ff00) >> 8
    red = (rgb_int & 0xff0000) >> 16
    return float(red), float(green), float(blue)

def parse_label(label_file):
    with open(label_file, 'r') as file:
        data_started = False
        data_list = []
        for line in file:
            if data_started:
                values = line.strip().split()
                if len(values) == 6:
                    x, y, z, rgb, label, object_id = map(float, values)
                    r, g, b = parse_rgb_from_int(int(rgb))
                    data_list.append([x, y, z, r, g, b, label]) # xyzrgbl
            elif line.strip() == "DATA ascii":
                data_started = True
    return data_list

def main():
    mode = 'val'
    area_list = find_dir(DATASET_DIR)
    output_folder = os.path.join(ROOT_DIR, 'data/biorob_3d/'+mode)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        for area in area_list:
            pcd_root = osp.join(osp.join(DATASET_DIR, area), mode)
            pcd_list = glob(osp.join(pcd_root, '*.pcd'))
            for pcd in pcd_list:
                xyxrgbl = parse_label(pcd)
                out_name = pcd.split("\\")[-1]
                out_name = osp.join(output_folder, out_name.split(".")[0]) + '.npy'
                np.save(out_name, np.array(xyxrgbl))
                print("Finish generating npy: ", pcd)
    except:
        print(area_list, 'ERROR!!')

if __name__=='__main__':
    main()


