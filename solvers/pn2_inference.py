# coding:utf-8            
# @Time    : 27/01/2024 18:21
# @Author  : Tyrone Chen HU

import open3d as o3d
import numpy as np
import torch
import os
import importlib
import sys
from tqdm import tqdm
from utils.HandleDataLoader import HandleDatasetTestRT, HandleDatasetTest
import time
from utils.data_process import remove_outliers_3d, pred_to_handle_pcd, arr_to_pcd
from utils.context_analysis import context_analysis, split_mulit_objects, split_plane_object
from utils.visualizer import visualize_np_points
from utils.constants import classes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = r'./'
sys.path.append(os.path.join(ROOT_DIR, 'models'))

DATA_ROOT = os.path.join(ROOT_DIR, 'datasets/biorob_3d/')
experiment_dir = os.path.join(ROOT_DIR, 'log/sem_seg/2024-01-23_19-57')

NUM_CLASSES = len(classes)
NUM_POINT = 4096
GPU = 0
BATCH_SIZE = 64
NUM_VOTES = 1

# def add_vote(vote_label_pool, point_idx, pred_label, weight):
def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            # if weight[b, n] != 0 and not np.isinf(weight[b, n]):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def inference(obj_root, obj_num_list):
    TEST_DATASET_WHOLE_SCENE = HandleDatasetTestRT(obj_root, split='test', block_points=NUM_POINT)
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    start_time = time.time()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        print('---- EVALUATION WHOLE SCENE----')
        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))
            for _ in tqdm(range(NUM_VOTES), total=NUM_VOTES):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                # batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    # batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    # batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                            batch_pred_label[0:real_batch_size, ...])
                                            # batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)
            handle_index = np.where(pred_label == 11)
            handle_points = whole_scene_data[handle_index]
            handle_points = handle_points[:, 0:3]

            for obj_num in obj_num_list:
                object_points = np.where(pred_label == obj_num)
                object_points = whole_scene_data[object_points]
                object_points = object_points[:, 0:3]

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print("Done! Elapsed time: {%.2f} m" % elapsed_time)
    return handle_points, object_points

if __name__ == '__main__':
    inference(DATA_ROOT)
