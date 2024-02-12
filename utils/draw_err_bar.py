# @Time    : 04/02/2024 22:41
# @Author  : Tyrone Chen HU

import numpy as np
from glob import glob
import os.path as osp
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_err_bar(csv_root, fig_name, pred_col, gt_col, save_fig=False):
    data_dict = load_csv_result(csv_root, pred_col, gt_col)
    data = pd.DataFrame(data_dict)

    plt.figure(figsize=(24, 20))
    sns.barplot(x="category", y="mean_rmse", data=data, color="lightblue")

    for i in range(data.shape[0]):
        plt.errorbar(x=i, y=data["mean_rmse"][i], yerr=data["std_rmse"][i], color='black', fmt='none', capsize=5)

    # plt.title("RMSE of Handle Objects in Grasp Point Distance Estimation")
    plt.xlabel("")
    plt.ylabel("RMSE mean values", fontsize=36, fontweight='bold')
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=50)
    # plt.xticks([])
    if save_fig:
        plt.savefig(fig_name, bbox_inches='tight')
        print("Figure output path: ", fig_name)

    plt.show()


def extract_pred_gt(csv_name, pred_col, gt_col):
    print(csv_name)
    with open(osp.join(csv_name), 'r') as file:
        csv_reader = csv.reader(file)
        # next(csv_reader)

        pred_col_list = []
        gt_col_list = []
        for row in csv_reader:
            pred_col_list.append(float(row[pred_col - 1]))
            gt_col_list.append(float(row[gt_col - 1]))

    return np.array(pred_col_list), np.array(gt_col_list)

def load_csv_result(csv_root, pred_col, gt_col):
    csv_list = glob(csv_root + '/*.csv')
    data = {"category": [], "mean_rmse": [], "std_rmse": []}

    for item in csv_list:
        name = item.split('\\')[-1]
        name = name.split('.')[0]
        data['category'].append(name[4:])
        pred_col_arr, gt_col_arr = extract_pred_gt(item, pred_col, gt_col)
        rmse = np.sqrt((pred_col_arr - gt_col_arr) ** 2)
        data['mean_rmse'].append(np.mean(rmse))
        data['std_rmse'].append(np.std(rmse))
        # data['std_rmse'].append(np.sqrt(sum((rmse - np.mean(rmse))**2) / (len(rmse) - 1)))

    return data

def main():
    simple_csv_root = './datasets/csv/simple_csv'
    complex_csv_root = './datasets/csv/complex_csv'
    out_root = r''
    simple_fig_name = 'rmse_simple.svg'
    complex_fig_name = 'rmse_complex.svg'

    draw_err_bar(simple_csv_root, osp.join(out_root, simple_fig_name), 1, 2, True)
    draw_err_bar(complex_csv_root, osp.join(out_root, complex_fig_name), 2, 3, True)

if __name__ == '__main__':
    main()
