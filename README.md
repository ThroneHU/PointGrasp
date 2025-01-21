# [PointGrasp: Point Cloud-based Grasping for Tendon-driven Soft Robotic Glove Applications](https://github.com/ThroneHU/PointGrasp/blob/main)
---
## C. Hu, S. Lyu, E. Rho, D. Kim, S. Luo, L. Gionfrida 
---
This project is an oral presentation at BioRob2024. <br>

[(Project Page)](https://github.com/ThroneHU/PointGrasp/blob/main) [(PDF)](https://github.com/ThroneHU/PointGrasp/blob/main) [(Slides)](https://github.com/ThroneHU/PointGrasp/blob/main) [(Video)](https://github.com/ThroneHU/PointGrasp/blob/main)

![image](https://github.com/ThroneHU/PointGrasp/blob/main/figs/Fig2.svg)

### Abstract

Controlling hand exoskeletons to assist individuals with grasping tasks poses a challenge due to the difficulty in understanding user intentions. We propose that most daily grasping tasks during daily living (ADL) activities can be deduced by analyzing object geometries (simple and complex) from 3D point clouds. The study introduces PointGrasp, a real-time system designed for identifying household scenes semantically, aiming to support and enhance assistance during ADL for tailored end-to-end grasping tasks. The system comprises an RGB-D camera with an inertial measurement unit and a microprocessor integrated into a tendon-driven soft robotic glove. The RGB-D camera processes 3D scenes at a rate exceeding 30 frames per second. The proposed pipeline demonstrates an average RMSE of 0.8 ± 0.39 cm for simple and of 0.11 ± 0.06 cm for complex geometries. Within each mode, it identifies and pinpoints reachable objects. This system shows promise in end-to-end vision-driven robotic-assisted rehabilitation manual tasks. 

### Installation

1. Install requirements:
```python
pip3 install -r requirements.txt
```

2. Clone the repository using the command:
```python
git clone https://github.com/ehsanik/touchTorch
cd touchTorch
```

### PointNet++

1. Download the PointNet++ from [here](https://github.com/charlesq34/pointnet2).

2. Move the train and Inference scripts to the solver directory.

### Data Preparation

1. Annotate point cloud data using *semantic-segmentation-editor* and collect **.pcd* labels.

2. Generate **.npy* data for `train/val/` by modifying mode.
```python
cd utils
python collect_biorob_3data.py --mode train
```

Processed data will be saved in `datasets/sub_ycb/train`.

### Run
```python
python train_complex.py
python test_e2e.py
```

### Reference By
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

### Citation

If you find this project useful in your research, please consider citing:
```
@article{hu2024pointgrasp,
  title={PointGrasp: Point Cloud-based Grasping for Tendon-driven Soft Robotic Glove Applications},
  author={Hu, Chen and Lyu, Shirui and Rho, Eojin and Kim, Daekyum and Luo, Shan and Gionfrida, Letizia},
  journal={arXiv preprint arXiv:2403.12631},
  year={2024}
}
```
