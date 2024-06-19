# **Ev-Layout: A Large-scale Event-based Multi-modal Dataset for Indoor Layout Estimation and Tracking**

## Dataset Organization
You can download the raw dataset for EV-Layout at the following [link](https://1drv.ms/f/s!AjtGtw9IBVWZhqomhu87JboTEKcQtA?e=gHZTQW).

Raw_data contains multiple sessions, each session containing all data collected during one dynamic acquisition. It includes raw_events, images, imu, labeled_img, and illumination.
- **Raw_events:** This folder contains raw event data captured by the prophesee EVK4, with a resolution of 1280x720 pixels.

- **Images:** This folder contains image data captured by the FLIRBFS-U316S2C camera, with a resolution of 1280x720 pixels.

- **Imu:** This folder contains IMU (Inertial Measurement Unit) data collected by the WHEELTEC N100.

- **Labeled_img:** This folder includes a subset of images extracted from the Images folder, annotated using the labelme annotation tool.

```bash
-dataset-part-1
|---session_1
|      |---images
|      |---imu
|      |---labeled_img
|      |---raw_events
|      |---illumination
|---session_2
|---session_3
|......
```
## Environment Setup Instructions

### Step 1: Create a virtual environment with Python 3.8
```bash
conda create -n EV-Layout python=3.8

pip3 install --no-cache-dir opencv-python-headless tb-nightly future Cython \
matplotlib numpy scipy pyyaml gitpython seaborn pycairo open3d descartes \
shapely panda3d yacs requests scikit-image tabulate networkx tqdm wandb h5py

conda install cuda -c nvidia/label/cuda-11.3.1

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
cudatoolkit=11.3 -c pytorch

pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch1.12.1+cu113.html
torch_scatter-2.1.0+pt112cu113-cp39-cp39-linux_x86_64.whl

pip install torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch1.12.1+cu113.html
torch_sparse-0.6.16+pt112cu113-cp39-cp39-linux_x86_64.whl

pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch1.12.1+cu113.html
torch_cluster-1.6.0+pt112cu113-cp39-cp39-linux_x86_64.whl

pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch1.12.1+cu113.html
torch_spline_conv-1.2.1+pt112cu113-cp39-cp39-linux_x86_64.whl

pip install torch-geometric==2.3.0
```
### Step 2: Fix Python path
 ```bash
 source init_env.sh
 
 ./build.sh
 ```
 ### Step 3: Download EV-Layout dataset and Model Weights
Download the data for training and testing at the [link](https://1drv.ms/f/s!AjtGtw9IBVWZhqonTiZwt-3Wwh7otQ?e=ma5Qis)

Put the downloaded data in the data directory
```bash
-data
|---EV-Layout
|      |---images
|      |---train.json
|      |---test.json
```

 ### Step 4: Run inference on test set
  ```bash
python test.py --config_file ./config-files/Pred-SRW-S3D.yaml --seed 2 --model_path ./model/EV-Layout.pth
 ```
The visualization of the results will be in the outputs folder
